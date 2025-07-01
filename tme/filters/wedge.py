"""
Implements class Wedge and WedgeReconstructed to create Fourier
filter representations.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, Dict
from dataclasses import dataclass

import numpy as np

from ..types import NDArray
from ..backends import backend as be
from .compose import ComposableFilter
from ..matching_utils import centered
from ..rotations import euler_to_rotationmatrix
from ..parser import XMLParser, StarParser, MDOCParser
from ._utils import (
    centered_grid,
    frequency_grid_at_angle,
    compute_tilt_shape,
    crop_real_fourier,
    fftfreqn,
    shift_fourier,
    create_reconstruction_filter,
)

__all__ = ["Wedge", "WedgeReconstructed"]


@dataclass
class Wedge(ComposableFilter):
    """
    Generate wedge mask for tomographic data.
    """

    #: The shape of the reconstruction volume.
    shape: Tuple[int] = None
    #: The tilt angles.
    angles: Tuple[float] = None
    #: The weights corresponding to each tilt angle.
    weights: Tuple[float] = None
    #: Axis the plane is tilted over, defaults to 0 (x).
    tilt_axis: int = 0
    #: The projection axis, defaults to 2 (z).
    opening_axis: int = 2
    #: The type of weighting to apply, defaults to None.
    weight_type: str = None
    #: Frequency cutoff for created mask. Nyquist 0.5 by default.
    frequency_cutoff: float = 0.5
    #: The sampling rate, defaults to 1 Ångstrom / voxel.
    sampling_rate: Tuple[float] = 1

    @classmethod
    def from_file(cls, filename: str) -> "Wedge":
        """
        Generate a :py:class:`Wedge` instance by reading tilt angles and weights.
        Supported extensions are:

            +-------+---------------------------------------------------------+
            | .star | Tomostar STAR file                                      |
            +-------+---------------------------------------------------------+
            | .xml  | WARP/M XML file                                         |
            +-------+---------------------------------------------------------+
            | .mdoc | SerialEM file                                           |
            +-------+---------------------------------------------------------+
            | .*    | Tab-separated file with optional column names           |
            +-------+---------------------------------------------------------+

        Parameters
        ----------
        filename : str
            The path to the file containing tilt angles and weights.

        Returns
        -------
        :py:class:`Wedge`
           Class instance instance initialized with angles and weights from the file.
        """
        func = _from_text
        if filename.lower().endswith("xml"):
            func = _from_xml
        elif filename.lower().endswith("star"):
            func = _from_star
        elif filename.lower().endswith("mdoc"):
            func = _from_mdoc

        data = func(filename)
        angles, weights = data.get("angles", None), data.get("weights", None)
        if angles is None:
            raise ValueError(f"Could not find colum angles in {filename}")

        if weights is None:
            weights = [1] * len(angles)

        if len(weights) != len(angles):
            raise ValueError("Length of weights and angles differ.")

        return cls(
            shape=None,
            tilt_axis=0,
            opening_axis=2,
            angles=np.array(angles, dtype=np.float32),
            weights=np.array(weights, dtype=np.float32),
        )

    def __call__(self, **kwargs: Dict) -> NDArray:
        """
        Returns a Wedge stack of chosen parameters with DC component in the center.
        """
        func_args = vars(self).copy()
        func_args.update(kwargs)

        weight_types = {
            None: self.weight_angle,
            "angle": self.weight_angle,
            "relion": self.weight_relion,
            "grigorieff": self.weight_grigorieff,
        }

        weight_type = func_args.get("weight_type", None)
        if weight_type not in weight_types:
            raise ValueError(
                f"Supported weight_types are {','.join(list(weight_types.keys()))}"
            )

        if weight_type == "angle":
            func_args["weights"] = np.cos(np.radians(self.angles))

        ret = weight_types[weight_type](**func_args)

        frequency_cutoff = func_args.get("frequency_cutoff", None)
        if frequency_cutoff is not None:
            for index, angle in enumerate(func_args["angles"]):
                frequency_grid = frequency_grid_at_angle(
                    shape=func_args["shape"],
                    opening_axis=self.opening_axis,
                    tilt_axis=self.tilt_axis,
                    angle=angle,
                    sampling_rate=1,
                )
                ret[index] = np.multiply(ret[index], frequency_grid <= frequency_cutoff)

        ret = be.astype(be.to_backend_array(ret), be._float_dtype)

        return {
            "data": ret,
            "shape": func_args["shape"],
            "return_real_fourier": func_args.get("return_real_fourier", False),
            "is_multiplicative_filter": True,
        }

    @staticmethod
    def weight_angle(
        shape: Tuple[int],
        weights: Tuple[float],
        angles: Tuple[float],
        opening_axis: int,
        tilt_axis: int,
        **kwargs,
    ) -> NDArray:
        """
        Generate weighted wedges based on the cosine of the current angle.
        """
        tilt_shape = compute_tilt_shape(
            shape=shape, opening_axis=opening_axis, reduce_dim=True
        )
        wedge, wedges = np.ones(tilt_shape), np.zeros((len(angles), *tilt_shape))
        for index, angle in enumerate(angles):
            wedge.fill(weights[index])
            wedges[index] = wedge

        return wedges

    def weight_relion(
        self,
        shape: Tuple[int],
        opening_axis: int,
        tilt_axis: int,
        sampling_rate: float = 1.0,
        **kwargs,
    ) -> NDArray:
        """
        Generate weighted wedges based on the RELION 1.4 formalism, weighting each
        angle using the cosine of the angle and a Gaussian lowpass filter computed
        with respect to the exposure per angstrom.

        Returns
        -------
        NDArray
            Weighted wedges.
        """
        tilt_shape = compute_tilt_shape(
            shape=shape, opening_axis=opening_axis, reduce_dim=True
        )
        wedges = np.zeros((len(self.angles), *tilt_shape))
        for index, angle in enumerate(self.angles):
            frequency_grid = frequency_grid_at_angle(
                shape=shape,
                opening_axis=opening_axis,
                tilt_axis=tilt_axis,
                angle=angle,
                sampling_rate=sampling_rate,
            )
            sigma = np.sqrt(self.weights[index] * 4 / (8 * np.pi**2))
            sigma = -2 * np.pi**2 * sigma**2
            np.square(frequency_grid, out=frequency_grid)
            np.multiply(sigma, frequency_grid, out=frequency_grid)
            np.exp(frequency_grid, out=frequency_grid)
            np.multiply(frequency_grid, np.cos(np.radians(angle)), out=frequency_grid)
            wedges[index] = frequency_grid

        return wedges

    def weight_grigorieff(
        self,
        shape: Tuple[int],
        opening_axis: int,
        tilt_axis: int,
        amplitude: float = 0.245,
        power: float = -1.665,
        offset: float = 2.81,
        sampling_rate: float = 1.0,
        **kwargs,
    ) -> NDArray:
        """
        Generate weighted wedges based on the formalism introduced in [1]_.

        Returns
        -------
        NDArray
            Weighted wedges.

        References
        ----------
        .. [1]  Timothy Grant, Nikolaus Grigorieff (2015), eLife 4:e06980.
        """
        tilt_shape = compute_tilt_shape(
            shape=shape, opening_axis=opening_axis, reduce_dim=True
        )

        wedges = np.zeros((len(self.angles), *tilt_shape), dtype=be._float_dtype)
        for index, angle in enumerate(self.angles):
            frequency_grid = frequency_grid_at_angle(
                shape=shape,
                opening_axis=opening_axis,
                tilt_axis=tilt_axis,
                angle=angle,
                sampling_rate=sampling_rate,
            )

            with np.errstate(divide="ignore"):
                np.power(frequency_grid, power, out=frequency_grid)
                np.multiply(amplitude, frequency_grid, out=frequency_grid)
                np.add(frequency_grid, offset, out=frequency_grid)
                np.multiply(-2, frequency_grid, out=frequency_grid)
                np.divide(
                    self.weights[index],
                    frequency_grid,
                    out=frequency_grid,
                )

            wedges[index] = np.exp(frequency_grid)

        return wedges


@dataclass
class WedgeReconstructed:
    """
    Initialize :py:class:`WedgeReconstructed`.
    """

    #: The tilt angles, defaults to None.
    angles: Tuple[float] = None
    #: Weights to assign to individual wedge components. Not considered for continuous wedge
    weights: Tuple[float] = None
    #: Whether individual wedge components should be weighted.
    weight_wedge: bool = False
    #: Whether to create a continous wedge or a per-component wedge.
    create_continuous_wedge: bool = False
    #: Frequency cutoff of filter
    frequency_cutoff: float = 0.5
    #: Axis the plane is tilted over, defaults to 0 (x).
    tilt_axis: int = 0
    #: The projection axis, defaults to 2 (z).
    opening_axis: int = 2
    #: Filter window applied during reconstruction.
    reconstruction_filter: str = None

    def __post_init__(self):
        if self.create_continuous_wedge:
            self.angles = (min(self.angles), max(self.angles))

    def __call__(
        self, shape: Tuple[int], return_real_fourier: bool = False, **kwargs
    ) -> Dict:
        """
        Generate the reconstructed wedge.

        Parameters
        ----------
        shape : tuple of int
            The shape of the reconstruction volume.
        return_real_fourier : tuple of int
            Return a shape compliant with rfft, i.e., omit the negative frequencies
            terms resulting in a return shape (*shape[:-1], shape[-1]//2+1). Defaults
            to False.
        **kwargs : Dict
            Additional keyword arguments.

        Returns
        -------
        dict
            data: BackendArray
                The filter mask.
            shape: tuple of ints
                The requested filter shape
            return_real_fourier: bool
                Whether data is compliant with rfftn.
            is_multiplicative_filter: bool
                Whether the filter is multiplicative in Fourier space.
        """
        func_args = vars(self).copy()
        func_args.update(kwargs)

        func = self.step_wedge
        if func_args.get("create_continuous_wedge", False):
            func = self.continuous_wedge

        weight_wedge = func_args.get("weight_wedge", False)
        if func_args.get("wedge_weights") is None and weight_wedge:
            func_args["weights"] = np.cos(
                np.radians(be.to_numpy_array(func_args.get("angles", (0,))))
            )

        ret = func(shape=shape, **func_args)
        frequency_cutoff = func_args.get("frequency_cutoff", None)
        if frequency_cutoff is not None:
            frequency_mask = fftfreqn(
                shape=shape,
                sampling_rate=1,
                compute_euclidean_norm=True,
                shape_is_real_fourier=False,
            )
            ret = np.multiply(ret, frequency_mask <= frequency_cutoff, out=ret)

        if not weight_wedge:
            ret = (ret > 0) * 1.0

        ret = be.astype(be.to_backend_array(ret), be._float_dtype)

        ret = shift_fourier(data=ret, shape_is_real_fourier=False)

        if return_real_fourier:
            ret = crop_real_fourier(ret)

        return {
            "data": ret,
            "shape": shape,
            "return_real_fourier": return_real_fourier,
            "is_multiplicative_filter": True,
        }

    @staticmethod
    def continuous_wedge(
        shape: Tuple[int],
        angles: Tuple[float],
        opening_axis: int,
        tilt_axis: int,
        **kwargs: Dict,
    ) -> NDArray:
        """
        Generate a continous wedge mask with DC component at the center.

        Parameters
        ----------
        shape : tuple of int
            The shape of the reconstruction volume.
        angles : tuple of float
            Start and stop tilt angle.
        opening_axis : int
            The axis around which the wedge is opened.
        tilt_axis : int
            The axis along which the tilt is applied.

        Returns
        -------
        NDArray
            Wedge mask.
        """
        angles = np.abs(np.asarray(angles))
        aspect_ratio = shape[opening_axis] / shape[tilt_axis]
        angles = np.degrees(np.arctan(np.tan(np.radians(angles)) * aspect_ratio))

        start_radians = np.tan(np.radians(90 - angles[0]))
        stop_radians = np.tan(np.radians(-1 * (90 - angles[1])))

        grid = centered_grid(shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(
                grid[opening_axis] == 0,
                np.tan(np.radians(90)) + 1,
                grid[tilt_axis] / grid[opening_axis],
            )

        wedge = np.logical_or(start_radians <= ratios, stop_radians >= ratios).astype(
            np.float32
        )

        return wedge

    @staticmethod
    def step_wedge(
        shape: Tuple[int],
        angles: Tuple[float],
        opening_axis: int,
        tilt_axis: int,
        weights: Tuple[float] = None,
        reconstruction_filter: str = None,
        **kwargs: Dict,
    ) -> NDArray:
        """
        Generate a per-angle wedge shape with DC component at the center.

        Parameters
        ----------
        shape : tuple of int
            The shape of the reconstruction volume.
        angles : tuple of float
            The tilt angles.
        opening_axis : int
            The axis around which the wedge is opened.
        tilt_axis : int
            The axis along which the tilt is applied.
        reconstruction_filter : str
            Filter used during reconstruction.
        weights : tuple of float, optional
            Weights to assign to individual tilts. Defaults to 1.

        Returns
        -------
        NDArray
            Wege mask.
        """
        from ..backends import NumpyFFTWBackend

        angles = np.asarray(be.to_numpy_array(angles))

        if weights is None:
            weights = np.ones(angles.size)
        weights = np.asarray(weights)

        shape = tuple(int(x) for x in shape)
        opening_axis, tilt_axis = int(opening_axis), int(tilt_axis)

        weights = np.repeat(weights, angles.size // weights.size)
        plane = np.zeros(
            (shape[opening_axis], shape[tilt_axis] + (1 - shape[tilt_axis] % 2)),
            dtype=np.float32,
        )

        aspect_ratio = plane.shape[0] / plane.shape[1]
        angles = np.degrees(np.arctan(np.tan(np.radians(angles)) * aspect_ratio))

        rec_filter = 1
        if reconstruction_filter is not None:
            rec_filter = create_reconstruction_filter(
                plane.shape[::-1], filter_type=reconstruction_filter, tilt_angles=angles
            ).T

        subset = tuple(
            slice(None) if i != 0 else slice(x // 2, x // 2 + 1)
            for i, x in enumerate(plane.shape)
        )
        plane_rotated, wedge_volume = np.zeros_like(plane), np.zeros_like(plane)
        for index in range(angles.shape[0]):
            plane_rotated.fill(0)
            plane[subset] = 1
            rotation_matrix = euler_to_rotationmatrix((angles[index], 0))
            rotation_matrix = rotation_matrix[np.ix_((0, 1), (0, 1))]

            NumpyFFTWBackend().rigid_transform(
                arr=plane * rec_filter,
                rotation_matrix=rotation_matrix,
                out=plane_rotated,
                use_geometric_center=True,
                order=1,
            )
            wedge_volume += plane_rotated * weights[index]

        wedge_volume = centered(wedge_volume, (shape[opening_axis], shape[tilt_axis]))
        np.fmin(wedge_volume, np.max(weights), wedge_volume)

        if opening_axis > tilt_axis:
            wedge_volume = np.moveaxis(wedge_volume, 1, 0)

        reshape_dimensions = tuple(
            x if i in (opening_axis, tilt_axis) else 1 for i, x in enumerate(shape)
        )

        wedge_volume = wedge_volume.reshape(reshape_dimensions)
        tile_dimensions = np.divide(shape, reshape_dimensions).astype(int)
        wedge_volume = np.tile(wedge_volume, tile_dimensions)

        return wedge_volume


def _from_xml(filename: str, **kwargs) -> Dict:
    """
    Read tilt data from a WARP/M XML file.

    Parameters
    ----------
    filename : str
        The path to the text file.

    Returns
    -------
    Dict
        A dictionary with one key for each column.
    """
    data = XMLParser(filename)
    return {"angles": data["Angles"], "weights": data["Dose"]}


def _from_star(filename: str, **kwargs) -> Dict:
    """
    Read tilt data from a tomostar STAR file.

    Parameters
    ----------
    filename : str
        The path to the text file.

    Returns
    -------
    Dict
        A dictionary with one key for each column.
    """
    data = StarParser(filename, delimiter=None)["data_"]
    return {"angles": data["_wrpAxisAngle"], "weights": data["_wrpDose"]}


def _from_mdoc(filename: str, **kwargs) -> Dict:
    """
    Read tilt data from an MDOC file.

    Parameters
    ----------
    filename : str
        The path to the text file.

    Returns
    -------
    Dict
        A dictionary with one key for each column.
    """
    data = MDOCParser(filename)
    cumulative_exposure = np.multiply(np.add(1, data["ZValue"]), data["ExposureDose"])
    return {"angles": data["TiltAngle"], "weights": cumulative_exposure}


def _from_text(filename: str, **kwargs) -> Dict:
    """
    Read column data from a text file.

    Parameters
    ----------
    filename : str
        The path to the text file.
    delimiter : str, optional
        The delimiter used in the file, defaults to '\t'.

    Returns
    -------
    Dict
        A dictionary with one key for each column.
    """
    with open(filename, mode="r", encoding="utf-8") as infile:
        data = [x.strip() for x in infile.read().split("\n")]
        data = [x.split("\t") for x in data if len(x)]

    if "angles" in data[0]:
        headers = data.pop(0)
    else:
        if len(data[0]) != 1:
            raise ValueError(
                "Found more than one column without column names. Please add "
                "column names to your file. If you only want to specify tilt "
                "angles without column names, use a single column file."
            )
        headers = ("angles",)
    ret = {header: list(column) for header, column in zip(headers, zip(*data))}

    return ret
