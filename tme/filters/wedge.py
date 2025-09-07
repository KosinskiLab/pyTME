"""
Implements class Wedge and WedgeReconstructed.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, Dict
from dataclasses import dataclass

import numpy as np

from ..types import NDArray
from ..backends import backend as be
from .compose import ComposableFilter
from ..matching_utils import center_slice
from ..parser import XMLParser, StarParser, MDOCParser
from ._utils import (
    centered_grid,
    frequency_grid_at_angle,
    compute_tilt_shape,
    fftfreqn,
    create_reconstruction_filter,
    shift_fourier,
)

__all__ = ["Wedge", "WedgeReconstructed"]


@dataclass
class Wedge(ComposableFilter):
    """
    Create per-tilt wedge mask for tomographic data.
    """

    #: Tilt angles in degrees.
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
    def from_file(cls, filename: str, **kwargs) -> "Wedge":
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
            tilt_axis=0,
            opening_axis=2,
            angles=np.array(angles, dtype=np.float32),
            weights=np.array(weights, dtype=np.float32),
            **kwargs,
        )

    def _evaluate(self, shape: Tuple[int, ...], **kwargs: Dict) -> NDArray:
        """Returns a Wedge stack of chosen parameters."""
        weight_types = {
            None: weight_uniform,
            "angle": weight_angle,
            "relion": weight_relion,
            "grigorieff": weight_grigorieff,
        }

        weight_type = kwargs.get("weight_type", None)
        if weight_type not in weight_types:
            raise ValueError(
                f"Supported weight_types are {','.join(list(weight_types.keys()))}"
            )

        if weight_type == "angle":
            kwargs["weights"] = np.cos(np.radians(self.angles))

        ret = weight_types[weight_type](shape=shape, **kwargs)

        frequency_cutoff = kwargs.get("frequency_cutoff", None)
        if frequency_cutoff is not None:
            for index, angle in enumerate(kwargs["angles"]):
                frequency_grid = frequency_grid_at_angle(
                    shape=shape,
                    opening_axis=kwargs["opening_axis"],
                    tilt_axis=kwargs["tilt_axis"],
                    angle=angle,
                    sampling_rate=1,
                )
                ret[index] = np.multiply(ret[index], frequency_grid <= frequency_cutoff)

        ret = be.astype(be.to_backend_array(ret), be._float_dtype)
        return {"data": ret, "shape": shape}


@dataclass
class WedgeReconstructed(Wedge):
    """
    Create wedge mask for tomographic reconstructions.
    """

    #: Tilt angles in degrees.
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

    def _evaluate(self, shape: Tuple[int, ...], **kwargs) -> Dict:
        """
        Generate a reconstructed wedge.

        Parameters
        ----------
        shape : tuple of int
            The shape to build the filter for.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            data: BackendArray
                The filter mask.
            shape: tuple of ints
                The requested filter shape
        """
        func = step_wedge
        angles = kwargs.pop("angles", (0,))
        if kwargs.get("create_continuous_wedge", False):
            func = continuous_wedge
            if len(angles) != 2:
                angles = (min(angles), max(angles))

        weight_wedge = kwargs.get("weight_wedge", False)
        if kwargs.get("wedge_weights") is None and weight_wedge:
            kwargs["weights"] = np.cos(np.radians(be.to_numpy_array(angles)))
        ret = func(shape=shape, angles=angles, **kwargs)

        # Move DC component to origin
        ret = shift_fourier(ret, shape_is_real_fourier=False)
        frequency_cutoff = kwargs.get("frequency_cutoff", None)
        if frequency_cutoff is not None:
            frequency_mask = (
                fftfreqn(
                    shape=shape,
                    sampling_rate=1,
                    compute_euclidean_norm=True,
                    shape_is_real_fourier=False,
                    fftshift=False,
                )
                <= frequency_cutoff
            )
            ret = np.multiply(ret, frequency_mask, out=ret)

        if not weight_wedge:
            ret = (ret > 0) * 1.0
        ret = be.astype(be.to_backend_array(ret), be._float_dtype)
        return {"data": ret, "shape": shape}


def continuous_wedge(
    shape: Tuple[int, ...],
    angles: Tuple[float, float],
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
        Start and stop tilt angle in degrees.
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


def step_wedge(
    shape: Tuple[int, ...],
    angles: Tuple[float, ...],
    opening_axis: int,
    tilt_axis: int,
    weights: Tuple[float, ...] = None,
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
        The tilt angles in degrees.
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

        angle_rad = np.radians(angles[index])
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
        # We want a push rotation but rigid transform assumes pull
        NumpyFFTWBackend().rigid_transform(
            arr=plane * rec_filter,
            rotation_matrix=rotation_matrix.T,
            out=plane_rotated,
            use_geometric_center=True,
            order=1,
        )
        wedge_volume += plane_rotated * weights[index]

    subset = center_slice(wedge_volume.shape, (shape[opening_axis], shape[tilt_axis]))
    wedge_volume = wedge_volume[subset]

    np.fmin(wedge_volume, np.max(weights), wedge_volume)

    if opening_axis > tilt_axis:
        wedge_volume = np.moveaxis(wedge_volume, 1, 0)

    reshape_dimensions = tuple(
        x if i in (opening_axis, tilt_axis) else 1 for i, x in enumerate(shape)
    )

    wedge_volume = wedge_volume.reshape(reshape_dimensions)
    tile_dimensions = np.divide(shape, reshape_dimensions).astype(int)
    return np.tile(wedge_volume, tile_dimensions)


def weight_uniform(angles: Tuple[float, ...], *args, **kwargs) -> NDArray:
    """
    Generate uniform weighted wedges.
    """
    return weight_angle(angles=np.zeros_like(angles), *args, **kwargs)


def weight_angle(
    shape: Tuple[int, ...],
    angles: Tuple[float, ...],
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
    wedges = np.zeros((len(angles), *tilt_shape))
    for index, angle in enumerate(angles):
        wedges[index] = np.cos(np.radians(angle))
    return wedges


def weight_relion(
    shape: Tuple[int, ...],
    angles: Tuple[float, ...],
    weights: Tuple[float, ...],
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
    wedges = np.zeros((len(angles), *tilt_shape))
    for index, angle in enumerate(angles):
        frequency_grid = frequency_grid_at_angle(
            shape=shape,
            opening_axis=opening_axis,
            tilt_axis=tilt_axis,
            angle=angle,
            sampling_rate=sampling_rate,
            fftshift=False,
        )
        sigma = np.sqrt(weights[index] * 4 / (8 * np.pi**2))
        sigma = -2 * np.pi**2 * sigma**2
        frequency_grid = np.square(frequency_grid, out=frequency_grid)
        frequency_grid = np.multiply(sigma, frequency_grid, out=frequency_grid)
        frequency_grid = np.exp(frequency_grid, out=frequency_grid)
        wedges[index] = np.multiply(frequency_grid, np.cos(np.radians(angle)))

    return wedges


def weight_grigorieff(
    shape: Tuple[int, ...],
    angles: Tuple[float, ...],
    weights: Tuple[float, ...],
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

    wedges = np.zeros((len(angles), *tilt_shape), dtype=be._float_dtype)
    for index, angle in enumerate(angles):
        frequency_grid = frequency_grid_at_angle(
            shape=shape,
            opening_axis=opening_axis,
            tilt_axis=tilt_axis,
            angle=angle,
            sampling_rate=sampling_rate,
            fftshift=False,
        )

        with np.errstate(divide="ignore"):
            np.power(frequency_grid, power, out=frequency_grid)
            np.multiply(amplitude, frequency_grid, out=frequency_grid)
            np.add(frequency_grid, offset, out=frequency_grid)
            np.multiply(-2, frequency_grid, out=frequency_grid)
            np.divide(weights[index], frequency_grid, out=frequency_grid)
        wedges[index] = np.exp(frequency_grid)

    return wedges


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
    Read tilt data from a STAR file.

    Parameters
    ----------
    filename : str
        The path to the text file.

    Returns
    -------
    Dict
        A dictionary with one key for each column.
    """
    data = StarParser(filename, delimiter=None)
    if "data_stopgap_wedgelist" in data:
        angles = data["data_stopgap_wedgelist"]["_tilt_angle"]
        weights = data["data_stopgap_wedgelist"]["_exposure"]
    else:
        angles = data["data_"]["_wrpAxisAngle"]
        weights = data["data_"]["_wrpDose"]
    return {"angles": angles, "weights": weights}


def _from_mdoc(filename: str, **kwargs) -> Dict:
    """
    Read tilt data from a SerialEM MDOC file.

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
