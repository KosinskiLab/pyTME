""" Defines filters on tomographic tilt series.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import re
from typing import Tuple, Dict
from dataclasses import dataclass

import numpy as np

from ..types import NDArray
from ..backends import backend as be
from ..matching_utils import euler_to_rotationmatrix, centered
from ._utils import (
    centered_grid,
    frequency_grid_at_angle,
    compute_tilt_shape,
    crop_real_fourier,
    fftfreqn,
    shift_fourier,
)


def create_reconstruction_filter(
    filter_shape: Tuple[int], filter_type: str, **kwargs: Dict
):
    """Create a reconstruction filter of given filter_type.

    Parameters
    ----------
    filter_shape : tuple of int
        Shape of the returned filter
    filter_type: str
        The type of created filter, available options are:

        +---------------+----------------------------------------------------+
        | ram-lak       | Returns |w|                                        |
        +---------------+----------------------------------------------------+
        | ramp-cont     | Principles of Computerized Tomographic Imaging Avin|
        |               | ash C. Kak and Malcolm Slaney Chap 3 Eq. 61 [1]_   |
        +---------------+----------------------------------------------------+
        | ramp          | Like ramp-cont but considering tilt angles         |
        +---------------+----------------------------------------------------+
        | shepp-logan   | |w| * sinc(|w| / 2) [2]_                           |
        +---------------+----------------------------------------------------+
        | cosine        | |w| * cos(|w| * pi / 2) [2]_                       |
        +---------------+----------------------------------------------------+
        | hamming       | |w| * (.54 + .46 ( cos(|w| * pi))) [2]_            |
        +---------------+----------------------------------------------------+
    kwargs: Dict
        Keyword arguments for particular filter_types.

    Returns
    -------
    NDArray
        Reconstruction filter

    References
    ----------
    .. [1]  Principles of Computerized Tomographic Imaging Avinash C. Kak and Malcolm Slaney Chap 3 Eq. 61
    .. [2]  https://odlgroup.github.io/odl/index.html
    """
    filter_type = str(filter_type).lower()
    freq = fftfreqn(filter_shape, sampling_rate=0.5, compute_euclidean_norm=True)

    if filter_type == "ram-lak":
        ret = np.copy(freq)
    elif filter_type == "ramp-cont":
        ret, ndim = None, len(filter_shape)
        for dim, size in enumerate(filter_shape):
            n = np.concatenate(
                (
                    np.arange(1, size / 2 + 1, 2, dtype=int),
                    np.arange(size / 2 - 1, 0, -2, dtype=int),
                )
            )
            ret1d = np.zeros(size)
            ret1d[0] = 0.25
            ret1d[1::2] = -1 / (np.pi * n) ** 2
            ret1d_shape = tuple(size if i == dim else 1 for i in range(ndim))
            ret1d = ret1d.reshape(ret1d_shape)
            if ret is None:
                ret = ret1d
            else:
                ret = ret * ret1d
        ret = 2 * np.real(np.fft.fftn(ret))
    elif filter_type == "ramp":
        tilt_angles = kwargs.get("tilt_angles", False)
        if tilt_angles is False:
            raise ValueError("'ramp' filter requires specifying tilt angles.")
        size = filter_shape[0]
        ret = fftfreqn((size,), sampling_rate = 1, compute_euclidean_norm = True)
        min_increment = np.radians(np.min(np.abs(np.diff(np.sort(tilt_angles)))))
        ret *= (min_increment * size)
        np.fmin(ret, 1, out=ret)

        ret = np.tile(ret[:, np.newaxis], (1, filter_shape[1]))

    elif filter_type == "shepp-logan":
        ret = freq * np.sinc(freq / 2)
    elif filter_type == "cosine":
        ret = freq * np.cos(freq * np.pi / 2)
    elif filter_type == "hamming":
        ret = freq * (0.54 + 0.46 * np.cos(freq * np.pi))
    else:
        raise ValueError("Unsupported filter type")

    return ret


@dataclass
class ReconstructFromTilt:
    """Reconstruct a volume from a tilt series."""

    #: Shape of the reconstruction.
    shape: Tuple[int] = None
    #: Angle of each individual tilt.
    angles: Tuple[float] = None
    #: The axis around which the volume is opened.
    opening_axis: int = 0
    #: Axis the plane is tilted over.
    tilt_axis: int = 2
    #: Whether to return a share compliant with rfftn.
    return_real_fourier: bool = True
    #: Interpolation order used for rotation
    interpolation_order: int = 1
    #: Filter window applied during reconstruction.
    reconstruction_filter: str = None

    def __call__(self, **kwargs):
        func_args = vars(self).copy()
        func_args.update(kwargs)

        ret = self.reconstruct(**func_args)

        return {
            "data": ret,
            "shape": ret.shape,
            "shape_is_real_fourier": func_args["return_real_fourier"],
            "angles": func_args["angles"],
            "tilt_axis": func_args["tilt_axis"],
            "opening_axis": func_args["opening_axis"],
            "is_multiplicative_filter": False,
        }

    @staticmethod
    def reconstruct(
        data: NDArray,
        shape: Tuple[int],
        angles: Tuple[float],
        opening_axis: int,
        tilt_axis: int,
        interpolation_order: int = 1,
        return_real_fourier: bool = True,
        reconstruction_filter: str = None,
        **kwargs,
    ):
        """
        Reconstruct a volume from a tilt series.

        Parameters
        ----------
        data : NDArray
            The tilt series data.
        shape : tuple of int
            Shape of the reconstruction.
        angles : tuple of float
            Angle of each individual tilt.
        opening_axis : int
            The axis around which the volume is opened.
        tilt_axis : int
            Axis the plane is tilted over.
        interpolation_order : int, optional
            Interpolation order used for rotation, defaults to 1.
        return_real_fourier : bool, optional
           Whether to return a shape compliant with rfftn, defaults to True.
        reconstruction_filter : bool, optional
           Filter window applied during reconstruction.
           See :py:meth:`create_reconstruction_filter` for available options.

        Returns
        -------
        NDArray
            The reconstructed volume.
        """
        if data.shape == shape:
            return data

        data = be.to_backend_array(data)
        volume_temp = be.zeros(shape, dtype=be._float_dtype)
        volume_temp_rotated = be.zeros(shape, dtype=be._float_dtype)
        volume = be.zeros(shape, dtype=be._float_dtype)

        slices = tuple(slice(a//2, (a//2) + 1) for a in shape)
        subset = tuple(
            slice(None) if i != opening_axis else slices[opening_axis]
            for i in range(len(shape))
        )
        angles_loop = be.zeros(len(shape))
        wedge_dim = [x for x in data.shape]
        wedge_dim.insert(1 + opening_axis, 1)
        wedges = be.reshape(data, wedge_dim)

        rec_filter = 1
        if reconstruction_filter is not None:
            rec_filter = create_reconstruction_filter(
                filter_type=reconstruction_filter,
                filter_shape=tuple(x for x in wedges[0].shape if x != 1),
                tilt_angles=angles,
            )
            if tilt_axis > 0:
                rec_filter = rec_filter.T

            # This is most likely an upstream bug
            if tilt_axis == 1 and opening_axis == 0:
                rec_filter = rec_filter.T

            rec_filter = be.to_backend_array(rec_filter)
            rec_filter = be.reshape(rec_filter, wedges[0].shape)

        angles = be.to_backend_array(angles)
        for index in range(len(angles)):
            be.fill(angles_loop, 0)
            be.fill(volume_temp, 0)
            be.fill(volume_temp_rotated, 0)

            volume_temp[subset] = wedges[index] * rec_filter

            angles_loop[tilt_axis] = angles[index]
            angles_loop = be.roll(angles_loop, (opening_axis - 1,), axis=0)
            rotation_matrix = euler_to_rotationmatrix(be.to_numpy_array(angles_loop))
            rotation_matrix = be.to_backend_array(rotation_matrix)

            be.rigid_transform(
                arr=volume_temp,
                rotation_matrix=rotation_matrix,
                out=volume_temp_rotated,
                use_geometric_center=True,
                order=interpolation_order,
            )
            be.add(volume, volume_temp_rotated, out=volume)

        volume = shift_fourier(data=volume, shape_is_real_fourier=False)

        if return_real_fourier:
            volume = crop_real_fourier(volume)

        return volume


class Wedge:
    """
    Generate wedge mask for tomographic data.

    Parameters
    ----------
    shape : tuple of int
        The shape of the reconstruction volume.
    tilt_axis : int
        Axis the plane is tilted over.
    opening_axis : int
        The axis around which the volume is opened.
    angles : tuple of float
        The tilt angles.
    weights : tuple of float
        The weights corresponding to each tilt angle.
    weight_type : str, optional
        The type of weighting to apply, defaults to None.
    frequency_cutoff : float, optional
        Frequency cutoff for created mask. Nyquist 0.5 by default.

    Returns
    -------
    Dict
        A dictionary containing weighted wedges and related information.
    """

    def __init__(
        self,
        shape: Tuple[int],
        tilt_axis: int,
        opening_axis: int,
        angles: Tuple[float],
        weights: Tuple[float],
        weight_type: str = None,
        frequency_cutoff: float = 0.5,
    ):
        self.shape = shape
        self.tilt_axis = tilt_axis
        self.opening_axis = opening_axis
        self.angles = angles
        self.weights = weights
        self.frequency_cutoff = frequency_cutoff

    @classmethod
    def from_file(cls, filename: str) -> "Wedge":
        """
        Generate a :py:class:`Wedge` instance by reading tilt angles and weights
        from a tab-separated text file.

        Parameters
        ----------
        filename : str
            The path to the file containing tilt angles and weights.

        Returns
        -------
        :py:class:`Wedge`
           Class instance instance initialized with angles and weights from the file.
        """
        data = cls._from_text(filename)

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

    @staticmethod
    def _from_text(filename: str, delimiter="\t") -> Dict:
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

        headers = data.pop(0)
        ret = {header: list(column) for header, column in zip(headers, zip(*data))}

        return ret

    def __call__(self, **kwargs: Dict) -> NDArray:
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
            for index, angle in enumerate(self.angles):
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
            "angles": func_args["angles"],
            "tilt_axis": func_args["tilt_axis"],
            "opening_axis": func_args["opening_axis"],
            "sampling_rate": func_args.get("sampling_rate", 1),
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

    def weight_relion(self,
        shape: Tuple[int],
        opening_axis: int,
        tilt_axis: int,
        **kwargs
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
                sampling_rate=1,
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
                sampling_rate=1,
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


class WedgeReconstructed:
    """
    Initialize :py:class:`WedgeReconstructed`.

    Parameters
    ----------
    angles :tuple of float, optional
        The tilt angles, defaults to None.
    opening_axis : int, optional
        The axis around which the wedge is opened, defaults to 0.
    tilt_axis : int, optional
        The axis along which the tilt is applied, defaults to 2.
    weights : tuple of float, optional
        Weights to assign to individual wedge components.
    weight_wedge : bool, optional
        Whether individual wedge components should be weighted. If True and weights
        is None, uses the cosine of the angle otherwise weights.
    create_continuous_wedge: bool, optional
        Whether to create a continous wedge or a per-component wedge. Weights are only
        considered for non-continuous wedges.
    frequency_cutoff : float, optional
        Filter window applied during reconstruction.
    **kwargs : Dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        angles: Tuple[float] = None,
        opening_axis: int = 0,
        tilt_axis: int = 2,
        weights : Tuple[float] = None,
        weight_wedge: bool = False,
        create_continuous_wedge: bool = False,
        frequency_cutoff: float = 0.5,
        reconstruction_filter: str = None,
        **kwargs: Dict,
    ):
        self.angles = angles
        self.opening_axis = opening_axis
        self.tilt_axis = tilt_axis
        self.weights = weights
        self.weight_wedge = weight_wedge
        self.reconstruction_filter = reconstruction_filter
        self.create_continuous_wedge = create_continuous_wedge
        self.frequency_cutoff = frequency_cutoff

    def __call__(self, shape: Tuple[int], **kwargs: Dict) -> Dict:
        """
        Generate the reconstructed wedge.

        Parameters
        ----------
        shape : tuple of int
            The shape of the reconstruction volume.
        **kwargs : Dict
            Additional keyword arguments.

        Returns
        -------
        Dict
            A dictionary containing the reconstructed wedge and related information.
        """
        func_args = vars(self).copy()
        func_args.update(kwargs)

        if kwargs.get("is_fourier_shape", False):
            print("Cannot create continuous wedge mask based on real fourier shape.")

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
        if func_args.get("return_real_fourier", False):
            ret = crop_real_fourier(ret)

        return {
            "data": ret,
            "shape_is_real_fourier": func_args["return_real_fourier"],
            "shape": ret.shape,
            "tilt_axis": func_args["tilt_axis"],
            "opening_axis": func_args["opening_axis"],
            "is_multiplicative_filter": True,
            "angles": func_args["angles"],
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
        reconstruction_filter : str = None,
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
            dtype=np.float32
        )

        # plane = np.zeros((shape[opening_axis], int(2 * np.max(shape)) + 1), dtype=np.float32)

        rec_filter = 1
        if reconstruction_filter is not None:
            rec_filter = create_reconstruction_filter(
                plane.shape[::-1], filter_type = reconstruction_filter, tilt_angles = angles
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


@dataclass
class CTF:
    """
    Representation of a contrast transfer function (CTF) [1]_.

    References
    ----------
    .. [1]  CTFFIND4: Fast and accurate defocus estimation from electron micrographs.
            Alexis Rohou and Nikolaus Grigorieff. Journal of Structural Biology 2015.
    """

    #: The shape of the to-be reconstructed volume.
    shape: Tuple[int]
    #: The defocus value in x direction.
    defocus_x: float
    #: The tilt angles.
    angles: Tuple[float] = None
    #: The axis around which the wedge is opened, defaults to None.
    opening_axis: int = None
    #: The axis along which the tilt is applied, defaults to None.
    tilt_axis: int = None
    #: Whether to correct defocus gradient, defaults to False.
    correct_defocus_gradient: bool = False
    #: The sampling rate, defaults to 1.
    sampling_rate: Tuple[float] = 1
    #: The acceleration voltage in Volts, defaults to 300e3.
    acceleration_voltage: float = 300e3
    #: The spherical aberration coefficient, defaults to 2.7e7.
    spherical_aberration: float = 2.7e7
    #: The amplitude contrast, defaults to 0.07.
    amplitude_contrast: float = 0.07
    #: The phase shift, defaults to 0.
    phase_shift: float = 0
    #: The defocus angle, defaults to 0.
    defocus_angle: float = 0
    #: The defocus value in y direction, defaults to None.
    defocus_y: float = None
    #: Whether the returned CTF should be phase-flipped.
    flip_phase: bool = True
    #: Whether to return a format compliant with rfft. Only relevant for single angles.
    return_real_fourier: bool = False

    @classmethod
    def from_file(cls, filename: str) -> "CTF":
        """
        Initialize :py:class:`CTF` from file.

        Parameters
        ----------
        filename : str
            The path to a file with ctf parameters. Supports the following formats:
            - CTFFIND4
        """
        data = cls._from_ctffind(filename=filename)

        return cls(
            shape=None,
            angles=None,
            defocus_x=data["defocus_1"],
            sampling_rate=data["pixel_size"],
            acceleration_voltage=data["acceleration_voltage"],
            spherical_aberration=data["spherical_aberration"],
            amplitude_contrast=data["amplitude_contrast"],
            phase_shift=data["additional_phase_shift"],
            defocus_angle=np.degrees(data["azimuth_astigmatism"]),
            defocus_y=data["defocus_2"],
        )

    @staticmethod
    def _from_ctffind(filename: str):
        parameter_regex = {
            "pixel_size": r"Pixel size: ([0-9.]+) Angstroms",
            "acceleration_voltage": r"acceleration voltage: ([0-9.]+) keV",
            "spherical_aberration": r"spherical aberration: ([0-9.]+) mm",
            "amplitude_contrast": r"amplitude contrast: ([0-9.]+)",
        }

        with open(filename, mode="r", encoding="utf-8") as infile:
            lines = [x.strip() for x in infile.read().split("\n")]
            lines = [x for x in lines if len(x)]

        def _screen_params(line, params, output):
            for parameter, regex_pattern in parameter_regex.items():
                match = re.search(regex_pattern, line)
                if match:
                    output[parameter] = float(match.group(1))

        columns = {
            "micrograph_number": 0,
            "defocus_1": 1,
            "defocus_2": 2,
            "azimuth_astigmatism": 3,
            "additional_phase_shift": 4,
            "cross_correlation": 5,
            "spacing": 6,
        }
        output = {k: [] for k in columns.keys()}
        for line in lines:
            if line.startswith("#"):
                _screen_params(line, params=parameter_regex, output=output)
                continue

            values = line.split()
            for key, value in columns.items():
                output[key].append(float(values[value]))

        for key in columns:
            output[key] = np.array(output[key])

        return output

    def __post_init__(self):
        self.defocus_angle = np.radians(self.defocus_angle)

    def _compute_electron_wavelength(self, acceleration_voltage: int = None):
        """Computes the wavelength of an electron in angstrom."""

        if acceleration_voltage is None:
            acceleration_voltage = self.acceleration_voltage

        # Physical constants expressed in SI units
        planck_constant = 6.62606896e-34
        electron_charge = 1.60217646e-19
        electron_mass = 9.10938215e-31
        light_velocity = 299792458

        energy = electron_charge * acceleration_voltage
        denominator = energy**2
        denominator += 2 * energy * electron_mass * light_velocity**2
        electron_wavelength = np.divide(
            planck_constant * light_velocity, np.sqrt(denominator)
        )
        # Convert to Ångstrom
        electron_wavelength *= 1e10
        return electron_wavelength

    def __call__(self, **kwargs) -> NDArray:
        func_args = vars(self).copy()
        func_args.update(kwargs)

        if len(func_args["angles"]) != len(func_args["defocus_x"]):
            func_args["angles"] = self.angles
            func_args["return_real_fourier"] = False
            func_args["tilt_axis"] = None
            func_args["opening_axis"] = None

        ret = self.weight(**func_args)
        ret = be.astype(be.to_backend_array(ret), be._float_dtype)
        return {
            "data": ret,
            "angles": func_args["angles"],
            "tilt_axis": func_args["tilt_axis"],
            "opening_axis": func_args["opening_axis"],
            "is_multiplicative_filter": True,
        }

    def weight(
        self,
        shape: Tuple[int],
        defocus_x: Tuple[float],
        angles: Tuple[float],
        opening_axis: int = None,
        tilt_axis: int = None,
        amplitude_contrast: float = 0.07,
        phase_shift: Tuple[float] = 0,
        defocus_angle: Tuple[float] = 0,
        defocus_y: Tuple[float] = None,
        correct_defocus_gradient: bool = False,
        sampling_rate: Tuple[float] = 1,
        acceleration_voltage: float = 300e3,
        spherical_aberration: float = 2.7e3,
        flip_phase: bool = True,
        return_real_fourier: bool = False,
        **kwargs: Dict,
    ) -> NDArray:
        """
        Compute the CTF weight tilt stack.

        Parameters
        ----------
        shape : tuple of int
            The shape of the CTF.
        defocus_x : tuple of float
            The defocus value in x direction.
        angles : tuple of float
            The tilt angles.
        opening_axis : int, optional
            The axis around which the wedge is opened, defaults to None.
        tilt_axis : int, optional
            The axis along which the tilt is applied, defaults to None.
        amplitude_contrast : float, optional
            The amplitude contrast, defaults to 0.07.
        phase_shift : tuple of float, optional
            The phase shift, defaults to 0.
        defocus_angle : tuple of float, optional
            The defocus angle, defaults to 0.
        defocus_y : tuple of float, optional
            The defocus value in y direction, defaults to None.
        correct_defocus_gradient : bool, optional
            Whether to correct defocus gradient, defaults to False.
        sampling_rate : tuple of float, optional
            The sampling rate, defaults to 1.
        acceleration_voltage : float, optional
            The acceleration voltage in electron microscopy, defaults to 300e3.
        spherical_aberration : float, optional
            The spherical aberration coefficient, defaults to 2.7e3.
        flip_phase : bool, optional
            Whether the returned CTF should be phase-flipped.
        **kwargs : Dict
            Additional keyword arguments.

        Returns
        -------
        NDArray
            A stack containing the CTF weight.
        """
        defoci_x = np.atleast_1d(defocus_x)
        defoci_y = np.atleast_1d(defocus_y)
        phase_shift = np.atleast_1d(phase_shift)
        angles = np.atleast_1d(angles)
        defocus_angle = np.atleast_1d(defocus_angle)

        sampling_rate = np.max(sampling_rate)
        tilt_shape = compute_tilt_shape(
            shape=shape, opening_axis=opening_axis, reduce_dim=True
        )
        stack = np.zeros((len(angles), *tilt_shape))

        correct_defocus_gradient &= len(shape) == 3
        correct_defocus_gradient &= tilt_axis is not None
        correct_defocus_gradient &= opening_axis is not None

        spherical_aberration /= sampling_rate
        electron_wavelength = self._compute_electron_wavelength() / sampling_rate
        for index, angle in enumerate(angles):
            defocus_x, defocus_y = defoci_x[index], defoci_y[index]

            defocus_x = defocus_x / sampling_rate if defocus_x is not None else None
            defocus_y = defocus_y / sampling_rate if defocus_y is not None else None
            if correct_defocus_gradient or defocus_y is not None:
                grid = fftfreqn(
                    shape=shape,
                    sampling_rate=be.divide(sampling_rate, shape),
                    return_sparse_grid=True,
                )

            # This should be done after defocus_x computation
            if correct_defocus_gradient:
                angle_rad = np.radians(angle)

                defocus_gradient = np.multiply(grid[1], np.sin(angle_rad))
                remaining_axis = tuple(
                    i for i in range(len(shape)) if i not in (opening_axis, tilt_axis)
                )[0]

                if tilt_axis > remaining_axis:
                    defocus_x = np.add(defocus_x, defocus_gradient)
                elif tilt_axis < remaining_axis and defocus_y is not None:
                    defocus_y = np.add(defocus_y, defocus_gradient.T)

            if defocus_y is not None:
                defocus_sum = np.add(defocus_x, defocus_y)
                defocus_difference = np.subtract(defocus_x, defocus_y)
                angular_grid = np.arctan2(grid[0], grid[1])
                defocus_difference *= np.cos(2 * (angular_grid - defocus_angle[index]))
                defocus_x = np.add(defocus_sum, defocus_difference)
                defocus_x *= 0.5

            frequency_grid = frequency_grid_at_angle(
                shape=shape,
                opening_axis=opening_axis,
                tilt_axis=tilt_axis,
                angle=angle,
                sampling_rate=1,
            )
            frequency_mask = frequency_grid < 0.5
            np.square(frequency_grid, out=frequency_grid)

            electron_aberration = spherical_aberration * electron_wavelength**2
            chi = defocus_x - 0.5 * electron_aberration * frequency_grid
            np.multiply(chi, np.pi * electron_wavelength, out=chi)
            np.multiply(chi, frequency_grid, out=chi)

            chi += phase_shift[index]
            chi += np.arctan(
                np.divide(
                    amplitude_contrast,
                    np.sqrt(1 - np.square(amplitude_contrast)),
                )
            )
            np.sin(-chi, out=chi)
            np.multiply(chi, frequency_mask, out=chi)
            stack[index] = chi

        # Avoid contrast inversion
        np.negative(stack, out=stack)
        if flip_phase:
            np.abs(stack, out=stack)

        stack = np.squeeze(stack)
        stack = be.to_backend_array(stack)

        if len(angles) == 1:
            stack = shift_fourier(data=stack, shape_is_real_fourier=False)
            if return_real_fourier:
                stack = crop_real_fourier(stack)

        return stack
