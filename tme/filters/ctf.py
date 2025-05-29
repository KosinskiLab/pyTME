""" Implements class CTF to create Fourier filter representations.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import re
import warnings
from typing import Tuple, Dict
from dataclasses import dataclass

import numpy as np

from ..types import NDArray
from ..parser import StarParser
from ..backends import backend as be
from .compose import ComposableFilter
from ._utils import (
    frequency_grid_at_angle,
    compute_tilt_shape,
    crop_real_fourier,
    fftfreqn,
    shift_fourier,
)

__all__ = ["CTF"]


@dataclass
class CTF(ComposableFilter):
    """
    Generate a contrast transfer function mask.

    References
    ----------
    .. [1]  CTFFIND4: Fast and accurate defocus estimation from electron micrographs.
            Alexis Rohou and Nikolaus Grigorieff. Journal of Structural Biology 2015.
    """

    #: The shape of the to-be reconstructed volume.
    shape: Tuple[int] = None
    #: The defocus value in x direction.
    defocus_x: float = None
    #: The tilt angles.
    angles: Tuple[float] = None
    #: The axis around which the wedge is opened, defaults to None.
    opening_axis: int = None
    #: The axis along which the tilt is applied, defaults to None.
    tilt_axis: int = None
    #: Whether to correct defocus gradient, defaults to False.
    correct_defocus_gradient: bool = False
    #: The sampling rate, defaults to 1 Angstrom / Voxel.
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
    #: Whether the output should not be used for n+1 dimensional reconstruction
    no_reconstruction: bool = True

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
        if filename.lower().endswith("star"):
            data = cls._from_gctf(filename=filename)
        else:
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

    @staticmethod
    def _from_gctf(filename: str):
        parser = StarParser(filename)
        ctf_data = parser["data_"]

        mapping = {
            "defocus_1": ("_rlnDefocusU", float),
            "defocus_2": ("_rlnDefocusV", float),
            "pixel_size": ("_rlnDetectorPixelSize", float),
            "acceleration_voltage": ("_rlnVoltage", float),
            "spherical_aberration": ("_rlnSphericalAberration", float),
            "amplitude_contrast": ("_rlnAmplitudeContrast", float),
            "additional_phase_shift": (None, float),
            "azimuth_astigmatism": ("_rlnDefocusAngle", float),
        }
        output = {}
        for out_key, (key, key_dtype) in mapping.items():
            if key not in ctf_data and key is not None:
                warnings.warn(f"ctf_data is missing key {key}.")

            key_value = ctf_data.get(key, [0])
            output[out_key] = [key_dtype(x) for x in key_value]

        longest_key = max(map(len, output.values()))
        output = {k: v * longest_key if len(v) == 1 else v for k, v in output.items()}
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

    @staticmethod
    def _pad_to_length(arr, length: int):
        ret = np.atleast_1d(arr)
        return np.repeat(ret, length // ret.size)

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
        no_reconstruction: bool = True,
        cutoff_frequency: float = 0.5,
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
        angles = np.atleast_1d(angles)
        defoci_x = self._pad_to_length(defocus_x, angles.size)
        defoci_y = self._pad_to_length(defocus_y, angles.size)
        phase_shift = self._pad_to_length(phase_shift, angles.size)
        defocus_angle = self._pad_to_length(defocus_angle, angles.size)
        spherical_aberration = self._pad_to_length(spherical_aberration, angles.size)
        amplitude_contrast = self._pad_to_length(amplitude_contrast, angles.size)

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
        electron_aberration = spherical_aberration * electron_wavelength**2

        for index, angle in enumerate(angles):
            defocus_x, defocus_y = defoci_x[index], defoci_y[index]

            defocus_x = defocus_x / sampling_rate if defocus_x is not None else None
            defocus_y = defocus_y / sampling_rate if defocus_y is not None else None

            if correct_defocus_gradient or defocus_y is not None:
                grid_shape = shape
                sampling = be.divide(sampling_rate, be.to_backend_array(shape))
                sampling = tuple(float(x) for x in sampling)
                if not no_reconstruction:
                    grid_shape = tilt_shape
                    sampling = tuple(
                        x for i, x in enumerate(sampling) if i != opening_axis
                    )

                grid = fftfreqn(
                    shape=grid_shape,
                    sampling_rate=sampling,
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

            # 0.5 * (dx + dy) + cos(2 * (azimuth - astigmatism) * (dx - dy))
            if defocus_y is not None:
                defocus_sum = np.add(defocus_x, defocus_y)
                defocus_difference = np.subtract(defocus_x, defocus_y)

                angular_grid = np.arctan2(grid[1], grid[0])
                defocus_difference = np.multiply(
                    defocus_difference,
                    np.cos(2 * (angular_grid - defocus_angle[index])),
                )
                defocus_x = np.add(defocus_sum, defocus_difference)
                defocus_x *= 0.5

            frequency_grid = frequency_grid_at_angle(
                shape=shape,
                opening_axis=opening_axis,
                tilt_axis=tilt_axis,
                angle=angle,
                sampling_rate=1,
            )
            frequency_mask = frequency_grid < cutoff_frequency

            # k^2*π*λ(dx - 0.5 * sph_abb * λ^2 * k^2) + phase_shift + ampl_contrast_term)
            np.square(frequency_grid, out=frequency_grid)
            chi = defocus_x - 0.5 * electron_aberration[index] * frequency_grid
            np.multiply(chi, np.pi * electron_wavelength, out=chi)
            np.multiply(chi, frequency_grid, out=chi)
            chi += phase_shift[index]
            chi += np.arctan(
                np.divide(
                    amplitude_contrast[index],
                    np.sqrt(1 - np.square(amplitude_contrast[index])),
                )
            )
            np.sin(-chi, out=chi)
            np.multiply(chi, frequency_mask, out=chi)

            if no_reconstruction:
                chi = shift_fourier(data=chi, shape_is_real_fourier=False)

            stack[index] = chi

        # Avoid contrast inversion
        np.negative(stack, out=stack)
        if flip_phase:
            np.abs(stack, out=stack)

        stack = be.to_backend_array(np.squeeze(stack))
        if no_reconstruction and return_real_fourier:
            stack = crop_real_fourier(stack)

        return stack
