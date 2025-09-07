"""
Implements class CTF and CTFReconstruced.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import re
from typing import Tuple, Dict
from dataclasses import dataclass

import numpy as np

from ..types import NDArray
from ..backends import backend as be
from .compose import ComposableFilter
from ..parser import StarParser, XMLParser, MDOCParser
from ._utils import (
    frequency_grid_at_angle,
    compute_tilt_shape,
    fftfreqn,
    pad_to_length,
)

__all__ = ["CTF", "CTFReconstructed", "create_ctf"]


@dataclass
class CTF(ComposableFilter):
    """
    Generate per-tilt contrast transfer function filter.
    """

    #: The defocus in x direction (in units of sampling rate).
    defocus_x: Tuple[float] = None
    #: The tilt angles in degrees.
    angles: Tuple[float] = None
    #: The microscope projection axis, defaults to 2 (z).
    opening_axis: int = 2
    #: The axis along which the tilt is applied, defaults to 0 (x).
    tilt_axis: int = 0
    #: The sampling rate, defaults to 1 Ångstrom / voxel.
    sampling_rate: Tuple[float] = 1
    #: The acceleration voltage in Volts, defaults to 300e3.
    acceleration_voltage: Tuple[float] = 300e3
    #: The spherical aberration, defaults to 2.7e7 (in units of sampling rate).
    spherical_aberration: Tuple[float] = 2.7e7
    #: The amplitude contrast, defaults to 0.07.
    amplitude_contrast: Tuple[float] = 0.07
    #: The phase shift in radians, defaults to 0.
    phase_shift: Tuple[float] = 0
    #: The defocus angle in radians, defaults to 0.
    defocus_angle: Tuple[float] = 0
    #: The defocus value in y direction, defaults to None (in units of sampling rate).
    defocus_y: Tuple[float] = None
    #: Whether the returned CTF should be phase-flipped, defaults to True.
    flip_phase: bool = True

    @classmethod
    def from_file(cls, filename: str, **kwargs) -> "CTF":
        """
        Initialize :py:class:`CTF` from file.

        Parameters
        ----------
        filename : str
            The path to a file with ctf parameters. Supports extensions are:

            +-------+---------------------------------------------------------+
            | .star | GCTF file                                               |
            +-------+---------------------------------------------------------+
            | .xml  | WARP/M XML file                                         |
            +-------+---------------------------------------------------------+
            | .mdoc | SerialEM file                                           |
            +-------+---------------------------------------------------------+
            | .*    | CTFFIND4 file                                           |
            +-------+---------------------------------------------------------+
        **kwargs : optional
            Overwrite fields that cannot be extracted from input file.
        """
        func = _from_ctffind
        if filename.lower().endswith("star"):
            func = _from_star
        elif filename.lower().endswith("xml"):
            func = _from_xml
        elif filename.lower().endswith("mdoc"):
            func = _from_mdoc

        data = func(filename=filename)

        # Pixel size needs to be overwritten by pixel size the ctf is generated for
        init_kwargs = {
            "angles": data.get("angles", None),
            "defocus_x": data["defocus_1"],
            "sampling_rate": data["pixel_size"],
            "acceleration_voltage": np.multiply(data["acceleration_voltage"], 1e3),
            "spherical_aberration": data.get("spherical_aberration"),
            "amplitude_contrast": data.get("amplitude_contrast"),
            "phase_shift": data.get("additional_phase_shift"),
            "defocus_angle": data.get("azimuth_astigmatism"),
            "defocus_y": data["defocus_2"],
        }
        for k, v in kwargs.items():
            if k in init_kwargs and init_kwargs.get(k) is None:
                init_kwargs[k] = v
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        # Moved format conversion from __post__init
        if "phase_shift" in init_kwargs:
            init_kwargs["phase_shift"] = np.radians(init_kwargs["phase_shift"])
        if "defocus_angle" in init_kwargs:
            init_kwargs["defocus_angle"] = np.radians(init_kwargs["defocus_angle"])
        return cls(**init_kwargs)

    def _evaluate(
        self,
        shape: Tuple[int, ...],
        defocus_x: Tuple[float],
        angles: Tuple[float],
        opening_axis: int = 2,
        tilt_axis: int = 0,
        amplitude_contrast: Tuple[float] = 0.07,
        phase_shift: Tuple[float] = 0,
        defocus_angle: Tuple[float] = 0,
        defocus_y: Tuple[float] = None,
        sampling_rate: Tuple[float] = 1,
        acceleration_voltage: float = 300e3,
        spherical_aberration: float = 2.7e7,
        flip_phase: bool = True,
        cutoff_frequency: float = 0.5,
        **kwargs: Dict,
    ) -> Dict:
        """
        Compute the CTF weight tilt stack.

        Parameters
        ----------
        shape : tuple of int
            The shape of the CTF.
        defocus_x : tuple of float
            Defocus along the first principal axis in spatial units of sampling rate.
        angles : tuple of float
            The tilt angles in degrees.
        opening_axis : int, optional
            The axis around which the wedge is opened, defaults to 2.
        tilt_axis : int, optional
            The axis along which the tilt is applied, defaults to 0.
        amplitude_contrast : tuple of float, optional
            Amplitude contrast of microscope, defaults to 0.07.
        phase_shift : tuple of float, optional
           CTF phase shift in radians, defaults to 0.
        defocus_angle : tuple of float, optional
            Astigmatism angle in radians, defaults to 0.
        defocus_y : tuple of float, optional
            Defocus along the second principal axis in spatial units of sampling rate.
        sampling_rate : tuple of float, optional
            The sampling rate, defaults to 1.
        acceleration_voltage : float, optional
            The acceleration voltage in electron microscopy, defaults to 300e3.
        spherical_aberration : float, optional
            Spherical aberration of microscope in units of sampling rate.
        flip_phase : bool, optional
            Whether the returned CTF should be phase-flipped, defaults to True.
        **kwargs : Dict
            Additional keyword arguments.
        """
        angles = np.atleast_1d(angles)
        defoci_x = pad_to_length(defocus_x, angles.size)
        defoci_y = pad_to_length(defocus_y, angles.size)
        phase_shift = pad_to_length(phase_shift, angles.size)
        defocus_angle = pad_to_length(defocus_angle, angles.size)
        spherical_aberration = pad_to_length(spherical_aberration, angles.size)
        amplitude_contrast = pad_to_length(amplitude_contrast, angles.size)
        acceleration_voltage = pad_to_length(acceleration_voltage, angles.size)

        sampling_rate = np.max(sampling_rate)
        ctf_shape = compute_tilt_shape(
            shape=shape, opening_axis=opening_axis, reduce_dim=True
        )
        stack = np.zeros((len(angles), *ctf_shape))

        # Shift tilt axis forward
        corrected_tilt_axis = tilt_axis
        if opening_axis and tilt_axis is not None:
            if opening_axis < tilt_axis:
                corrected_tilt_axis -= 1

        for index, angle in enumerate(angles):
            chi = create_ctf(
                angle=angle,
                shape=ctf_shape,
                defocus_x=defoci_x[index],
                defocus_y=defoci_y[index],
                sampling_rate=sampling_rate,
                acceleration_voltage=acceleration_voltage[index],
                spherical_aberration=spherical_aberration[index],
                cutoff_frequency=cutoff_frequency,
                phase_shift=phase_shift[index],
                defocus_angle=defocus_angle[index],
                amplitude_contrast=amplitude_contrast[index],
                tilt_axis=corrected_tilt_axis,
                opening_axis=opening_axis,
                full_shape=shape,
            )

            stack[index] = chi

        # Avoid contrast inversion
        stack = np.negative(stack, out=stack)
        if flip_phase:
            stack = np.abs(stack, out=stack)
        return {"data": be.to_backend_array(stack), "shape": shape}


@dataclass
class CTFReconstructed(CTF):
    """
    Generate CTF filter for reconstructions.
    """

    def _evaluate(
        self,
        shape: Tuple[int],
        defocus_x: Tuple[float],
        amplitude_contrast: float = 0.07,
        phase_shift: Tuple[float] = 0,
        defocus_angle: Tuple[float] = 0,
        defocus_y: Tuple[float] = None,
        sampling_rate: Tuple[float] = 1,
        acceleration_voltage: float = 300e3,
        spherical_aberration: float = 2.7e3,
        flip_phase: bool = True,
        cutoff_frequency: float = 0.5,
        **kwargs: Dict,
    ) -> Dict:
        """
        Compute the CTF weight tilt stack.

        Parameters
        ----------
        shape : tuple of int
            The shape of the CTF.
        defocus_x : tuple of float
            Defocus along the first principal axis in spatial units of sampling rate.
        opening_axis : int, optional
            The axis around which the wedge is opened, defaults to 2.
        amplitude_contrast : float, optional
            The amplitude contrast, defaults to 0.07.
        phase_shift : tuple of float, optional
           CTF phase shift in radians, defaults to 0.
        defocus_angle : tuple of float, optional
            The defocus angle in radians, defaults to 0.
        defocus_y : tuple of float, optional
            Defocus along the second principal axis in spatial units of sampling rate.
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
        stack = create_ctf(
            shape=shape,
            defocus_x=defocus_x,
            defocus_y=defocus_y,
            sampling_rate=np.max(sampling_rate),
            acceleration_voltage=self.acceleration_voltage,
            spherical_aberration=spherical_aberration,
            cutoff_frequency=cutoff_frequency,
            phase_shift=phase_shift,
            defocus_angle=defocus_angle,
            amplitude_contrast=amplitude_contrast,
        )
        # Avoid contrast inversion
        stack = np.negative(stack, out=stack)
        if flip_phase:
            stack = np.abs(stack, out=stack)
        return {"data": be.to_backend_array(stack), "shape": shape}


def _from_xml(filename: str) -> Dict:
    data = XMLParser(filename)

    params = {
        "PhaseShift": None,
        "Amplitude": None,
        "Defocus": None,
        "Voltage": None,
        "Cs": None,
        "DefocusAngle": None,
        "PixelSize": None,
        "Angles": data["Angles"],
    }

    ctf_options = data["CTF"]["Param"]
    for option in ctf_options:
        option = option["@attributes"]
        name = option["Name"]
        if name in params:
            params[name] = option["Value"]

    if "GridCTF" in data:
        ctf = data["GridCTF"]["Node"]
        params["Defocus"] = [ctf[i]["@attributes"]["Value"] for i in range(len(ctf))]
        ctf_phase = data["GridCTFPhase"]["Node"]
        params["PhaseShift"] = [
            ctf_phase[i]["@attributes"]["Value"] for i in range(len(ctf_phase))
        ]
        params["PhaseShift"] = np.degrees(params["PhaseShift"])
        ctf_ast = data["GridCTFDefocusAngle"]["Node"]
        params["DefocusAngle"] = [
            ctf_ast[i]["@attributes"]["Value"] for i in range(len(ctf_ast))
        ]

    missing = [k for k, v in params.items() if v is None]
    if len(missing):
        raise ValueError(f"Could not find {missing} in {filename}.")

    params = {
        k: np.array(v) if hasattr(v, "__len__") else float(v) for k, v in params.items()
    }

    # Convert units to sampling rate (we assume it is Angstrom)
    params["Cs"] = float(params["Cs"] * 1e7)
    params["Defocus"] = params["Defocus"] * 1e4

    mapping = {
        "angles": "Angles",
        "defocus_1": "Defocus",
        "defocus_2": "Defocus",
        "azimuth_astigmatism": "DefocusAngle",
        "additional_phase_shift": "PhaseShift",
        "acceleration_voltage": "Voltage",
        "spherical_aberration": "Cs",
        "amplitude_contrast": "Amplitude",
        "pixel_size": "PixelSize",
    }
    return {k: params[v] for k, v in mapping.items()}


def _from_ctffind(filename: str) -> Dict:
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

    output["additional_phase_shift"] = np.degrees(output["additional_phase_shift"])
    cs = output.get("spherical_aberration")
    if cs is not None:
        output["spherical_aberration"] = float(cs) * 1e7
    return output


def _from_star(filename: str) -> Dict:
    parser = StarParser(filename)

    if "data_stopgap_wedgelist" in parser:
        key = "data_stopgap_wedgelist"
        mapping = {
            "angles": ("_tilt_angle", float, 1),
            "defocus_1": ("_defocus", float, 1e4),
            "defocus_2": (None, float, 1e4),
            "pixel_size": ("_pixelsize", float, 1),
            "acceleration_voltage": ("_voltage", float, 1),
            "spherical_aberration": ("_cs", float, 1e7),
            "amplitude_contrast": ("_amp_contrast", float, 1),
            "additional_phase_shift": (None, float, 1),
            "azimuth_astigmatism": (None, float, 1),
        }
    else:
        key = "data_"
        mapping = {
            "defocus_1": ("_rlnDefocusU", float, 1),
            "defocus_2": ("_rlnDefocusV", float, 1),
            "pixel_size": ("_rlnDetectorPixelSize", float, 1),
            "acceleration_voltage": ("_rlnVoltage", float, 1),
            "spherical_aberration": ("_rlnSphericalAberration", float, 1),
            "amplitude_contrast": ("_rlnAmplitudeContrast", float, 1),
            "additional_phase_shift": (None, float, 1),
            "azimuth_astigmatism": ("_rlnDefocusAngle", float, 1),
        }

    output = {}
    ctf_data = parser[key]
    for out_key, (key, key_dtype, scale) in mapping.items():
        key_value = ctf_data.get(key)
        if key_value is not None:
            try:
                key_value = [key_dtype(x) * scale for x in key_value]
            except Exception:
                pass
        output[out_key] = key_value
    return output


def _from_mdoc(filename: str) -> Dict:
    parser = MDOCParser(filename)

    mapping = {
        "angles": ("TiltAngle", float),
        "defocus_1": ("Defocus", float),
        "acceleration_voltage": ("Voltage", float),
        # These will be None, but on purpose
        "pixel_size": ("_rlnDetectorPixelSize", float),
        "defocus_2": ("Defocus2", float),
        "spherical_aberration": ("_rlnSphericalAberration", float),
        "amplitude_contrast": ("_rlnAmplitudeContrast", float),
        "additional_phase_shift": (None, float),
        "azimuth_astigmatism": ("_rlnDefocusAngle", float),
    }
    output = {}
    for out_key, (key, key_dtype) in mapping.items():
        output[out_key] = parser.get(key, None)

    # Adjust convention and convert to Angstrom
    output["defocus_1"] = np.multiply(output["defocus_1"], -1e4)
    return output


def _compute_electron_wavelength(acceleration_voltage: int = 300e3):
    """Computes the wavelength of an electron in angstrom."""

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


def create_ctf(
    shape: Tuple[int],
    defocus_x: float,
    acceleration_voltage: float = 300e3,
    defocus_angle: float = 0,
    phase_shift: float = 0,
    defocus_y: float = None,
    sampling_rate: float = 1,
    spherical_aberration: float = 2.7e7,
    amplitude_contrast: float = 0.07,
    cutoff_frequency: float = 0.5,
    angle: float = None,
    tilt_axis: int = 0,
    opening_axis: int = None,
    full_shape: Tuple[int] = None,
) -> NDArray:
    """
    Create CTF representation using the definition from [1]_.

    Parameters
    ----------
    shape : Tuple[int]
        Shape of the returned CTF mask.
    defocus_x : float
        Defocus along the first principal axis in spatial units of sampling rate,
        e.g. 30000 Angstrom.
    acceleration_voltage : float, optional
        Acceleration voltage in keV, defaults to 300e3.
    defocus_angle : float, optional
        Astigmatism angle in radians, defaults to 0.
    phase_shift : float, optional
       CTF phase shift in radians, defaults to 0.
    defocus_y : float, optional
        Defocus along the second principal axis in spatial units of sampling rate.
    tilt_axis : int, optional
        Axes the specimen was tilted over, defaults to 0 (x-axis).
    sampling_rate : float or tuple of floats
        Sampling rate throughout shape, e.g., 4 Angstrom per voxel.
    amplitude_contrast : float, optional
        Amplitude contrast of microscope, defaults to 0.07.
    spherical_aberration : float, optional
        Spherical aberration of microscope in units of sampling rate.
    angle : float, optional
        Assume the created CTF is a projection observed at angle degrees.
    opening_axis : int, optional
        Projection axis, only relevant if angle is given.
    full_shape : tuple of ints
        Shape of the entire volume we are observing a projection of. This is required
        to compute aspect ratios for correct scaling. For instance, the 2D CTF slice
        could be (50,50), while the final 3D CTF volume is (50,50,25) with the
        opening_axis being 2, i.e., the z-axis.

    Returns
    -------
    NDArray
        CTF mask.

    References
    ----------
    .. [1]  CTFFIND4: Fast and accurate defocus estimation from electron micrographs.
            Alexis Rohou and Nikolaus Grigorieff. Journal of Structural Biology 2015.
    """
    electron_wavelength = _compute_electron_wavelength(acceleration_voltage)
    electron_wavelength /= sampling_rate
    aberration = (spherical_aberration / sampling_rate) * electron_wavelength**2

    defocus_x = defocus_x / sampling_rate if defocus_x is not None else None
    defocus_y = defocus_y / sampling_rate if defocus_y is not None else None
    if defocus_y is not None:
        if len(shape) < 2:
            raise ValueError(f"Length of shape needs to be at least 2, got {shape}")

        # Axial distance from grid center in voxels
        grid = fftfreqn(
            shape=shape,
            sampling_rate=None,
            return_sparse_grid=True,
            fftshift=False,
        )

    # 0.5 * (dx + dy) + cos(2 * (azimuth - astigmatism) * (dx - dy))
    if defocus_y is not None:
        defocus_sum = np.add(defocus_x, defocus_y)
        defocus_difference = np.subtract(defocus_x, defocus_y)

        # Reusing grid, but in principle pure frequencies would suffice
        angular_grid = np.arctan2(grid[1], grid[0])
        defocus_difference = np.multiply(
            defocus_difference,
            np.cos(2 * (angular_grid - defocus_angle)),
        )
        defocus_x = np.add(defocus_sum, defocus_difference)
        defocus_x *= 0.5

    frequency_grid = fftfreqn(
        shape, sampling_rate=1, compute_euclidean_norm=True, fftshift=False
    )
    if angle is not None and opening_axis is not None and full_shape is not None:
        frequency_grid = frequency_grid_at_angle(
            shape=full_shape,
            tilt_axis=tilt_axis,
            opening_axis=opening_axis,
            angle=angle,
            sampling_rate=1,
            fftshift=False,
        )
    frequency_mask = frequency_grid <= cutoff_frequency

    # k^2*π*λ(dx - 0.5 * sph_abb * λ^2 * k^2) + phase_shift + ampl_contrast_term)
    frequency_grid = np.square(frequency_grid, out=frequency_grid)
    chi = defocus_x - 0.5 * aberration * frequency_grid
    chi = np.multiply(chi, np.pi * electron_wavelength, out=chi)
    chi = np.multiply(chi, frequency_grid, out=chi)
    chi += phase_shift
    chi += np.arctan(
        np.divide(
            amplitude_contrast,
            np.sqrt(1 - np.square(amplitude_contrast)),
        )
    )
    chi = np.sin(-chi, out=chi)
    return np.multiply(chi, frequency_mask, out=chi)
