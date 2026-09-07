"""
Implements class CTF and CTFReconstruced.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import re
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Literal

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
    """Generate per-tilt contrast transfer function filter masks."""

    #: The mean defocus value in Ångstrom (positive is underfocus).
    defocus: Tuple[float] = None
    #: The tilt angles in degrees.
    angles: Tuple[float] = None
    #: The microscope projection axis, defaults to 2 (z).
    opening_axis: int = 2
    #: The axis along which the tilt is applied, defaults to 0 (x).
    tilt_axis: int = 0
    #: The sampling rate, defaults to 1 Ångstrom / voxel.
    sampling_rate: Tuple[float, ...] = 1
    #: The acceleration voltage in kV, defaults to 300
    acceleration_voltage: Tuple[float, ...] = 300
    #: The spherical aberration, defaults to 2.7e7 (in Angstrom).
    spherical_aberration: Tuple[float, ...] = 2.7e7
    #: The amplitude contrast, defaults to 0.07.
    amplitude_contrast: Tuple[float, ...] = 0.07
    #: The phase shift in radians, defaults to 0.
    phase_shift: Tuple[float, ...] = 0
    #: The astigmatism angle in radians, defaults to 0.
    astigmatism_angle: Tuple[float, ...] = 0
    #: The defocus difference (half-delta) in Ångstrom, defaults to 0.
    defocus_delta: Optional[Tuple[float, ...]] = 0
    #: CTF correction mode: 'raw', 'wiener', 'phase-flip', 'phase-flip-weighted'.
    correction_mode: Literal["raw", "wiener", "phase-flip", "phase-flip-weighted"] = (
        "phase-flip"
    )
    #: B-factor style falloff for Wiener filter (Å²). Controls SNR decay with frequency.
    wiener_falloff: Optional[float] = None

    @classmethod
    def from_file(cls, filename: str, **kwargs) -> "CTF":
        """
        Initialize :py:class:`CTF` from file.

        Parameters
        ----------
        filename : str
            The path to a file with ctf parameters. Supports formats are:

            +-------+---------------------------------------------------------+
            | .star | GCTF file                                               |
            +-------+---------------------------------------------------------+
            | .xml  | WARP/M XML file                                         |
            +-------+---------------------------------------------------------+
            | .mdoc | SerialEM file                                           |
            +-------+---------------------------------------------------------+
            | .*    | AreTomo3 CTF file (auto-detected by header)             |
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
        elif _is_aretomo(filename):
            func = _from_aretomo

        data = func(filename=filename)

        # Pixel size needs to be overwritten by pixel size the ctf is generated for
        init_kwargs = {
            "angles": data.get("angles", None),
            "defocus": data["defocus"],
            "sampling_rate": data.get("pixel_size", 1.0),
            "acceleration_voltage": data.get("acceleration_voltage", 300),
            "spherical_aberration": data.get("spherical_aberration"),
            "amplitude_contrast": data.get("amplitude_contrast"),
            "phase_shift": data.get("additional_phase_shift"),
            "astigmatism_angle": data.get("astigmatism_angle"),
            "defocus_delta": data.get("defocus_delta"),
        }
        for k, v in kwargs.items():
            if k in init_kwargs and init_kwargs.get(k) is None:
                init_kwargs[k] = v
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        # Moved format conversion from __post__init
        if "phase_shift" in init_kwargs:
            init_kwargs["phase_shift"] = np.radians(init_kwargs["phase_shift"])
        if "astigmatism_angle" in init_kwargs:
            init_kwargs["astigmatism_angle"] = np.radians(
                init_kwargs["astigmatism_angle"]
            )
        return cls(**init_kwargs)

    def _evaluate(
        self,
        shape: Tuple[int, ...],
        defocus: Tuple[float],
        angles: Tuple[float],
        opening_axis: int = 2,
        tilt_axis: int = 0,
        amplitude_contrast: Tuple[float] = 0.07,
        phase_shift: Tuple[float] = 0,
        astigmatism_angle: Tuple[float] = 0,
        defocus_delta: Tuple[float] = 0,
        sampling_rate: Tuple[float] = 1,
        acceleration_voltage: float = 300,
        spherical_aberration: float = 2.7e7,
        correction_mode: Literal[
            "raw", "phase-flip", "phase-flip-weighted", "wiener"
        ] = "phase-flip",
        wiener_falloff: Optional[float] = None,
        centered_position: Optional[Tuple[float, float, float]] = None,
        **kwargs: Dict,
    ) -> Dict:
        """
        Compute the CTF weight tilt stack.

        Parameters
        ----------
        shape : tuple of int
            The shape of the CTF.
        defocus : tuple of float
            Mean defocus value in Ångstrom (positive is underfocus).
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
        astigmatism_angle : tuple of float, optional
            Astigmatism angle in radians, defaults to 0.
        defocus_delta : tuple of float, optional
            Defocus difference (half-delta) in Ångstrom, defaults to 0.
        sampling_rate : tuple of float, optional
            The sampling rate, defaults to 1.
        acceleration_voltage : float, optional
            Electron acceleration voltage in kV, defaults to 300.
        spherical_aberration : float, optional
            Spherical aberration of microscope in units of sampling rate.
        correction_mode : str, optional
            CTF correction method: 'phase-flip' (default), 'multiply', or 'wiener'.
        wiener_falloff : float, optional
            B-factor style falloff for Wiener filter SNR estimation (Å²).
        **kwargs : Dict
            Additional keyword arguments.
        """
        angles = np.atleast_1d(angles)
        defoci = pad_to_length(defocus, angles.size)
        defoci_delta = pad_to_length(defocus_delta, angles.size)
        phase_shift = pad_to_length(phase_shift, angles.size)
        astigmatism_angle = pad_to_length(astigmatism_angle, angles.size)
        spherical_aberration = pad_to_length(spherical_aberration, angles.size)
        amplitude_contrast = pad_to_length(amplitude_contrast, angles.size)
        acceleration_voltage = pad_to_length(acceleration_voltage, angles.size)

        sampling_rate = np.max(sampling_rate)
        ctf_shape = compute_tilt_shape(
            shape=shape, opening_axis=opening_axis, reduce_dim=True
        )
        stack = np.zeros((len(angles), *ctf_shape))

        if centered_position is not None:
            tilt_rad = np.radians(angles)
            defocus_offset = (
                np.sin(tilt_rad) * centered_position[tilt_axis]
                + np.cos(tilt_rad) * centered_position[opening_axis]
            )
            defocus_offset = defocus_offset * sampling_rate
            defoci = defoci + defocus_offset

        for index, angle in enumerate(angles):
            chi = create_ctf(
                angle=angle,
                shape=ctf_shape,
                defocus=defoci[index],
                defocus_delta=defoci_delta[index],
                sampling_rate=sampling_rate,
                acceleration_voltage=acceleration_voltage[index],
                spherical_aberration=spherical_aberration[index],
                phase_shift=phase_shift[index],
                astigmatism_angle=astigmatism_angle[index],
                amplitude_contrast=amplitude_contrast[index],
                tilt_axis=tilt_axis,
                opening_axis=opening_axis,
                full_shape=shape,
            )

            stack[index] = chi

        mode = correction_mode
        if correction_mode == "phase-flip-weighted":
            mode = "phase-flip"

        stack = _apply_ctf_correction(
            ctf=stack,
            correction_mode=mode,
            sampling_rate=sampling_rate,
            shape=ctf_shape,
            wiener_falloff=wiener_falloff,
        )
        ret = {"data": be.to_backend_array(stack), "shape": shape}
        if correction_mode == "phase-flip-weighted":
            ret["data"] = be.multiply(ret["data"], ret["data"])
            ret["data_weights"] = be.to_backend_array(np.abs(stack))
        return ret


@dataclass
class CTFReconstructed(CTF):
    """
    Generate CTF filter for reconstructions.
    """

    def _evaluate(
        self,
        shape: Tuple[int],
        defocus: Tuple[float],
        amplitude_contrast: float = 0.07,
        phase_shift: Tuple[float] = 0,
        astigmatism_angle: Tuple[float] = 0,
        defocus_delta: Tuple[float] = 0,
        sampling_rate: Tuple[float] = 1,
        acceleration_voltage: float = 300,
        spherical_aberration: float = 2.7e7,
        correction_mode: str = "phase-flip",
        wiener_falloff: Optional[float] = None,
        **kwargs: Dict,
    ) -> Dict:
        """
        Compute the CTF weight tilt stack.

        Parameters
        ----------
        shape : tuple of int
            The shape of the CTF.
        defocus : tuple of float
            Mean defocus value in Ångstrom (positive is underfocus).
        opening_axis : int, optional
            The axis around which the wedge is opened, defaults to 2.
        amplitude_contrast : float, optional
            The amplitude contrast, defaults to 0.07.
        phase_shift : tuple of float, optional
           CTF phase shift in radians, defaults to 0.
        astigmatism_angle : tuple of float, optional
            The astigmatism angle in radians, defaults to 0.
        defocus_delta : tuple of float, optional
            Defocus difference (half-delta) in Ångstrom, defaults to 0.
        sampling_rate : tuple of float, optional
            The sampling rate, defaults to 1.
        acceleration_voltage : float, optional
            The acceleration voltage in kV, defaults to 300.
        spherical_aberration : float, optional
            The spherical aberration coefficient, defaults to 2.7e7.
        correction_mode : str, optional
            CTF correction mode: 'phase-flip', 'raw', 'wiener', or 'wiener-phase-flip'.
        wiener_falloff : float, optional
            B-factor style falloff for Wiener filter SNR estimation (Å²).
        **kwargs : Dict
            Additional keyword arguments.

        Returns
        -------
        NDArray
            A stack containing the CTF weight.
        """
        sampling_rate = np.max(sampling_rate)
        stack = create_ctf(
            shape=shape,
            defocus=defocus,
            defocus_delta=defocus_delta,
            sampling_rate=sampling_rate,
            acceleration_voltage=acceleration_voltage,
            spherical_aberration=spherical_aberration,
            phase_shift=phase_shift,
            astigmatism_angle=astigmatism_angle,
            amplitude_contrast=amplitude_contrast,
        )
        stack = _apply_ctf_correction(
            ctf=stack,
            correction_mode=correction_mode,
            sampling_rate=sampling_rate,
            shape=shape,
            wiener_falloff=wiener_falloff,
        )
        return {"data": be.to_backend_array(stack), "shape": shape}


def _is_aretomo(filename: str) -> bool:
    """Check whether *filename* looks like an AreTomo3 CTF output file."""
    with open(filename, mode="r", encoding="utf-8") as infile:
        first_line = infile.readline().strip()
    return first_line.startswith("# Columns:")


def _from_aretomo(filename: str) -> Dict:
    """
    Parse an AreTomo3 *_CTF.txt file.

    The file contains a header line with 8 columnes of whitespace-delimited data

        #1 micrograph number
        #2 defocus1 [Å]
        #3 defocus2 [Å]
        #4 azimuth of astigmatism [deg]
        #5 additional phase shift [rad]
        #6 cross-correlation
        #7 spacing [Å]
        #8 dfHand

    Notes
    -----
    Micorscope parameters for the current run, i.e., pixel size, voltage, Cs, and
    amplitude contrast are not stored in the file and must be specified.
    """
    with open(filename, mode="r", encoding="utf-8") as infile:
        lines = [x.strip() for x in infile.read().split("\n")]
        lines = [x for x in lines if len(x) and not x.startswith("#")]

    columns = {
        "defocus_1": 1,
        "defocus_2": 2,
        "astigmatism_angle": 3,
        "additional_phase_shift": 4,
    }

    output = {key: [] for key in columns}
    for line in lines:
        values = line.split()
        for key, col in columns.items():
            output[key].append(float(values[col]))

    for key in columns:
        output[key] = np.array(output[key])

    output["defocus"], output["defocus_delta"] = _ctffind_to_warp_defocus(
        output.pop("defocus_1", None), output.pop("defocus_2", None)
    )
    output["additional_phase_shift"] = np.degrees(output["additional_phase_shift"])

    output["pixel_size"] = None
    output["acceleration_voltage"] = None
    output["spherical_aberration"] = None
    output["amplitude_contrast"] = None
    return output


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

        ctf_ddefocus = data["GridCTFDefocusDelta"]["Node"]
        params["DefocusDelta"] = [
            ctf_ddefocus[i]["@attributes"]["Value"] for i in range(len(ctf_ddefocus))
        ]

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

    # Convert units to Angstrom
    params["Cs"] = float(params["Cs"] * 1e7)
    params["Defocus"] = params["Defocus"] * 1e4
    params["DefocusDelta"] = params["DefocusDelta"] * 0.5 * 1e4

    mapping = {
        "angles": "Angles",
        "defocus": "Defocus",
        "defocus_delta": "DefocusDelta",
        "astigmatism_angle": "DefocusAngle",
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
        "astigmatism_angle": 3,
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

    output["defocus"], output["defocus_delta"] = _ctffind_to_warp_defocus(
        output.pop("defocus_1", None), output.pop("defocus_2", None)
    )
    output["additional_phase_shift"] = np.degrees(output["additional_phase_shift"])
    if output.get("spherical_aberration") is not None:
        output["spherical_aberration"] = float(output["spherical_aberration"]) * 1e7
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
            "astigmatism_angle": (None, float, 1),
        }
    else:
        mapping = {
            "defocus_1": ("_rlnDefocusU", float, 1),
            "defocus_2": ("_rlnDefocusV", float, 1),
            "pixel_size": ("_rlnDetectorPixelSize", float, 1),
            "acceleration_voltage": ("_rlnVoltage", float, 1),
            "spherical_aberration": ("_rlnSphericalAberration", float, 1),
            "amplitude_contrast": ("_rlnAmplitudeContrast", float, 1),
            "additional_phase_shift": (None, float, 1),
            "astigmatism_angle": ("_rlnDefocusAngle", float, 1),
            "angles": ("_rlnTomoNominalStageTiltAngle", float, 1),
        }

        potential_keys = [x for x in parser.keys() if x.startswith("data_")]
        if len(potential_keys) != 1:
            raise ValueError(
                f"Expected one 'data_*' field, got {len(potential_keys)} {potential_keys}"
            )
        key = potential_keys[0]

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

    output["defocus"], output["defocus_delta"] = _ctffind_to_warp_defocus(
        output.pop("defocus_1", None), output.pop("defocus_2", None)
    )
    return output


def _from_mdoc(filename: str) -> Dict:
    parser = MDOCParser(filename)

    mapping = {
        "angles": ("TiltAngle", float),
        "defocus_1": ("Defocus", float),
        "acceleration_voltage": ("Voltage", float),
        # These will be None
        "pixel_size": ("_rlnDetectorPixelSize", float),
        "defocus_2": ("Defocus2", float),
        "spherical_aberration": ("_rlnSphericalAberration", float),
        "amplitude_contrast": ("_rlnAmplitudeContrast", float),
        "additional_phase_shift": (None, float),
        "astigmatism_angle": ("_rlnDefocusAngle", float),
    }
    output = {}
    for out_key, (key, key_dtype) in mapping.items():
        output[out_key] = parser.get(key, None)

    defocus1 = output.pop("defocus_1", None)
    defocus2 = output.pop("defocus_2", None)
    defocus, defocus_delta = _ctffind_to_warp_defocus(defocus1, defocus2)

    # Convert from microns to Angstrom
    output["defocus"] = np.multiply(-defocus, 1e4)
    if defocus_delta is not None:
        defocus_delta = np.multiply(defocus_delta, 1e4)
    output["defocus_delta"] = defocus_delta
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


def _ctffind_to_warp_defocus(defocus1, defocus2):
    if defocus2 is None:
        return defocus1, None

    defocus = np.add(defocus1, defocus2) / 2
    defocus_delta = np.subtract(defocus1, defocus2) / 2
    return defocus, defocus_delta


def _apply_ctf_correction(
    ctf: NDArray,
    correction_mode: str,
    sampling_rate: float,
    shape: Tuple[int, ...],
    wiener_falloff: float = 100.0,
) -> NDArray:
    """
    Apply CTF correction based on the specified mode.

    Parameters
    ----------
    ctf : NDArray
        Raw CTF values (can be 2D or 3D stack).
    correction_mode : str
        Correction mode based on tomogram state:
        'phase-flip' for phase-flip corrected tomograms,
        'wiener' for tomograms without CTF correction with Wiener filter,
        'raw' for tomograms without CTF correction.
    sampling_rate : float
        Sampling rate in Å/voxel.
    shape : tuple of int
        Shape of the CTF array (excluding batch dimension if present).
    wiener_falloff : float
        B-factor style falloff for Wiener filter (Å²), default is 100.

    Returns
    -------
    NDArray
        Corrected CTF filter.
    """
    if correction_mode == "phase-flip":
        return np.abs(ctf)
    elif correction_mode == "raw":
        return ctf
    elif correction_mode == "wiener":
        freq_grid = fftfreqn(
            shape,
            sampling_rate=sampling_rate,
            compute_euclidean_norm=True,
            fftshift=False,
        )

        # Estimate SNR as SNR(k) = exp(-B * k^2 / 4) with k 1/Å and B as falloff
        if wiener_falloff is None:
            wiener_falloff = 100.0

        snr = np.exp(-wiener_falloff * np.square(freq_grid) / 4.0)

        # CTF / (CTF^2 + 1/SNR + e)
        ctf_sq = np.square(ctf)
        denominator = ctf_sq + np.divide(1.0, snr + 1e-10)
        return np.divide(ctf, denominator + 1e-10)
    else:
        raise ValueError(
            f"Unknown correction_mode '{correction_mode}'. "
            "Expected 'phase-flip', 'wiener', or 'raw'."
        )


def create_ctf(
    shape: Tuple[int],
    defocus: float,
    acceleration_voltage: float = 300,
    astigmatism_angle: float = 0,
    phase_shift: float = 0,
    defocus_delta: float = 0,
    sampling_rate: float = 1,
    spherical_aberration: float = 2.7e7,
    amplitude_contrast: float = 0.07,
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
    defocus : float
        Mean defocus value in Ångstrom (positive is underfocus).
    acceleration_voltage : float, optional
        Acceleration voltage in kV, defaults to 300.
    astigmatism_angle : float, optional
        Astigmatism angle in radians, defaults to 0.
    phase_shift : float, optional
       CTF phase shift in radians, defaults to 0.
    defocus_delta : float, optional
        Defocus difference (half-delta) in Ångstrom, defaults to 0.
    tilt_axis : int, optional
        Axes the specimen was tilted over, defaults to 0 (x-axis).
    sampling_rate : float or tuple of floats
        Sampling rate throughout shape, e.g., 4 Ångstrom per voxel.
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
    electron_wavelength = _compute_electron_wavelength(acceleration_voltage * 1e3)
    aberration = spherical_aberration * electron_wavelength**2

    # WARP style effective defocus
    # eff_defocus = defocus + defocus_delta * cos(2 * (angle - astigmatism_angle))
    eff_defocus = defocus
    if defocus_delta is not None and defocus_delta != 0:
        if len(shape) < 2:
            raise ValueError(f"Length of shape needs to be at least 2, got {shape}")

        grid = fftfreqn(
            shape=shape,
            sampling_rate=None,
            return_sparse_grid=True,
            fftshift=False,
        )
        # x/y definition is swapped because we transpose input
        angular_grid = np.arctan2(grid[1], grid[0])
        eff_defocus = defocus + defocus_delta * np.cos(
            2 * (angular_grid - astigmatism_angle)
        )

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
    frequency_grid = np.divide(frequency_grid, sampling_rate, out=frequency_grid)

    # k^2*π*λ(defocus - 0.5 * sph_abb * λ^2 * k^2) + phase_shift + ampl_contrast_term)
    frequency_grid = np.square(frequency_grid, out=frequency_grid)
    chi = eff_defocus - 0.5 * aberration * frequency_grid
    chi = np.multiply(chi, np.pi * electron_wavelength, out=chi)
    chi = np.multiply(chi, frequency_grid, out=chi)
    chi += phase_shift
    chi += np.arctan(
        np.divide(
            amplitude_contrast,
            np.sqrt(1 - np.square(amplitude_contrast)),
        )
    )
    return np.sin(chi, out=chi)
