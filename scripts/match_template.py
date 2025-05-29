#!python3
""" CLI for basic pyTME template matching functions.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import os
import argparse
import warnings
from sys import exit
from time import time
from typing import Tuple
from copy import deepcopy
from os.path import exists
from tempfile import gettempdir

import numpy as np

from tme.backends import backend as be
from tme import Density, __version__
from tme.matching_utils import scramble_phases, write_pickle
from tme.matching_exhaustive import scan_subsets, MATCHING_EXHAUSTIVE_REGISTER
from tme.rotations import (
    get_cone_rotations,
    get_rotation_matrices,
)
from tme.matching_data import MatchingData
from tme.analyzer import (
    MaxScoreOverRotations,
    PeakCallerMaximumFilter,
)
from tme.filters import (
    CTF,
    Wedge,
    Compose,
    BandPassFilter,
    WedgeReconstructed,
    ReconstructFromTilt,
    LinearWhiteningFilter,
)


def get_func_fullname(func) -> str:
    """Returns the full name of the given function, including its module."""
    return f"<function '{func.__module__}.{func.__name__}'>"


def print_block(name: str, data: dict, label_width=20) -> None:
    """Prints a formatted block of information."""
    print(f"\n> {name}")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            value = value.shape
        formatted_value = str(value)
        print(f"  - {key + ':':<{label_width}} {formatted_value}")


def print_entry() -> None:
    width = 80
    text = f" pytme v{__version__} "
    padding_total = width - len(text) - 2
    padding_left = padding_total // 2
    padding_right = padding_total - padding_left

    print("*" * width)
    print(f"*{ ' ' * padding_left }{text}{ ' ' * padding_right }*")
    print("*" * width)


def check_positive(value):
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive float." % value)
    return ivalue


def load_and_validate_mask(mask_target: "Density", mask_path: str, **kwargs):
    """
    Loadsa mask in CCP4/MRC format and assess whether the sampling_rate
    and shape matches its target.

    Parameters
    ----------
    mask_target : Density
        Object the mask should be applied to
    mask_path : str
        Path to the mask in CCP4/MRC format.
    kwargs : dict, optional
        Keyword arguments passed to :py:meth:`tme.density.Density.from_file`.
    Raise
    -----
    ValueError
        If shape or sampling rate do not match between mask_target and mask

    Returns
    -------
    Density
        A density instance if the mask was validated and loaded otherwise None
    """
    mask = mask_path
    if mask is not None:
        mask = Density.from_file(mask, **kwargs)
        mask.origin = deepcopy(mask_target.origin)
        if not np.allclose(mask.shape, mask_target.shape):
            raise ValueError(
                f"Expected shape of {mask_path} was {mask_target.shape},"
                f" got f{mask.shape}"
            )
        if not np.allclose(
            np.round(mask.sampling_rate, 2), np.round(mask_target.sampling_rate, 2)
        ):
            raise ValueError(
                f"Expected sampling_rate of {mask_path} was {mask_target.sampling_rate}"
                f", got f{mask.sampling_rate}"
            )
    return mask


def parse_rotation_logic(args, ndim):
    if args.angular_sampling is not None:
        rotations = get_rotation_matrices(
            angular_sampling=args.angular_sampling,
            dim=ndim,
            use_optimized_set=not args.no_use_optimized_set,
        )
        if args.angular_sampling >= 180:
            rotations = np.eye(ndim).reshape(1, ndim, ndim)
        return rotations

    if args.axis_sampling is None:
        args.axis_sampling = args.cone_sampling

    rotations = get_cone_rotations(
        cone_angle=args.cone_angle,
        cone_sampling=args.cone_sampling,
        axis_angle=args.axis_angle,
        axis_sampling=args.axis_sampling,
        n_symmetry=args.axis_symmetry,
        axis=[0 if i != args.cone_axis else 1 for i in range(ndim)],
        reference=[0, 0, -1],
    )
    return rotations


def compute_schedule(
    args,
    matching_data: MatchingData,
    callback_class,
    pad_edges: bool = False,
):
    # User requested target padding
    if args.pad_edges is True:
        pad_edges = True

    splits, schedule = matching_data.computation_schedule(
        matching_method=args.score,
        analyzer_method=callback_class.__name__,
        use_gpu=args.use_gpu,
        pad_fourier=False,
        pad_target_edges=pad_edges,
        available_memory=args.memory,
        max_cores=args.cores,
    )

    if splits is None:
        print(
            "Found no suitable parallelization schedule. Consider increasing"
            " available RAM or decreasing number of cores."
        )
        exit(-1)

    n_splits = np.prod(list(splits.values()))
    if pad_edges is False and len(matching_data._target_dim) == 0 and n_splits > 1:
        args.pad_edges = True
        return compute_schedule(args, matching_data, callback_class, True)
    return splits, schedule


def setup_filter(args, template: Density, target: Density) -> Tuple[Compose, Compose]:
    needs_reconstruction = False
    template_filter, target_filter = [], []
    if args.tilt_angles is not None:
        needs_reconstruction = args.tilt_weighting is not None
        try:
            wedge = Wedge.from_file(args.tilt_angles)
            wedge.weight_type = args.tilt_weighting
            if args.tilt_weighting in ("angle", None) and args.ctf_file is None:
                wedge = WedgeReconstructed(
                    angles=wedge.angles,
                    weight_wedge=args.tilt_weighting == "angle",
                    opening_axis=args.wedge_axes[0],
                    tilt_axis=args.wedge_axes[1],
                )
        except FileNotFoundError:
            tilt_step, create_continuous_wedge = None, True
            tilt_start, tilt_stop = args.tilt_angles.split(",")
            if ":" in tilt_stop:
                create_continuous_wedge = False
                tilt_stop, tilt_step = tilt_stop.split(":")
            tilt_start, tilt_stop = float(tilt_start), float(tilt_stop)
            tilt_angles = (tilt_start, tilt_stop)
            if tilt_step is not None:
                tilt_step = float(tilt_step)
                tilt_angles = np.arange(
                    -tilt_start, tilt_stop + tilt_step, tilt_step
                ).tolist()

            if args.tilt_weighting is not None and tilt_step is None:
                raise ValueError(
                    "Tilt weighting is not supported for continuous wedges."
                )
            if args.tilt_weighting not in ("angle", None):
                raise ValueError(
                    "Tilt weighting schemes other than 'angle' or 'None' require "
                    "a specification of electron doses via --tilt_angles."
                )

            wedge = Wedge(
                angles=tilt_angles,
                opening_axis=args.wedge_axes[0],
                tilt_axis=args.wedge_axes[1],
                shape=None,
                weight_type=None,
                weights=np.ones_like(tilt_angles),
            )
            if args.tilt_weighting in ("angle", None) and args.ctf_file is None:
                wedge = WedgeReconstructed(
                    angles=tilt_angles,
                    weight_wedge=args.tilt_weighting == "angle",
                    create_continuous_wedge=create_continuous_wedge,
                    reconstruction_filter=args.reconstruction_filter,
                    opening_axis=args.wedge_axes[0],
                    tilt_axis=args.wedge_axes[1],
                )
                wedge_target = WedgeReconstructed(
                    angles=(np.abs(np.min(tilt_angles)), np.abs(np.max(tilt_angles))),
                    weight_wedge=False,
                    create_continuous_wedge=True,
                    opening_axis=args.wedge_axes[0],
                    tilt_axis=args.wedge_axes[1],
                )
                target_filter.append(wedge_target)

        wedge.sampling_rate = template.sampling_rate
        template_filter.append(wedge)
        if not isinstance(wedge, WedgeReconstructed):
            reconstruction_filter = ReconstructFromTilt(
                reconstruction_filter=args.reconstruction_filter,
                interpolation_order=args.reconstruction_interpolation_order,
            )
            template_filter.append(reconstruction_filter)

    if args.ctf_file is not None or args.defocus is not None:
        needs_reconstruction = True
        if args.ctf_file is not None:
            ctf = CTF.from_file(args.ctf_file)
            n_tilts_ctfs, n_tils_angles = len(ctf.defocus_x), len(wedge.angles)
            if n_tilts_ctfs != n_tils_angles:
                raise ValueError(
                    f"CTF file contains {n_tilts_ctfs} micrographs, but match_template "
                    f"recieved {n_tils_angles} tilt angles. Expected one angle "
                    "per micrograph."
                )
            ctf.angles = wedge.angles
            ctf.no_reconstruction = False
            ctf.opening_axis, ctf.tilt_axis = args.wedge_axes
        else:
            needs_reconstruction = False
            ctf = CTF(
                defocus_x=[args.defocus],
                phase_shift=[args.phase_shift],
                defocus_y=None,
                angles=[0],
                shape=None,
                return_real_fourier=True,
            )
        ctf.sampling_rate = template.sampling_rate
        ctf.flip_phase = args.no_flip_phase
        ctf.amplitude_contrast = args.amplitude_contrast
        ctf.spherical_aberration = args.spherical_aberration
        ctf.acceleration_voltage = args.acceleration_voltage * 1e3
        ctf.correct_defocus_gradient = args.correct_defocus_gradient

        if not needs_reconstruction:
            template_filter.append(ctf)
        elif isinstance(template_filter[-1], ReconstructFromTilt):
            template_filter.insert(-1, ctf)
        else:
            template_filter.insert(0, ctf)
            template_filter.insert(
                1,
                ReconstructFromTilt(
                    reconstruction_filter=args.reconstruction_filter,
                    interpolation_order=args.reconstruction_interpolation_order,
                ),
            )

    if args.lowpass or args.highpass is not None:
        lowpass, highpass = args.lowpass, args.highpass
        if args.pass_format == "voxel":
            if lowpass is not None:
                lowpass = np.max(np.multiply(lowpass, template.sampling_rate))
            if highpass is not None:
                highpass = np.max(np.multiply(highpass, template.sampling_rate))
        elif args.pass_format == "frequency":
            if lowpass is not None:
                lowpass = np.max(np.divide(template.sampling_rate, lowpass))
            if highpass is not None:
                highpass = np.max(np.divide(template.sampling_rate, highpass))

        try:
            if args.lowpass >= args.highpass:
                warnings.warn("--lowpass should be smaller than --highpass.")
        except Exception:
            pass

        bandpass = BandPassFilter(
            use_gaussian=args.no_pass_smooth,
            lowpass=lowpass,
            highpass=highpass,
            sampling_rate=template.sampling_rate,
        )
        template_filter.append(bandpass)
        target_filter.append(bandpass)

    if args.whiten_spectrum:
        whitening_filter = LinearWhiteningFilter()
        template_filter.append(whitening_filter)
        target_filter.append(whitening_filter)

    if needs_reconstruction and args.reconstruction_filter is None:
        warnings.warn(
            "Consider using a --reconstruction_filter such as 'ramp' to avoid artifacts."
        )

    template_filter = Compose(template_filter) if len(template_filter) else None
    target_filter = Compose(target_filter) if len(target_filter) else None
    if args.no_filter_target:
        target_filter = None

    return template_filter, target_filter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform template matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument(
        "-m",
        "--target",
        dest="target",
        type=str,
        required=True,
        help="Path to a target in CCP4/MRC, EM, H5 or another format supported by "
        "tme.density.Density.from_file "
        "https://kosinskilab.github.io/pyTME/reference/api/tme.density.Density.from_file.html",
    )
    io_group.add_argument(
        "--target_mask",
        dest="target_mask",
        type=str,
        required=False,
        help="Path to a mask for the target in a supported format (see target).",
    )
    io_group.add_argument(
        "-i",
        "--template",
        dest="template",
        type=str,
        required=True,
        help="Path to a template in PDB/MMCIF or other supported formats (see target).",
    )
    io_group.add_argument(
        "--template_mask",
        dest="template_mask",
        type=str,
        required=False,
        help="Path to a mask for the template in a supported format (see target).",
    )
    io_group.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=False,
        default="output.pickle",
        help="Path to the output pickle file.",
    )
    io_group.add_argument(
        "--invert_target_contrast",
        dest="invert_target_contrast",
        action="store_true",
        default=False,
        help="Invert the target's contrast for cases where templates to-be-matched have "
        "negative values, e.g. tomograms.",
    )
    io_group.add_argument(
        "--scramble_phases",
        dest="scramble_phases",
        action="store_true",
        default=False,
        help="Phase scramble the template to generate a noise score background.",
    )

    scoring_group = parser.add_argument_group("Scoring")
    scoring_group.add_argument(
        "-s",
        dest="score",
        type=str,
        default="FLCSphericalMask",
        choices=list(MATCHING_EXHAUSTIVE_REGISTER.keys()),
        help="Template matching scoring function.",
    )

    angular_group = parser.add_argument_group("Angular Sampling")
    angular_exclusive = angular_group.add_mutually_exclusive_group(required=True)

    angular_exclusive.add_argument(
        "-a",
        dest="angular_sampling",
        type=check_positive,
        default=None,
        help="Angular sampling rate using optimized rotational sets."
        "A lower number yields more rotations. Values >= 180 sample only the identity.",
    )
    angular_exclusive.add_argument(
        "--cone_angle",
        dest="cone_angle",
        type=check_positive,
        default=None,
        help="Half-angle of the cone to be sampled in degrees. Allows to sample a "
        "narrow interval around a known orientation, e.g. for surface oversampling.",
    )
    angular_group.add_argument(
        "--cone_axis",
        dest="cone_axis",
        type=check_positive,
        default=2,
        help="Principal axis to build cone around.",
    )
    angular_group.add_argument(
        "--invert_cone",
        dest="invert_cone",
        action="store_true",
        help="Invert cone handedness.",
    )
    angular_group.add_argument(
        "--cone_sampling",
        dest="cone_sampling",
        type=check_positive,
        default=None,
        help="Sampling rate of the cone in degrees.",
    )
    angular_group.add_argument(
        "--axis_angle",
        dest="axis_angle",
        type=check_positive,
        default=360.0,
        required=False,
        help="Sampling angle along the z-axis of the cone.",
    )
    angular_group.add_argument(
        "--axis_sampling",
        dest="axis_sampling",
        type=check_positive,
        default=None,
        required=False,
        help="Sampling rate along the z-axis of the cone. Defaults to --cone_sampling.",
    )
    angular_group.add_argument(
        "--axis_symmetry",
        dest="axis_symmetry",
        type=check_positive,
        default=1,
        required=False,
        help="N-fold symmetry around z-axis of the cone.",
    )
    angular_group.add_argument(
        "--no_use_optimized_set",
        dest="no_use_optimized_set",
        action="store_true",
        default=False,
        required=False,
        help="Whether to use random uniform instead of optimized rotation sets.",
    )

    computation_group = parser.add_argument_group("Computation")
    computation_group.add_argument(
        "-n",
        dest="cores",
        required=False,
        type=int,
        default=4,
        help="Number of cores used for template matching.",
    )
    computation_group.add_argument(
        "--use_gpu",
        dest="use_gpu",
        action="store_true",
        default=False,
        help="Whether to perform computations on the GPU.",
    )
    computation_group.add_argument(
        "--gpu_indices",
        dest="gpu_indices",
        type=str,
        default=None,
        help="Comma-separated list of GPU indices to use. For example,"
        " 0,1 for the first and second GPU. Only used if --use_gpu is set."
        " If not provided but --use_gpu is set, CUDA_VISIBLE_DEVICES will"
        " be respected.",
    )
    computation_group.add_argument(
        "-r",
        "--ram",
        dest="memory",
        required=False,
        type=int,
        default=None,
        help="Amount of memory that can be used in bytes.",
    )
    computation_group.add_argument(
        "--memory_scaling",
        dest="memory_scaling",
        required=False,
        type=float,
        default=0.85,
        help="Fraction of available memory to be used. Ignored if --ram is set.",
    )
    computation_group.add_argument(
        "--temp_directory",
        dest="temp_directory",
        default=None,
        help="Directory for temporary objects. Faster I/O improves runtime.",
    )
    computation_group.add_argument(
        "--backend",
        dest="backend",
        default=None,
        choices=be.available_backends(),
        help="[Expert] Overwrite default computation backend.",
    )
    filter_group = parser.add_argument_group("Filters")
    filter_group.add_argument(
        "--lowpass",
        dest="lowpass",
        type=float,
        required=False,
        help="Resolution to lowpass filter template and target to in the same unit "
        "as the sampling rate of template and target (typically Ångstrom).",
    )
    filter_group.add_argument(
        "--highpass",
        dest="highpass",
        type=float,
        required=False,
        help="Resolution to highpass filter template and target to in the same unit "
        "as the sampling rate of template and target (typically Ångstrom).",
    )
    filter_group.add_argument(
        "--no_pass_smooth",
        dest="no_pass_smooth",
        action="store_false",
        default=True,
        help="Whether a hard edge filter should be used for --lowpass and --highpass.",
    )
    filter_group.add_argument(
        "--pass_format",
        dest="pass_format",
        type=str,
        required=False,
        default="sampling_rate",
        choices=["sampling_rate", "voxel", "frequency"],
        help="How values passed to --lowpass and --highpass should be interpreted. ",
    )
    filter_group.add_argument(
        "--whiten_spectrum",
        dest="whiten_spectrum",
        action="store_true",
        default=None,
        help="Apply spectral whitening to template and target based on target spectrum.",
    )
    filter_group.add_argument(
        "--wedge_axes",
        dest="wedge_axes",
        type=str,
        required=False,
        default=None,
        help="Indices of wedge opening and tilt axis, e.g. '2,0' for a wedge open "
        "in z and tilted over the x-axis.",
    )
    filter_group.add_argument(
        "--tilt_angles",
        dest="tilt_angles",
        type=str,
        required=False,
        default=None,
        help="Path to a tab-separated file containing the column angles and optionally "
        " weights, or comma separated start and stop stage tilt angle, e.g. 50,45, which "
        " yields a continuous wedge mask. Alternatively, a tilt step size can be "
        "specified like 50,45:5.0 to sample 5.0 degree tilt angle steps.",
    )
    filter_group.add_argument(
        "--tilt_weighting",
        dest="tilt_weighting",
        type=str,
        required=False,
        choices=["angle", "relion", "grigorieff"],
        default=None,
        help="Weighting scheme used to reweight individual tilts. Available options: "
        "angle (cosine based weighting), "
        "relion (relion formalism for wedge weighting) requires,"
        "grigorieff (exposure filter as defined in Grant and Grigorieff 2015)."
        "relion and grigorieff require electron doses in --tilt_angles weights column.",
    )
    filter_group.add_argument(
        "--reconstruction_filter",
        dest="reconstruction_filter",
        type=str,
        required=False,
        choices=["ram-lak", "ramp", "ramp-cont", "shepp-logan", "cosine", "hamming"],
        default=None,
        help="Filter applied when reconstructing (N+1)-D from N-D filters.",
    )
    filter_group.add_argument(
        "--reconstruction_interpolation_order",
        dest="reconstruction_interpolation_order",
        type=int,
        default=1,
        required=False,
        help="Analogous to --interpolation_order but for reconstruction.",
    )
    filter_group.add_argument(
        "--no_filter_target",
        dest="no_filter_target",
        action="store_true",
        default=False,
        help="Whether to not apply potential filters to the target.",
    )

    ctf_group = parser.add_argument_group("Contrast Transfer Function")
    ctf_group.add_argument(
        "--ctf_file",
        dest="ctf_file",
        type=str,
        required=False,
        default=None,
        help="Path to a file with CTF parameters from CTFFIND4. Each line will be "
        "interpreted as tilt obtained at the angle specified in --tilt_angles. ",
    )
    ctf_group.add_argument(
        "--defocus",
        dest="defocus",
        type=float,
        required=False,
        default=None,
        help="Defocus in units of sampling rate (typically Ångstrom). "
        "Superseded by --ctf_file.",
    )
    ctf_group.add_argument(
        "--phase_shift",
        dest="phase_shift",
        type=float,
        required=False,
        default=0,
        help="Phase shift in degrees. Superseded by --ctf_file.",
    )
    ctf_group.add_argument(
        "--acceleration_voltage",
        dest="acceleration_voltage",
        type=float,
        required=False,
        default=300,
        help="Acceleration voltage in kV.",
    )
    ctf_group.add_argument(
        "--spherical_aberration",
        dest="spherical_aberration",
        type=float,
        required=False,
        default=2.7e7,
        help="Spherical aberration in units of sampling rate (typically Ångstrom).",
    )
    ctf_group.add_argument(
        "--amplitude_contrast",
        dest="amplitude_contrast",
        type=float,
        required=False,
        default=0.07,
        help="Amplitude contrast.",
    )
    ctf_group.add_argument(
        "--no_flip_phase",
        dest="no_flip_phase",
        action="store_false",
        required=False,
        help="Do not perform phase-flipping CTF correction.",
    )
    ctf_group.add_argument(
        "--correct_defocus_gradient",
        dest="correct_defocus_gradient",
        action="store_true",
        required=False,
        help="[Experimental] Whether to compute a more accurate 3D CTF incorporating "
        "defocus gradients.",
    )

    performance_group = parser.add_argument_group("Performance")
    performance_group.add_argument(
        "--no_centering",
        dest="no_centering",
        action="store_true",
        help="Assumes the template is already centered and omits centering.",
    )
    performance_group.add_argument(
        "--pad_edges",
        dest="pad_edges",
        action="store_true",
        default=False,
        help="Whether to pad the edges of the target. Useful if the target does not "
        "a well-defined bounding box. Defaults to True if splitting is required.",
    )
    performance_group.add_argument(
        "--pad_filter",
        dest="pad_filter",
        action="store_true",
        default=False,
        help="Pads the filter to the shape of the target. Particularly useful for fast "
        "oscilating filters to avoid aliasing effects.",
    )
    performance_group.add_argument(
        "--interpolation_order",
        dest="interpolation_order",
        required=False,
        type=int,
        default=3,
        help="Spline interpolation used for rotations.",
    )
    performance_group.add_argument(
        "--use_mixed_precision",
        dest="use_mixed_precision",
        action="store_true",
        default=False,
        help="Use float16 for real values operations where possible.",
    )
    performance_group.add_argument(
        "--use_memmap",
        dest="use_memmap",
        action="store_true",
        default=False,
        help="Use memmaps to offload large data objects to disk. "
        "Particularly useful for large inputs in combination with --use_gpu.",
    )

    analyzer_group = parser.add_argument_group("Analyzer")
    analyzer_group.add_argument(
        "--score_threshold",
        dest="score_threshold",
        required=False,
        type=float,
        default=0,
        help="Minimum template matching scores to consider for analysis.",
    )
    analyzer_group.add_argument(
        "-p",
        dest="peak_calling",
        action="store_true",
        default=False,
        help="Perform peak calling instead of score aggregation.",
    )
    analyzer_group.add_argument(
        "--number_of_peaks",
        dest="number_of_peaks",
        action="store_true",
        default=1000,
        help="Number of peaks to call, 1000 by default.",
    )
    args = parser.parse_args()
    args.version = __version__

    if args.interpolation_order < 0:
        args.interpolation_order = None

    if args.temp_directory is None:
        args.temp_directory = gettempdir()

    os.environ["TMPDIR"] = args.temp_directory
    if args.score not in MATCHING_EXHAUSTIVE_REGISTER:
        raise ValueError(
            f"score has to be one of {', '.join(MATCHING_EXHAUSTIVE_REGISTER.keys())}"
        )

    gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if args.gpu_indices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_indices

    if args.use_gpu:
        gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if gpu_devices is None:
            print(
                "No GPU indices provided and CUDA_VISIBLE_DEVICES is not set.",
                "Assuming device 0.",
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        args.gpu_indices = [
            int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        ]

    if args.tilt_angles is not None:
        if args.wedge_axes is None:
            raise ValueError("Need to specify --wedge_axes when --tilt_angles is set.")
        if not exists(args.tilt_angles):
            try:
                float(args.tilt_angles.split(",")[0])
            except ValueError:
                raise ValueError(f"{args.tilt_angles} is not a file nor a range.")

    if args.ctf_file is not None and args.tilt_angles is None:
        raise ValueError("Need to specify --tilt_angles when --ctf_file is set.")

    if args.wedge_axes is not None:
        args.wedge_axes = tuple(int(i) for i in args.wedge_axes.split(","))

    return args


def main():
    args = parse_args()
    print_entry()

    target = Density.from_file(args.target, use_memmap=True)

    try:
        template = Density.from_file(args.template)
    except Exception:
        template = Density.from_structure(
            filename_or_structure=args.template,
            sampling_rate=target.sampling_rate,
        )

    if target.sampling_rate.size == template.sampling_rate.size:
        if not np.allclose(
            np.round(target.sampling_rate, 2), np.round(template.sampling_rate, 2)
        ):
            print(
                f"Resampling template to {target.sampling_rate}. "
                "Consider providing a template with the same sampling rate as the target."
            )
            template = template.resample(target.sampling_rate, order=3)

    template_mask = load_and_validate_mask(
        mask_target=template, mask_path=args.template_mask
    )
    target_mask = load_and_validate_mask(
        mask_target=target, mask_path=args.target_mask, use_memmap=True
    )

    initial_shape = target.shape
    print_block(
        name="Target",
        data={
            "Initial Shape": initial_shape,
            "Sampling Rate": tuple(np.round(target.sampling_rate, 2)),
            "Final Shape": target.shape,
        },
    )

    if target_mask:
        print_block(
            name="Target Mask",
            data={
                "Initial Shape": initial_shape,
                "Sampling Rate": tuple(np.round(target_mask.sampling_rate, 2)),
                "Final Shape": target_mask.shape,
            },
        )

    initial_shape = template.shape
    translation = np.zeros(len(template.shape), dtype=np.float32)
    if not args.no_centering:
        template, translation = template.centered(0)
    print_block(
        name="Template",
        data={
            "Initial Shape": initial_shape,
            "Sampling Rate": tuple(np.round(template.sampling_rate, 2)),
            "Final Shape": template.shape,
        },
    )

    if template_mask is None:
        template_mask = template.empty
        if not args.no_centering:
            enclosing_box = template.minimum_enclosing_box(
                0, use_geometric_center=False
            )
            template_mask.adjust_box(enclosing_box)

        template_mask.data[:] = 1
        translation = np.zeros_like(translation)

    template_mask.pad(template.shape, center=False)
    origin_translation = np.divide(
        np.subtract(template.origin, template_mask.origin), template.sampling_rate
    )
    translation = np.add(translation, origin_translation)

    template_mask = template_mask.rigid_transform(
        rotation_matrix=np.eye(template_mask.data.ndim),
        translation=-translation,
        order=1,
    )
    template_mask.origin = template.origin.copy()
    print_block(
        name="Template Mask",
        data={
            "Inital Shape": initial_shape,
            "Sampling Rate": tuple(np.round(template_mask.sampling_rate, 2)),
            "Final Shape": template_mask.shape,
        },
    )
    print("\n" + "-" * 80)

    if args.scramble_phases:
        template.data = scramble_phases(
            template.data, noise_proportion=1.0, normalize_power=False
        )

    # Determine suitable backend for the selected operation
    available_backends = be.available_backends()
    if args.backend is not None:
        req_backend = args.backend
        if req_backend not in available_backends:
            raise ValueError("Requested backend is not available.")
        available_backends = [req_backend]

    be_selection = ("numpyfftw", "pytorch", "jax", "mlx")
    if args.use_gpu:
        args.cores = len(args.gpu_indices)
        be_selection = ("pytorch", "cupy", "jax")
    if args.use_mixed_precision:
        be_selection = tuple(x for x in be_selection if x in ("cupy", "numpyfftw"))

    available_backends = [x for x in available_backends if x in be_selection]
    if args.peak_calling:
        if "jax" in available_backends:
            available_backends.remove("jax")
        if args.use_gpu and "pytorch" in available_backends:
            available_backends = ("pytorch",)

    # dim_match = len(template.shape) == len(target.shape) <= 3
    # if dim_match and args.use_gpu and "jax" in available_backends:
    #     args.interpolation_order = 1
    #     available_backends = ["jax"]

    backend_preference = ("numpyfftw", "pytorch", "jax", "mlx")
    if args.use_gpu:
        backend_preference = ("cupy", "pytorch", "jax")
    for pref in backend_preference:
        if pref not in available_backends:
            continue
        be.change_backend(pref)
        if pref == "pytorch":
            be.change_backend(pref, device="cuda" if args.use_gpu else "cpu")

        if args.use_mixed_precision:
            be.change_backend(
                backend_name=pref,
                default_dtype=be._array_backend.float16,
                complex_dtype=be._array_backend.complex64,
                default_dtype_int=be._array_backend.int16,
            )
        break

    if pref == "pytorch" and args.interpolation_order == 3:
        warnings.warn(
            "Pytorch does not support --interpolation_order 3, setting it to 1."
        )
        args.interpolation_order = 1

    available_memory = be.get_available_memory() * be.device_count()
    if args.memory is None:
        args.memory = int(args.memory_scaling * available_memory)

    callback_class = MaxScoreOverRotations
    if args.peak_calling:
        callback_class = PeakCallerMaximumFilter

    matching_data = MatchingData(
        target=target,
        template=template.data,
        target_mask=target_mask,
        template_mask=template_mask,
        invert_target=args.invert_target_contrast,
        rotations=parse_rotation_logic(args=args, ndim=template.data.ndim),
    )

    matching_setup, matching_score = MATCHING_EXHAUSTIVE_REGISTER[args.score]
    matching_data.template_filter, matching_data.target_filter = setup_filter(
        args, template, target
    )

    matching_data.set_matching_dimension(
        target_dim=target.metadata.get("batch_dimension", None),
        template_dim=template.metadata.get("batch_dimension", None),
    )
    splits, schedule = compute_schedule(args, matching_data, callback_class)

    n_splits = np.prod(list(splits.values()))
    target_split = ", ".join(
        [":".join([str(x) for x in axis]) for axis in splits.items()]
    )
    gpus_used = 0 if args.gpu_indices is None else len(args.gpu_indices)
    options = {
        "Angular Sampling": f"{args.angular_sampling}"
        f" [{matching_data.rotations.shape[0]} rotations]",
        "Center Template": not args.no_centering,
        "Scramble Template": args.scramble_phases,
        "Invert Contrast": args.invert_target_contrast,
        "Extend Target Edges": args.pad_edges,
        "Interpolation Order": args.interpolation_order,
        "Setup Function": f"{get_func_fullname(matching_setup)}",
        "Scoring Function": f"{get_func_fullname(matching_score)}",
    }

    print_block(
        name="Template Matching",
        data=options,
        label_width=max(len(key) for key in options.keys()) + 3,
    )

    compute_options = {
        "Backend": be._BACKEND_REGISTRY[be._backend_name],
        "Compute Devices": f"CPU [{args.cores}], GPU [{gpus_used}]",
        "Use Mixed Precision": args.use_mixed_precision,
        "Assigned Memory [MB]": f"{args.memory // 1e6} [out of {available_memory//1e6}]",
        "Temporary Directory": args.temp_directory,
        "Target Splits": f"{target_split} [N={n_splits}]",
    }
    print_block(
        name="Computation",
        data=compute_options,
        label_width=max(len(key) for key in options.keys()) + 3,
    )

    filter_args = {
        "Lowpass": args.lowpass,
        "Highpass": args.highpass,
        "Smooth Pass": args.no_pass_smooth,
        "Pass Format": args.pass_format,
        "Spectral Whitening": args.whiten_spectrum,
        "Wedge Axes": args.wedge_axes,
        "Tilt Angles": args.tilt_angles,
        "Tilt Weighting": args.tilt_weighting,
        "Reconstruction Filter": args.reconstruction_filter,
        "Extend Filter Grid": args.pad_filter,
    }
    if args.ctf_file is not None or args.defocus is not None:
        filter_args["CTF File"] = args.ctf_file
        filter_args["Defocus"] = args.defocus
        filter_args["Phase Shift"] = args.phase_shift
        filter_args["Flip Phase"] = args.no_flip_phase
        filter_args["Acceleration Voltage"] = args.acceleration_voltage
        filter_args["Spherical Aberration"] = args.spherical_aberration
        filter_args["Amplitude Contrast"] = args.amplitude_contrast
        filter_args["Correct Defocus"] = args.correct_defocus_gradient

    filter_args = {k: v for k, v in filter_args.items() if v is not None}
    if len(filter_args):
        print_block(
            name="Filters",
            data=filter_args,
            label_width=max(len(key) for key in options.keys()) + 3,
        )

    analyzer_args = {
        "score_threshold": args.score_threshold,
        "number_of_peaks": args.number_of_peaks,
        "min_distance": max(template.shape) // 3,
        "use_memmap": args.use_memmap,
    }
    print_block(
        name="Analyzer",
        data={"Analyzer": callback_class, **analyzer_args},
        label_width=max(len(key) for key in options.keys()) + 3,
    )
    print("\n" + "-" * 80)

    outer_jobs = f"{schedule[0]} job{'s' if schedule[0] > 1 else ''}"
    inner_jobs = f"{schedule[1]} core{'s' if schedule[1] > 1 else ''}"
    n_splits = f"{n_splits} split{'s' if n_splits > 1 else ''}"
    print(f"\nDistributing {n_splits} on {outer_jobs} each using {inner_jobs}.")

    start = time()
    print("Running Template Matching. This might take a while ...")
    candidates = scan_subsets(
        matching_data=matching_data,
        job_schedule=schedule,
        matching_score=matching_score,
        matching_setup=matching_setup,
        callback_class=callback_class,
        callback_class_args=analyzer_args,
        target_splits=splits,
        pad_target_edges=args.pad_edges,
        pad_template_filter=args.pad_filter,
        interpolation_order=args.interpolation_order,
    )

    candidates = list(candidates) if candidates is not None else []
    if issubclass(callback_class, MaxScoreOverRotations):
        if target_mask is not None and args.score != "MCC":
            candidates[0] *= target_mask.data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            nbytes = be.datatype_bytes(be._float_dtype)
            dtype = np.float32 if nbytes == 4 else np.float16
            rot_dim = matching_data.rotations.shape[1]
            candidates[3] = {
                x: np.frombuffer(i, dtype=dtype).reshape(rot_dim, rot_dim)
                for i, x in candidates[3].items()
            }
    candidates.append((target.origin, template.origin, template.sampling_rate, args))
    write_pickle(data=candidates, filename=args.output)

    runtime = time() - start
    print("\n" + "-" * 80)
    print(f"\nRuntime real: {runtime:.3f}s user: {(runtime * args.cores):.3f}s.")


if __name__ == "__main__":
    main()
