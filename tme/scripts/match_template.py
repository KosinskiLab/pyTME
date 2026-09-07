#!python3
"""CLI for basic pyTME template matching functions.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import os
import argparse
import warnings
from sys import exit
from time import time
from os.path import exists
from tempfile import gettempdir

import numpy as np

from tme.backends import backend as be
from tme.matching_data import MatchingData
from tme.matching_exhaustive import match_exhaustive
from tme.utils import cli, normal_field, serialization
from tme.matching_scores import MATCHING_EXHAUSTIVE_REGISTER
from tme.matching_utils import compute_extraction_box, create_mask
from tme import Density, __version__, Orientations, filters, analyzer
from tme.rotations import (
    align_vectors,
    get_rotation_matrices,
    reduce_rotations_by_symmetry,
    euler_to_rotationmatrix,
)


def parse_rotation_logic(args, ndim):
    if args.particle_diameter is not None:
        resolution = Density.from_file(args.target, use_memmap=True)
        resolution = 360 * np.maximum(
            np.max(2 * resolution.sampling_rate),
            args.lowpass if args.lowpass is not None else 0,
        )
        args.angular_sampling = resolution / (3.14159265358979 * args.particle_diameter)

    if args.angular_sampling >= 180:
        return np.eye(ndim).reshape(1, ndim, ndim)

    rotations = get_rotation_matrices(
        angular_sampling=args.angular_sampling,
        dim=ndim,
        use_optimized_set=True,
    )
    if ndim == 3 and args.symmetry not in (None, "C1"):
        rotations = reduce_rotations_by_symmetry(rotations, symmetry=args.symmetry)
    return rotations


def setup_filter(args, template: Density, target: Density):
    template_filter, target_filter = [], []

    wedge = None
    if args.tilt_angles is not None:
        try:
            wedge = filters.Wedge.from_file(args.tilt_angles)
            wedge.weight_type = args.tilt_weighting

            # Avoid reconstructing the 3D wedge from individual tilts
            if args.tilt_weighting in ("angle", None) and not args.match_projection:
                wedge = filters.WedgeReconstructed(
                    angles=wedge.angles,
                    weight_wedge=args.tilt_weighting == "angle",
                )

        except (FileNotFoundError, AttributeError):
            wedge = filters.WedgeReconstructed(
                angles=args.tilt_angles,
                create_continuous_wedge=len(args.tilt_angles) == 2,
                weight_wedge=False,
            )

        wedge.sampling_rate = template.sampling_rate
        wedge.opening_axis, wedge.tilt_axis = args.wedge_axes
        template_filter.append(wedge)

        # When projection matching we can reuse the template wedge mask
        wedge_target = wedge
        if not args.match_projection:
            wedge_target = filters.WedgeReconstructed(
                angles=wedge.angles,
                weight_wedge=False,
                create_continuous_wedge=True,
                opening_axis=wedge.opening_axis,
                tilt_axis=wedge.tilt_axis,
            )

            wedge_target.sampling_rate = template.sampling_rate
        else:
            n_angles, n_tilts = len(wedge_target.angles), target.shape[0]
            if n_angles != n_tilts:
                raise ValueError(
                    f"Target contains {n_tilts} tilts, but the input specified "
                    f"{n_angles} tilt angles."
                )
        target_filter.append(wedge_target)

    if args.ctf_file is not None or args.defocus is not None:
        ctf_kw = {
            "spherical_aberration": args.spherical_aberration,
            "acceleration_voltage": args.acceleration_voltage,
            "amplitude_contrast": args.amplitude_contrast,
            "phase_shift": args.phase_shift,
        }
        ctf_kw = {k: v for k, v in ctf_kw.items() if v is not None}
        try:
            ctf = filters.CTF.from_file(args.ctf_file, **ctf_kw)
            # We ensure in parse_args that wedge will be valid here
            if ctf.angles is None or len(ctf.angles) == 0:
                ctf.angles = wedge.angles

            # There are several ways we can end up here. Bottom line, we are using
            # a non-reconstructed wedge, which contains a different number of tilts
            # than the ctf. We use defocus, as not all ctf_files specify angles.
            n_tilts_ctfs, n_tils_angles = len(ctf.defocus), len(wedge.angles)
            if (n_tilts_ctfs != n_tils_angles) and type(wedge) is filters.Wedge:
                raise ValueError(
                    f"CTF file contains {n_tilts_ctfs} tilt, but recieved "
                    f"{n_tils_angles} tilt angles. Expected one angle per tilt"
                )

        except (FileNotFoundError, AttributeError):
            ctf_cl = (
                filters.CTFReconstructed if not args.match_projection else filters.CTF
            )
            ctf = ctf_cl(defocus=args.defocus, **ctf_kw)

        ctf.correction_mode = args.ctf_correction

        # This only makes sense when we are using reconstruction gridding
        if not isinstance(ctf, filters.CTFReconstructed):
            if ctf.correction_mode == "phase-flip":
                ctf.correction_mode = "phase-flip-weighted"

        ctf.sampling_rate = template.sampling_rate
        ctf.opening_axis, ctf.tilt_axis = args.wedge_axes
        template_filter.append(ctf)

    if args.lowpass is not None or args.highpass is not None:
        if args.lowpass is not None and args.highpass is not None:
            if args.lowpass >= args.highpass:
                raise ValueError("--lowpass should be smaller than --highpass.")

        bp_cl = (
            filters.BandPassReconstructed
            if not args.match_projection
            else filters.BandPass
        )
        bandpass = bp_cl(
            use_gaussian=True,
            lowpass=args.lowpass,
            highpass=args.highpass,
            sampling_rate=template.sampling_rate,
        )
        bandpass.opening_axis, bandpass.tilt_axis = args.wedge_axes
        template_filter.append(bandpass)
        target_filter.append(bandpass)

    if not args.match_projection:
        rec_filt = (filters.Wedge, filters.CTF)
        needs_reconstruction = sum(type(x) in rec_filt for x in template_filter)
        template_filter = sorted(
            template_filter, key=lambda x: type(x) in rec_filt, reverse=True
        )
        if needs_reconstruction > 0:
            reconstruction_filter = filters.ReconstructFromTilt(
                angles=template_filter[0].angles,
                opening_axis=args.wedge_axes[0],
                tilt_axis=args.wedge_axes[1],
                method="gridding",
            )
            template_filter.insert(needs_reconstruction, reconstruction_filter)
    else:
        template_filter.append(filters.ShiftFourier())
        if len(target_filter):
            target_filter.append(filters.ShiftFourier())

    if args.whiten_spectrum:
        angles = None
        if wedge is not None and not args.match_projection:
            angles = getattr(wedge, "angles", None)
        # Boxes below the template edge would resolve less than the data we filter,
        # the patch count follows a fixed compute budget.
        patch_size = max(64, 1 << (max(template.shape) - 1).bit_length())
        patch_size = min(patch_size, 256, min(target.shape))
        max_patches = int(np.clip(round(1024 * 64**3 / patch_size**3), 64, 1024))

        profile = filters.estimate_radial_noise_spectrum(
            data=target.data,
            angles=angles,
            opening_axis=args.wedge_axes[0],
            tilt_axis=args.wedge_axes[1],
            patch_size=patch_size,
            max_patches=max_patches,
        )
        whitening_filter = filters.Curve(spectrum=profile)
        template_filter.append(whitening_filter)
        target_filter.append(whitening_filter)

    template_filter = filters.Compose(template_filter) if len(template_filter) else None
    target_filter = filters.Compose(target_filter) if len(target_filter) else None
    if args.no_filter_target:
        target_filter = None

    return template_filter, target_filter


def _format_sampling(arr, decimals: int = 2):
    return tuple(round(float(x), decimals) for x in arr)


def _resolve_orientation_scaling(scaling, sampling_rate, is_mesh: bool) -> float:
    """Resolve the coordinater scaling for orientation seed coordinates.

    Points default to voxel coordinates (factor 1.0). Meshes default to the
    target sampling rate, i.e. they are assumed to be in physical coordinates.
    """
    if scaling is not None:
        return float(scaling)
    if is_mesh:
        return float(np.max(sampling_rate))
    return 1.0


def add_matching_arguments(parser, batch_mode=False):
    io_group = parser.add_argument_group("Input / Output")
    constrain_group = parser.add_argument_group(
        "Constrained Matching",
        "Enforce translational and orientational constraints during matching.",
    )
    scoring_group = parser.add_argument_group("Scoring")
    angular_group = parser.add_argument_group(
        "Angular Sampling", "Define how to generate rotation sets for template search."
    )
    computation_group = parser.add_argument_group("Computation")
    filter_group = parser.add_argument_group("Filters")
    ctf_group = parser.add_argument_group("Contrast Transfer Function")
    performance_group = parser.add_argument_group("Performance")

    if not batch_mode:
        io_group.add_argument(
            "-m",
            "--target",
            type=cli.existing_file,
            required=True,
            help="Target (MRC, EM, H5, or other formats supported by Density).",
        )
        io_group.add_argument(
            "-M",
            "--target-mask",
            type=cli.existing_file,
            help="Target mask (same formats as --target).",
        )
        io_group.add_argument(
            "-o",
            "--output",
            type=str,
            default="output.pickle",
            help="Output file (.hdf5, .pickle or .pickle.gz).",
        )
        constrain_group.add_argument(
            "--orientations",
            type=cli.existing_file,
            help="Seed points with translations and rotations (STAR). "
            "Alternatively, a triangle mesh file (OBJ, PLY, STL). Point seeds default "
            "to voxel coordinates and meshes to Angstrom, both overridable "
            "with --orientations-scaling.",
        )
        filter_group.add_argument(
            "--tilt-angles",
            type=str,
            help="Path to a file specifying tilt angles. This can be a Warp/M XML file, "
            "a tomostar STAR file, an IMOD .tlt file. Alternatively, it is possible to "
            "specify a continuous wedge mask using, e.g., 50,45.",
        )
        ctf_group.add_argument(
            "--ctf-file",
            type=cli.existing_file,
            help="Path to a file with CTF parameters. This can be a Warp/M XML file "
            "a GCTF/Relion STAR file, an MDOC file, or the output of CTFFIND4. If the "
            "file does not specify tilt angles, --tilt-angles are used.",
        )

    io_group.add_argument(
        "-i",
        "--template",
        type=cli.existing_file,
        required=True,
        help="Template (PDB, MMCIF, GRO, and same formats as --target).",
    )
    io_group.add_argument(
        "-I",
        "--template-mask",
        type=cli.existing_file,
        help="Template mask (same formats as --target).",
    )
    io_group.add_argument(
        "--invert-target-contrast",
        action="store_true",
        help="Invert target contrast (multiply by -1).",
    )
    io_group.add_argument(
        "--sampling-rate",
        type=cli.check_bounded_dtype(float, 1e-6),
        default=None,
        help="Overwrite target sampling rate by this value. Use if the "
        "sampling rate / pixel size in the header is incorrect.",
    )
    constrain_group.add_argument(
        "--orientations-scaling",
        type=float,
        help="Angstrom-per-voxel factor dividing seed coordinates to voxels. "
        "Omit to assume points are in voxels and meshes in Angstrom.",
    )
    constrain_group.add_argument(
        "--orientations-cone",
        type=cli.check_bounded_dtype(float, 0, 90),
        default=None,
        help="Accept matches within cone angle (degrees) of seed orientation. "
        "Default is no orientational constraint.",
    )
    constrain_group.add_argument(
        "--orientations-uncertainty",
        type=str,
        default=None,
        help="Search radius around each seed in voxels, as a single value ('10') "
        "or per-axis ('10,15,10'). Meshes use a single value (distance from "
        "surface). Default searches only at each seed position.",
    )
    constrain_group.add_argument(
        "--orientations-mode",
        type=str,
        choices=["outward", "inward", "natural", "outside_only", "inside_only"],
        default="outward",
        help="How mesh surface is used to create orientation constraints. "
        "'outward': normals point away from mesh interior. "
        "'inward': normals point toward mesh interior. "
        "'natural': interior point normals point inwards and exterior outwards. "
        "'outside_only': only voxels outside mesh, normals point outward. "
        "'inside_only': only voxels inside mesh, normals point inward. "
        "Ignored for non-mesh input.",
    )
    constrain_group.add_argument(
        "--trust-vertex-normals",
        action="store_true",
        help="Use mesh vertex normals to set the sign of the signed distance field. "
        "Enable when the mesh has meaningful vertex normals. Ignored for non-mesh "
        "input.",
    )
    constrain_group.add_argument(
        "--orientations-offset",
        type=float,
        default=None,
        help="Offset seed points / meshes along their normal vectors in voxels.",
    )

    scoring_group.add_argument(
        "-s",
        "--score",
        type=str,
        default="FLCSphericalMask",
        choices=list(MATCHING_EXHAUSTIVE_REGISTER.keys()),
        help="Template matching scoring function.",
    )
    scoring_group.add_argument(
        "--background-correction",
        choices=["phase-scrambling"],
        help="Transform cross-correlation into SNR-like values using a given method: "
        "'phase-scrambling' uses a phase-scrambled template as background",
    )

    angular_exclusive = angular_group.add_mutually_exclusive_group(
        required=not batch_mode
    )
    angular_exclusive.add_argument(
        "-a",
        "--angular-sampling",
        type=cli.check_bounded_dtype(float, 0),
        help="Angular sampling in degrees between tested orientations. "
        "Smaller values test more orientations (slower but more thorough). "
        "10 tests ~7k, 5 ~53k rotations.",
    )
    angular_exclusive.add_argument(
        "--particle-diameter",
        type=cli.check_bounded_dtype(float, 0),
        help="Particle diameter in units of sampling rate. "
        "Automatically determines angular sampling based on size and resolution.",
    )
    angular_group.add_argument(
        "--symmetry",
        type=cli.check_symmetry,
        default="C1",
        help="Point group symmetry of the template about its z-axis, e.g. C4 or D2. "
        "Restricts the search to the symmetry's fundamental domain. Default C1.",
    )

    computation_group.add_argument(
        "-n",
        "--processes",
        type=cli.check_bounded_dtype(int, 1),
        default=4,
        help="Number of processes used for template matching.",
    )
    computation_group.add_argument(
        "--gpu-indices",
        type=str,
        default=os.environ.get("CUDA_VISIBLE_DEVICES"),
        help="Comma-separated GPU indices (default: CUDA_VISIBLE_DEVICES).",
    )
    computation_group.add_argument(
        "--memory",
        action=cli.DeprecatedAction,
        replacement="Use --memory-scaling.",
        help=argparse.SUPPRESS,
    )
    computation_group.add_argument(
        "--memory-scaling",
        type=cli.check_bounded_dtype(float, 0.0),
        default=0.85,
        help="Fraction of available memory to be used.",
    )
    computation_group.add_argument(
        "--temp-directory",
        help="Temporary directory for memmaps. Better I/O improves runtime.",
    )
    computation_group.add_argument(
        "--backend",
        default=be._backend_name,
        choices=be.available_backends(),
        help="Set computation backend.",
    )
    computation_group.add_argument(
        "--scheduling-mode",
        type=str,
        default="uniform",
        choices=["uniform", "subdivide"],
        help="How to schedule the search over the target. 'uniform' searches the "
        "bounding box of --scheduling-mask (the full target if no mask is given). "
        "'subdivide' recursively decomposes the mask into sub-regions to skip empty "
        "space and avoid scoring positions that cannot contain a match.",
    )
    computation_group.add_argument(
        "--scheduling-mask",
        type=cli.existing_file,
        help="Binary mask marking where a match may occur, used to restrict the "
        "search. Derived from --orientations if not given. Required for "
        "--scheduling-mode subdivide.",
    )

    filter_group.add_argument(
        "--lowpass",
        type=cli.check_bounded_dtype(float, 0),
        help="Resolution to lowpass filter template and target to.",
    )
    filter_group.add_argument(
        "--highpass",
        type=cli.check_bounded_dtype(float, 0),
        help="Resolution to highpass filter template and target to.",
    )
    filter_group.add_argument(
        "--whiten-spectrum",
        action="store_true",
        help="Whiten template and target spectra.",
    )
    filter_group.add_argument(
        "--wedge-axes",
        type=str,
        default="2,0",
        help="Indices of projection (wedge opening) and tilt axis, e.g., '2,0' "
        "for the typical projection over z and tilting over the x-axis.",
    )
    filter_group.add_argument(
        "--tilt-weighting",
        type=str,
        choices=["angle", "relion", "grigorieff"],
        help="Tilt weighting schemes. Available options: "
        "angle (cosine based weighting), "
        "relion (relion formalism for exposure) requires,"
        "grigorieff (Grant and Grigorieff 2015 exposure formalism). "
        "relion and grigorieff require electron doses in --tilt-angles weights column.",
    )
    filter_group.add_argument(
        "--no-filter-target",
        action="store_true",
        help="Whether to not apply potential filters to the target.",
    )

    ctf_group.add_argument(
        "--defocus",
        type=float,
        help="Defocus in units of sampling rate (typically Ångstrom). "
        "Superseded by --ctf-file.",
    )
    ctf_group.add_argument(
        "--phase-shift",
        type=float,
        help="Phase shift in degrees. Defaults to 0. Superseded by --ctf-file.",
    )
    ctf_group.add_argument(
        "--acceleration-voltage",
        type=cli.check_bounded_dtype(float, 0.0),
        help="Acceleration voltage in kV. Defaults to 300",
    )
    ctf_group.add_argument(
        "--spherical-aberration",
        type=cli.check_bounded_dtype(float, 0.0),
        help="Spherical aberration in units of sampling rate (typically Ångstrom). "
        "Defaults to 27000000.0 Å",
    )
    ctf_group.add_argument(
        "--amplitude-contrast",
        type=cli.check_bounded_dtype(float, 0.0),
        help="Amplitude contrast Defaults to 0.07.",
    )
    ctf_group.add_argument(
        "--ctf-correction",
        type=str,
        default="phase-flip",
        choices=["phase-flip", "raw"],
        help="'phase-flip' for phase-flip corrected targets (default), "
        "'raw' for targets without CTF correction.",
    )

    performance_group.add_argument(
        "--centering",
        action="store_true",
        help="Translate the template's center of mass to the center of the box.",
    )
    performance_group.add_argument(
        "--pad-edges",
        action="store_true",
        help="Zero pad the target. Automatically set if target needs to be split.",
    )
    performance_group.add_argument(
        "--interpolation-order",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        help="Spline order for rotation, default is 3 and 1 for jax and pytorch.",
    )
    performance_group.add_argument(
        "--use-memmap",
        action="store_true",
        help="Memmap analyzer data, useful for matching on very large inputs.",
    )

    # Deprecated with v0.3.4
    filter_group.add_argument(
        "--pass-format",
        type=str,
        required=False,
        choices=["sampling_rate", "voxel", "frequency"],
        action=cli.DeprecatedAction,
        help=argparse.SUPPRESS,
    )
    ctf_group.add_argument(
        "--no-flip-phase",
        required=False,
        action=cli.DeprecatedAction,
        help=argparse.SUPPRESS,
    )
    filter_group.add_argument(
        "--reconstruction-filter",
        type=str,
        choices=["ram-lak", "ramp", "ramp-cont", "shepp-logan", "cosine", "hamming"],
        action=cli.DeprecatedAction,
        help=argparse.SUPPRESS,
    )
    filter_group.add_argument(
        "--reconstruction-interpolation-order",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        action=cli.DeprecatedAction,
        help=argparse.SUPPRESS,
    )
    angular_group.add_argument(
        "--invert-cone",
        action=cli.DeprecatedAction,
        help=argparse.SUPPRESS,
    )
    _cone_hint = (
        "Use --symmetry for symmetric particles, or --orientations with "
        "--orientations-cone for per-position angular constraints."
    )
    angular_group.add_argument(
        "--cone-sampling",
        action=cli.DeprecatedAction,
        replacement="Use -a / --angular-sampling.",
        help=argparse.SUPPRESS,
    )
    angular_group.add_argument(
        "--cone-angle",
        action=cli.DeprecatedAction,
        replacement=_cone_hint,
        help=argparse.SUPPRESS,
    )
    angular_group.add_argument(
        "--cone-axis", action=cli.DeprecatedAction, help=argparse.SUPPRESS
    )
    angular_group.add_argument(
        "--axis-angle", action=cli.DeprecatedAction, help=argparse.SUPPRESS
    )
    angular_group.add_argument(
        "--axis-sampling", action=cli.DeprecatedAction, help=argparse.SUPPRESS
    )
    angular_group.add_argument(
        "--axis-symmetry",
        action=cli.DeprecatedAction,
        replacement=_cone_hint,
        help=argparse.SUPPRESS,
    )

    meta = cli.ArgumentMetadata()
    groups = [
        io_group,
        constrain_group,
        scoring_group,
        angular_group,
        computation_group,
        filter_group,
        ctf_group,
        performance_group,
    ]

    for group in groups:
        meta.update(group)
    return meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform template matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_matching_arguments(parser, batch_mode=False)
    args = parser.parse_args()
    args.version = __version__
    args.match_projection = False

    if args.temp_directory is None:
        args.temp_directory = gettempdir()
    os.environ["TMPDIR"] = args.temp_directory

    # Tilt angles can be specified as range or using a suitable input file
    is_file = exists(args.tilt_angles) if args.tilt_angles is not None else False
    if args.tilt_angles is not None and not is_file:
        try:
            args.tilt_angles = tuple(abs(float(x)) for x in args.tilt_angles.split(","))
        except Exception:
            raise ValueError(f"{args.tilt_angles} is not a file nor a range.")

    # Since both Wedge.from_file and CTF.from_file parse similar inputs, we can
    # fall back to assigning the ctf_file to args.tilt_angles
    if args.ctf_file is not None and args.tilt_angles is None:
        ctf = filters.CTF.from_file(args.ctf_file)
        if ctf.angles is None:
            raise ValueError(
                "Need to specify --tilt-angles when not provided in --ctf-file."
            )
        args.tilt_angles = args.ctf_file

    # For projection matching we cannot use continuous wedge masks
    if not is_file and args.match_projection:
        raise ValueError(
            "Projection angles are required via --tilt-angles or --ctf-file."
        )

    if args.orientations is not None and args.orientations_uncertainty is None:
        raise ValueError("--orientations-uncertainty is required for --orientations.")

    # Enforce constraint to be tuple of len 3 as (int, int, int)
    if args.orientations_uncertainty is not None:
        args.orientations_uncertainty = tuple(
            int(x) for x in args.orientations_uncertainty.split(",")
        )
        if len(args.orientations_uncertainty) not in (1, 3):
            raise ValueError(
                "--orientations-uncertainty can either be one or one value per-axis."
            )
        if len(args.orientations_uncertainty) == 1:
            args.orientations_uncertainty = args.orientations_uncertainty * 3

    # Handle backend specificities
    if args.interpolation_order is None:
        args.interpolation_order = 3
        if args.backend in ("jax", "pytorch"):
            args.interpolation_order = 1

    # This flag is not passed to backend yet, but might aswell be verbose about it
    if args.interpolation_order != 1 and args.backend == "jax":
        warnings.warn("Setting interpolation order to order jax supports (1).")
        args.interpolation_order = 1

    if args.interpolation_order == 3 and args.backend == "pytorch":
        warnings.warn("Pytorch does not support order 3, changing it to 1.")
        args.interpolation_order = 1

    # Handle GPU device specification for suitable backends
    if args.backend in ("pytorch", "cupy", "jax"):
        if args.gpu_indices is None:
            warnings.warn(
                "No GPU indices provided and CUDA_VISIBLE_DEVICES is not set. "
                "Assuming device 0.",
            )
            args.gpu_indices = "0"

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_indices
        args.gpu_indices = [int(x) for x in args.gpu_indices.split(",")]
        args.processes = len(args.gpu_indices)

    # Wedge axes do not have meaning for projections
    args.wedge_axes = tuple(int(i) for i in args.wedge_axes.split(","))
    if args.match_projection:
        args.wedge_axes = None, None

    if args.match_projection and args.backend != "jax":
        raise ValueError("Projection matching is only supported for --backend jax.")

    if args.scheduling_mode != "uniform":
        args.pad_edges = True
        if args.scheduling_mask is None and args.orientations is None:
            raise ValueError(
                f"Scheduling mode '{args.scheduling_mode}' requires "
                "--scheduling-mask or --orientations."
            )
    return args


def main():
    args = parse_args()
    cli.print_entry()

    target = Density.from_file(args.target, use_memmap=True)
    try:
        template = Density.from_file(args.template)
    except Exception:
        template = Density.from_structure(
            filename_or_structure=args.template,
            sampling_rate=target.sampling_rate,
        )

    if np.allclose(target.sampling_rate, 1):
        warnings.warn(
            "Target sampling rate is 1.0, which may indicate missing or incorrect "
            "metadata. Verify that your target file contains proper sampling rate "
            "information, as filters (CTF, Wedge, BandPass) require accurate "
            "sampling rates to function correctly."
        )

    template_mask = cli.load_and_validate_mask(template, args.template_mask)
    target_mask = cli.load_and_validate_mask(target, args.target_mask, use_memmap=True)
    if args.sampling_rate is not None:
        target.sampling_rate = args.sampling_rate
        if target_mask is not None:
            target_mask.sampling_rate = args.sampling_rate

    if target.sampling_rate.size == template.sampling_rate.size:
        sampling_rate_match = np.allclose(
            np.round(target.sampling_rate, 2), np.round(template.sampling_rate, 2)
        )
        # For projection we omit the warning as the leading dimension has no sampling
        if not sampling_rate_match and not args.match_projection:
            warnings.warn(
                f"Sampling rate mismatch detected: target={target.sampling_rate} "
                f"template={template.sampling_rate}. Proceeding with user-provided "
                f"values. Make sure this is intentional."
            )

    cli.print_block(
        name="Target",
        data={
            "Shape": target.shape,
            "Sampling Rate": _format_sampling(target.sampling_rate),
        },
    )

    if target_mask:
        cli.print_block(
            name="Target Mask",
            data={
                "Shape": target_mask.shape,
                "Sampling Rate": _format_sampling(target_mask.sampling_rate),
            },
        )

    if args.centering:
        template = template.centered(0)

    cli.print_block(
        name="Template",
        data={
            "Shape": template.shape,
            "Sampling Rate": _format_sampling(template.sampling_rate),
        },
    )

    if template_mask is None:
        template_mask = template.empty

        # Pre 0.3.2 we used to perform a rigid transform on the template mask to match
        # the template origin, but this seems overly pedantic given the sporadic use
        # of the origin parameter in the matching pipeline
        template_mask.data = np.ones(template.shape, dtype=template.data.dtype)

    cli.print_block(
        name="Template Mask",
        data={
            "Shape": template_mask.shape,
            "Sampling Rate": _format_sampling(template_mask.sampling_rate),
        },
    )
    print("\n" + "-" * 80)

    callback_class = analyzer.MaxScoreOverRotations
    if args.orientations is not None:
        callback_class = analyzer.MaxScoreOverRotationsConstrained

    # We currently do not allow parallelizing angular searches in the GPU compatible
    # backends, so we keep this flag to compute a suitable splitting schedule
    args.use_gpu = False
    if args.backend in ("jax", "pytorch", "cupy"):
        args.use_gpu = True
    be.change_backend(args.backend, device="cuda" if args.use_gpu else "cpu")

    available_memory = be.get_available_memory() * be.device_count()
    if args.memory is None:
        args.memory = int(args.memory_scaling * available_memory)

    matching_data = MatchingData(
        target=target,
        template=template.data,
        target_mask=target_mask,
        template_mask=template_mask,
        invert_target=args.invert_target_contrast,
        rotations=parse_rotation_logic(args=args, ndim=template.data.ndim),
    )
    matching_data.set_matching_dimension(
        target_batched=target.metadata.get("batch_dimension", None) is not None,
        template_batched=template.metadata.get("batch_dimension", None) is not None,
    )
    if args.match_projection:
        matching_data.set_matching_dimension(target_batched=True)

    args.batch_dims = (0, 1) if matching_data._has_batch else ()
    matching_setup, matching_score = MATCHING_EXHAUSTIVE_REGISTER[args.score]
    matching_data.template_filter, matching_data.target_filter = setup_filter(
        args, template, target
    )

    analyzer_args = {"use_memmap": args.use_memmap}

    # Handle constrained matching inputs
    target_subset, mask, mask_spacing = None, None, 1
    if args.orientations is not None:
        analyzer_args["reference"] = (0, 0, 1)
        analyzer_args["cone_angle"] = args.orientations_cone

        # Orientations specified using seed points
        try:
            ori = Orientations.from_file(args.orientations)
            translations = ori.translations
            scaling = _resolve_orientation_scaling(
                args.orientations_scaling, target.sampling_rate, is_mesh=False
            )
            if args.orientations_scaling is None:
                print(
                    "Seed points assumed to be in voxel coordinates. Pass "
                    "--orientations-scaling for physical (e.g. Angstrom) coordinates."
                )
            else:
                print(f"Converting seed points to voxels with scaling {scaling}.")
            translations = np.divide(translations, scaling)

            analyzer_args["acceptance_radius"] = args.orientations_uncertainty
            rotations = euler_to_rotationmatrix(ori.rotations, seq="ZYZ")

            # Apply offset along orientation direction
            if args.orientations_offset is not None:
                normals = rotations.T @ np.array(analyzer_args["reference"])
                translations = np.add(translations, normals * args.orientations_offset)

        # Orientations specified using mesh
        except ValueError:
            scaling = _resolve_orientation_scaling(
                args.orientations_scaling, target.sampling_rate, is_mesh=True
            )
            if args.orientations_scaling is None:
                print(
                    "Mesh input assumed to be in physical coordinates, normalizing "
                    f"by target sampling rate {scaling:.3f} Angstrom/voxel. Override "
                    "with --orientations-scaling."
                )
            else:
                print(f"Normalizing mesh vertices to voxels with scaling {scaling}.")

            translations, normals = normal_field.compute_normal_field(
                mesh=args.orientations,
                voxel_size=scaling,
                height=args.orientations_uncertainty[-1],
                mode=args.orientations_mode,
                trust_vertex_normals=args.trust_vertex_normals,
                normal_offset=args.orientations_offset,
            )

            analyzer_args["unique_positions"] = True
            analyzer_args["acceptance_radius"] = None

            # Map local coordinate system into global reference, i.e.,
            # rotations.T @ analyzer_args["reference"] = normals
            rotations = align_vectors(normals, analyzer_args["reference"])

        valid = np.all((translations >= 0) & (translations < target.shape), axis=1)
        analyzer_args["positions"] = translations[valid].astype(np.int32)
        analyzer_args["rotations"] = rotations[valid].astype(np.float32)

        # Only translational uncertainty as the subset is later padded
        margin = 0
        if analyzer_args["acceptance_radius"] is not None:
            margin += np.max(args.orientations_uncertainty)

        lower_bound = analyzer_args["positions"].min(axis=0) - 1
        lower_bound = np.subtract(lower_bound, margin)

        upper_bound = analyzer_args["positions"].max(axis=0) + 1
        upper_bound = np.add(upper_bound, margin)

        lower_bound = np.maximum(lower_bound.astype(int), 0)
        upper_bound = np.minimum(upper_bound.astype(int), matching_data._target.shape)
        target_subset = tuple(
            slice(int(x), int(y)) for x, y in zip(lower_bound, upper_bound)
        )

    if args.scheduling_mask is not None:
        mask = Density.from_file(args.scheduling_mask, use_memmap=True)
        mask_spacing = np.divide(mask.sampling_rate, target.sampling_rate)
        mask = mask.data

    # Orientations is guaranteed to be defined in this case due to check in parse_args
    # We do not create this mask for uniform scheduling as the added detail is
    # not considered in the computation schedule.
    if args.scheduling_mode != "uniform" and args.scheduling_mask is None:
        ndim = analyzer_args["positions"].shape[-1]
        acceptance_radius = analyzer_args.get("acceptance_radius")

        mask = np.zeros(target.shape, dtype=bool)
        if acceptance_radius is None:
            mask[tuple(analyzer_args["positions"].T)] = 1
        else:
            extend = max(acceptance_radius)
            mask_center = tuple(extend for _ in range(ndim))
            mask_shape = tuple(2 * extend + 1 for _ in range(ndim))

            sc_beg, sc_end, tp_beg, tp_end, keep = compute_extraction_box(
                centers=be.to_backend_array(analyzer_args["positions"]),
                extraction_shape=mask_shape,
                original_shape=target.shape,
            )
            for i in range(analyzer_args["rotations"].shape[0]):
                tmpl_mask = create_mask(
                    mask_type="ellipse",
                    radius=acceptance_radius,
                    shape=mask_shape,
                    center=mask_center,
                    orientation=analyzer_args["rotations"][i].T,
                )

                slc = tuple(slice(int(x), int(y)) for x, y in zip(sc_beg[i], sc_end[i]))
                tp_slc = tuple(
                    slice(int(x), int(y)) for x, y in zip(tp_beg[i], tp_end[i])
                )
                mask[slc] |= tmpl_mask[tp_slc] > 0

    splits, schedule = matching_data.computation_schedule(
        matching_method=args.score,
        analyzer_method=callback_class.__name__,
        split_only_outer=args.use_gpu,
        pad_fourier=False,
        pad_target_edges=True,
        max_memory=args.memory,
        max_workers=args.processes,
        mode=args.scheduling_mode,
        target_subset=target_subset,
        mask=mask,
        mask_spacing=mask_spacing,
    )
    if len(splits) == 0:
        exit(
            "Found no suitable parallelization schedule for current settings. "
            "You can retry after increasing --memory or --memory-scaling."
        )
    elif len(splits) > 1 and not args.pad_edges:
        warnings.warn("Setting --pad-edges to avoid artifacts from splitting.")
        args.pad_edges = True

    options = {
        "Angular Sampling": f"{args.angular_sampling}"
        f" [{matching_data.rotations.shape[0]} rotations]",
        "Center Template": args.centering,
        "Symmetry": args.symmetry,
        "Background Correction": args.background_correction,
        "Invert Contrast": args.invert_target_contrast,
        "Extend Target Edges": args.pad_edges,
        "Interpolation Order": args.interpolation_order,
        "Setup Function": f"{cli.get_func_fullname(matching_setup)}",
        "Scoring Function": f"{cli.get_func_fullname(matching_score)}",
    }
    cli.print_block(
        name="Template Matching",
        data=options,
        label_width=max(len(key) for key in options.keys()) + 3,
    )

    n_splits = len(splits)
    gpus_used = 0 if args.gpu_indices is None else len(args.gpu_indices)
    compute_options = {
        "Backend": be._BACKEND_REGISTRY[be._backend_name],
        "Compute Devices": f"CPU [{args.processes}], GPU [{gpus_used}]",
        "Assigned Memory [MB]": f"{args.memory // 1e6} [out of {available_memory//1e6}]",
        "Temporary Directory": args.temp_directory,
        "Target Splits": n_splits,
    }
    cli.print_block(
        name="Computation",
        data=compute_options,
        label_width=max(len(key) for key in options.keys()) + 3,
    )

    filter_args = {
        "Lowpass": args.lowpass,
        "Highpass": args.highpass,
        "Spectral Whitening": args.whiten_spectrum,
        "Wedge Axes": args.wedge_axes,
        "Tilt Angles": args.tilt_angles,
        "Tilt Weighting": args.tilt_weighting,
    }
    if args.ctf_file is not None or args.defocus is not None:
        filter_args["CTF File"] = args.ctf_file
        filter_args["CTF Correction"] = args.ctf_correction

    filter_args = {k: v for k, v in filter_args.items() if v is not None}
    if len(filter_args):
        cli.print_block(
            name="Filters",
            data=filter_args,
            label_width=max(len(key) for key in options.keys()) + 3,
        )

    cli.print_block(
        name="Analyzer",
        data={
            "Analyzer": callback_class,
            **{cli.sanitize_name(k): v for k, v in analyzer_args.items()},
        },
        label_width=max(len(key) for key in options.keys()) + 3,
    )
    print("\n" + "-" * 80)

    outer_jobs = f"{schedule[0]} job{'s' if schedule[0] > 1 else ''}"
    inner_jobs = f"{schedule[1]} core{'s' if schedule[1] > 1 else ''}"
    n_splits = f"{n_splits} split{'s' if n_splits > 1 else ''}"
    print(f"\nDistributing {n_splits} on {outer_jobs} each using {inner_jobs}.")

    start = time()
    print("Running Template Matching. This might take a while ...")
    candidates = match_exhaustive(
        matching_data=matching_data,
        job_schedule=schedule,
        matching_score=matching_score,
        matching_setup=matching_setup,
        callback_class=callback_class,
        callback_class_args=analyzer_args,
        target_splits=splits,
        pad_target_edges=args.pad_edges,
        interpolation_order=args.interpolation_order,
        match_projection=args.match_projection,
        background_correction=args.background_correction,
    )

    candidates = list(candidates) if candidates is not None else []
    candidates.append((target.origin, template.origin, template.sampling_rate, args))
    serialization.serialize(data=candidates, filename=args.output)

    runtime = time() - start
    print("\n" + "-" * 80)
    print(f"\nRuntime real: {runtime:.3f}s user: {(runtime * args.processes):.3f}s.")


if __name__ == "__main__":
    main()
