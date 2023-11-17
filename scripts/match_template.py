#!python3
""" CLI interface for basic pyTME template matching functions.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import os
import argparse
import warnings
import importlib.util
from sys import exit
from time import time
from copy import deepcopy
from os.path import abspath

import numpy as np

from tme import Density, Preprocessor, __version__
from tme.matching_utils import (
    get_rotation_matrices,
    compute_parallelization_schedule,
    euler_from_rotationmatrix,
    scramble_phases,
    generate_tempfile_name,
    write_pickle,
)
from tme.matching_exhaustive import scan_subsets, MATCHING_EXHAUSTIVE_REGISTER
from tme.matching_data import MatchingData
from tme.analyzer import (
    MaxScoreOverRotations,
    PeakCallerMaximumFilter,
)
from tme.backends import backend


def get_func_fullname(func) -> str:
    """Returns the full name of the given function, including its module."""
    return f"<function '{func.__module__}.{func.__name__}'>"


def print_block(name: str, data: dict, label_width=20) -> None:
    """Prints a formatted block of information."""
    print(f"\n> {name}")
    for key, value in data.items():
        formatted_value = str(value)
        print(f"  - {key + ':':<{label_width}} {formatted_value}")


def print_entry() -> None:
    width = 80
    text = f" pyTME v{__version__} "
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
        if not np.allclose(mask.sampling_rate, mask_target.sampling_rate):
            raise ValueError(
                f"Expected sampling_rate of {mask_path} was {mask_target.sampling_rate}"
                f", got f{mask.sampling_rate}"
            )
    return mask


def crop_data(data: Density, cutoff: float, data_mask: Density = None) -> bool:
    """
    Crop the provided data and mask to a smaller box based on a cutoff value.

    Parameters
    ----------
    data : Density
        The data that should be cropped.
    cutoff : float
        The threshold value to determine which parts of the data should be kept.
    data_mask : Density, optional
        A mask for the data that should be cropped.

    Returns
    -------
    bool
        Returns True if the data was adjusted (cropped), otherwise returns False.

    Notes
    -----
    Cropping is performed in place.
    """
    if cutoff is None:
        return False

    box = data.trim_box(cutoff=cutoff)
    box_mask = box
    if data_mask is not None:
        box_mask = data_mask.trim_box(cutoff=cutoff)
    box = tuple(
        slice(min(arr.start, mask.start), max(arr.stop, mask.stop))
        for arr, mask in zip(box, box_mask)
    )
    if box == tuple(slice(0, x) for x in data.shape):
        return False

    data.adjust_box(box)

    if data_mask:
        data_mask.adjust_box(box)

    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Perform template matching.")
    parser.add_argument(
        "-m",
        "--target",
        dest="target",
        type=str,
        required=True,
        help="Path to a target in CCP4/MRC format.",
    ),
    parser.add_argument(
        "--target_mask",
        dest="target_mask",
        type=str,
        required=False,
        help="Path to a mask for the target target in CCP4/MRC format.",
    ),
    parser.add_argument(
        "--cutoff_target",
        dest="cutoff_target",
        type=float,
        required=False,
        help="Target contour level (used for cropping).",
        default=None,
    ),
    parser.add_argument(
        "--cutoff_template",
        dest="cutoff_template",
        type=float,
        required=False,
        help="Template contour level (used for cropping).",
        default=None,
    ),
    parser.add_argument(
        "-i",
        "--template",
        dest="template",
        type=str,
        required=True,
        help="Path to a template in PDB/MMCIF or CCP4/MRC format.",
    ),
    parser.add_argument(
        "--template_mask",
        dest="template_mask",
        type=str,
        required=False,
        help="Path to a mask for the template in CCP4/MRC format.",
    ),
    parser.add_argument(
        "-o",
        dest="output",
        type=str,
        required=False,
        default="output.pickle",
        help="Path to output pickle file.",
    )
    parser.add_argument(
        "-s",
        dest="score",
        type=str,
        default="CC",
        help="Template matching scoring function.",
        choices=MATCHING_EXHAUSTIVE_REGISTER.keys(),
    )
    parser.add_argument(
        "-n",
        dest="cores",
        required=False,
        type=int,
        default=4,
        help="Number of cores used for template matching.",
    )
    parser.add_argument(
        "-r",
        "--ram",
        dest="ram",
        required=False,
        type=int,
        default=None,
        help="Amount of RAM that can be used in bytes.",
    )
    parser.add_argument(
        "-a",
        dest="angular_sampling",
        type=check_positive,
        default=40.0,
        help="Angular sampling rate for template matching. "
        "A lower number yields more rotations.",
    )
    parser.add_argument(
        "-p",
        dest="peak_calling",
        action="store_true",
        default=False,
        help="When set perform peak calling instead of score aggregation.",
    )
    parser.add_argument(
        "--use_gpu",
        dest="use_gpu",
        action="store_true",
        default=False,
        help="Whether to perform computations on the GPU.",
    )
    parser.add_argument(
        "--gpu_indices",
        dest="gpu_indices",
        type=str,
        default=None,
        help="Comma-separated list of GPU indices to use. For example,"
        " 0,1 for the first and second GPU. Only used if --use_gpu is set."
        " If not provided but --use_gpu is set, CUDA_VISIBLE_DEVICES will"
        " be respected.",
    )
    parser.add_argument(
        "--no_edge_padding",
        dest="no_edge_padding",
        action="store_true",
        default=False,
        help="Whether to pad the edges of the target. This is useful, if the target"
        " has a well defined bounding box, e.g. a density map.",
    )
    parser.add_argument(
        "--no_fourier_padding",
        dest="no_fourier_padding",
        action="store_true",
        default=False,
        help="Whether input arrays should be zero-padded to the full convolution shape"
        " for numerical stability.",
    )
    parser.add_argument(
        "--scramble_phases",
        dest="scramble_phases",
        action="store_true",
        default=False,
        help="Whether to phase scramble the template for subsequent normalization.",
    )
    parser.add_argument(
        "--interpolation_order",
        dest="interpolation_order",
        required=False,
        type=int,
        default=3,
        help="Spline interpolation used during rotations. If less than zero"
        " no interpolation is performed.",
    )
    parser.add_argument(
        "--use_mixed_precision",
        dest="use_mixed_precision",
        action="store_true",
        default=False,
        help="Use float16 for real values operations where possible.",
    )
    parser.add_argument(
        "--use_memmap",
        dest="use_memmap",
        action="store_true",
        default=False,
        help="Use memmaps to offload large data objects to disk. This is"
        " particularly useful for large inputs when using --use_gpu..",
    )
    parser.add_argument(
        "--temp_directory",
        dest="temp_directory",
        default=None,
        help="Directory for temporary objects. Faster I/O typically improves runtime.",
    )
    parser.add_argument(
        "--gaussian_sigma",
        dest="gaussian_sigma",
        type=float,
        required=False,
        help="Sigma parameter for Gaussian filtering the template.",
    )

    parser.add_argument(
        "--bandpass_band",
        dest="bandpass_band",
        type=str,
        required=False,
        help="Comma separated start and stop frequency for bandpass filtering the"
        " template, e.g. 0.1, 0.5",
    )
    parser.add_argument(
        "--bandpass_smooth",
        dest="bandpass_smooth",
        type=float,
        required=False,
        default=None,
        help="Smooth parameter for the bandpass filter.",
    )

    parser.add_argument(
        "--tilt_range",
        dest="tilt_range",
        type=str,
        required=False,
        help="Comma separated start and stop stage tilt angle, e.g. '50,45'. Used"
        " to create a wedge mask to be applied to the template.",
    )
    parser.add_argument(
        "--tilt_step",
        dest="tilt_step",
        type=float,
        required=False,
        default=None,
        help="Step size between tilts, e.g. '5'. When set a more accurate"
        " wedge mask will be computed.",
    )
    parser.add_argument(
        "--wedge_smooth",
        dest="wedge_smooth",
        type=float,
        required=False,
        default=None,
        help="Gaussian sigma used to smooth the wedge mask.",
    )

    args = parser.parse_args()

    if args.interpolation_order < 0:
        args.interpolation_order = None

    if args.temp_directory is None:
        default = abspath(".")
        if os.environ.get("TMPDIR", None) is not None:
            default = os.environ.get("TMPDIR")
        args.temp_directory = default

    os.environ["TMPDIR"] = args.temp_directory

    args.pad_target_edges = not args.no_edge_padding
    args.pad_fourier = not args.no_fourier_padding

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
            # raise ValueError(
            #     "No GPU indices provided and CUDA_VISIBLE_DEVICES is not set."
            # )
            print(
                "No GPU indices provided and CUDA_VISIBLE_DEVICES is not set.",
                "Assuming device 0.",
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        args.gpu_indices = [
            int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        ]

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

    if not np.allclose(target.sampling_rate, template.sampling_rate):
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
    is_cropped = crop_data(
        data=target, data_mask=target_mask, cutoff=args.cutoff_target
    )
    print_block(
        name="Target",
        data={
            "Inital Shape": initial_shape,
            "Sampling Rate": tuple(np.round(target.sampling_rate, 2)),
            "Final Shape": target.shape,
        },
    )
    if is_cropped:
        args.target = generate_tempfile_name(suffix=".mrc")
        target.to_file(args.target)

        if target_mask:
            args.target_mask = generate_tempfile_name(suffix=".mrc")
            target_mask.to_file(args.target_mask)
            print_block(
                name="Target Mask",
                data={
                    "Inital Shape": initial_shape,
                    "Sampling Rate": tuple(np.round(target_mask.sampling_rate, 2)),
                    "Final Shape": target_mask.shape,
                },
            )

    initial_shape = template.shape
    _ = crop_data(data=template, data_mask=template_mask, cutoff=args.cutoff_template)
    template, translation = template.centered(0)
    print_block(
        name="Template",
        data={
            "Inital Shape": initial_shape,
            "Sampling Rate": tuple(np.round(template.sampling_rate, 2)),
            "Final Shape": template.shape,
        },
    )

    template_filter = {}
    if args.gaussian_sigma is not None:
        template.data = Preprocessor().gaussian_filter(
            sigma=args.gaussian_sigma, template=template.data
        )

    if args.bandpass_band is not None:
        bandpass_start, bandpass_stop = [
            float(x) for x in args.bandpass_band.split(",")
        ]
        if args.bandpass_smooth is None:
            args.bandpass_smooth = 0

        template_filter["bandpass_mask"] = {
            "minimum_frequency": bandpass_start,
            "maximum_frequency": bandpass_stop,
            "gaussian_sigma": args.bandpass_smooth,
        }

    if args.tilt_range is not None:
        args.wedge_smooth if args.wedge_smooth is not None else 0
        tilt_start, tilt_stop = [float(x) for x in args.tilt_range.split(",")]

        if args.tilt_step is not None:
            tilt_angles = np.arange(
                -tilt_start, tilt_stop + args.tilt_step, args.tilt_step
            )
            angles = np.zeros((template.data.ndim, tilt_angles.size))
            angles[2, :] = tilt_angles
            template_filter["wedge_mask"] = {
                "tilt_angles": angles,
                "sigma": args.wedge_smooth,
            }
        else:
            template_filter["continuous_wedge_mask"] = {
                "start_tilt": tilt_start,
                "stop_tilt": tilt_stop,
                "tilt_axis": 1,
                "infinite_plane": True,
                "sigma": args.wedge_smooth,
            }

    if template_mask is None:
        enclosing_box = template.minimum_enclosing_box(0, use_geometric_center=False)
        template_mask = template.empty
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
    )

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
        template.data = scramble_phases(template.data, noise_proportion=1.0)

    available_memory, ram_scaling = backend.get_available_memory(), 1.0
    if args.use_gpu:
        args.cores, ram_scaling = len(args.gpu_indices), 0.85
        has_torch = importlib.util.find_spec("torch") is not None
        has_cupy = importlib.util.find_spec("cupy") is not None

        if not has_torch and not has_cupy:
            raise ValueError(
                "Found neither CuPy nor PyTorch installation. You need to install"
                " either to enable GPU support."
            )

        if args.peak_calling:
            preferred_backend = "pytorch"
            if not has_torch:
                preferred_backend = "cupy"
            backend.change_backend(backend_name=preferred_backend, device="cuda")
        else:
            preferred_backend = "cupy"
            if not has_cupy:
                preferred_backend = "pytorch"
            backend.change_backend(backend_name=preferred_backend, device="cuda")
            if args.use_mixed_precision and preferred_backend == "pytorch":
                raise NotImplementedError(
                    "pytorch backend does not yet support mixed precision."
                    " Consider installing CuPy to enable this feature."
                )
            elif args.use_mixed_precision:
                backend.change_backend(
                    backend_name="cupy",
                    default_dtype=backend._array_backend.float16,
                    complex_dtype=backend._array_backend.complex64,
                    default_dtype_int=backend._array_backend.int16,
                )
        available_memory = backend.get_available_memory() * args.cores

    if args.ram is None:
        args.ram = int(ram_scaling * available_memory)

    target_padding = np.zeros_like(template.shape)
    if args.pad_target_edges:
        target_padding = template.shape

    template_box = template.shape
    if not args.pad_fourier:
        template_box = np.ones(len(template_box), dtype=int)

    callback_class = MaxScoreOverRotations
    if args.peak_calling:
        callback_class = PeakCallerMaximumFilter

    splits, schedule = compute_parallelization_schedule(
        shape1=target.shape,
        shape2=template_box,
        shape1_padding=target_padding,
        max_cores=args.cores,
        max_ram=args.ram,
        split_only_outer=args.use_gpu,
        matching_method=args.score,
        analyzer_method=callback_class.__name__,
        backend=backend._backend_name,
        float_nbytes=backend.datatype_bytes(backend._default_dtype),
        complex_nbytes=backend.datatype_bytes(backend._complex_dtype),
        integer_nbytes=backend.datatype_bytes(backend._default_dtype_int),
    )

    if splits is None:
        print(
            "Found no suitable parallelization schedule. Consider increasing"
            " available RAM or decreasing number of cores."
        )
        exit(-1)

    analyzer_args = {
        "score_threshold": 0.2,
        "number_of_peaks": 1000,
        "convolution_mode": "valid",
        "use_memmap": args.use_memmap,
    }

    matching_setup, matching_score = MATCHING_EXHAUSTIVE_REGISTER[args.score]
    matching_data = MatchingData(target=target, template=template.data)
    matching_data.rotations = get_rotation_matrices(
        angular_sampling=args.angular_sampling, dim=target.data.ndim
    )

    matching_data.template_filter = template_filter
    if target_mask is not None:
        matching_data.target_mask = target_mask
    if template_mask is not None:
        matching_data.template_mask = template_mask.data

    n_splits = np.prod(list(splits.values()))
    target_split = ", ".join(
        [":".join([str(x) for x in axis]) for axis in splits.items()]
    )
    gpus_used = 0 if args.gpu_indices is None else len(args.gpu_indices)
    options = {
        "CPU Cores": args.cores,
        "Run on GPU": f"{args.use_gpu} [N={gpus_used}]",
        "Use Mixed Precision": args.use_mixed_precision,
        "Assigned Memory [MB]": f"{args.ram // 1e6} [out of {available_memory//1e6}]",
        "Temporary Directory": args.temp_directory,
        "Extend Fourier Grid": not args.no_fourier_padding,
        "Extend Target Edges": not args.no_edge_padding,
        "Interpolation Order": args.interpolation_order,
        "Score": f"{args.score}",
        "Setup Function": f"{get_func_fullname(matching_setup)}",
        "Scoring Function": f"{get_func_fullname(matching_score)}",
        "Angular Sampling": f"{args.angular_sampling}"
        f" [{matching_data.rotations.shape[0]} rotations]",
        "Scramble Template": args.scramble_phases,
        "Target Splits": f"{target_split} [N={n_splits}]",
    }

    print_block(
        name="Template Matching Options",
        data=options,
        label_width=max(len(key) for key in options.keys()) + 2,
    )

    options = {"Analyzer": callback_class, **analyzer_args}
    print_block(
        name="Score Analysis Options",
        data=options,
        label_width=max(len(key) for key in options.keys()) + 2,
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
        pad_target_edges=args.pad_target_edges,
        pad_fourier=args.pad_fourier,
        interpolation_order=args.interpolation_order,
    )

    candidates = list(candidates) if candidates is not None else []
    if callback_class == MaxScoreOverRotations:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            candidates[3] = {
                x: euler_from_rotationmatrix(
                    np.frombuffer(i, dtype=matching_data.rotations.dtype).reshape(
                        candidates[0].ndim, candidates[0].ndim
                    )
                )
                for i, x in candidates[3].items()
            }

    candidates.append((target.origin, template.origin, target.sampling_rate, args))
    write_pickle(data=candidates, filename=args.output)

    runtime = time() - start
    print(f"\nRuntime real: {runtime:.3f}s user: {(runtime * args.cores):.3f}s.")


if __name__ == "__main__":
    main()
