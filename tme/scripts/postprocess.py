#!python3
"""CLI to analyse the output of match_template.py.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import os
import re
import atexit
import argparse
import tempfile
from sys import exit
from os import getcwd
from pathlib import Path
from typing import Tuple, List
from os.path import join, splitext, basename

import numpy as np
from numpy.typing import NDArray

from tme.utils import cli, serialization, logging
from tme import Density, Structure, Orientations
from tme.matching_optimization import create_score_object, optimize_match
from tme.rotations import euler_to_rotationmatrix, euler_from_rotationmatrix
from tme.analyzer import (
    PeakCallerSort,
    PeakCallerMaximumFilter,
    PeakCallerFast,
    PeakCallerRecursiveMasking,
    PeakCallerScipy,
)

logger = logging.get_logger("postprocess")


ROTATION_DTYPE = np.uint16
NO_ROTATION = np.iinfo(ROTATION_DTYPE).max  # 65535, reserved sentinel
MAX_ROTATIONS = NO_ROTATION - 1  # 65534 distinct rotation indices allowed


_SCRATCH_FILES: List[str] = []


def _cleanup_scratch_files():
    """Remove any leftover memmap scratch files. Idempotent."""
    while _SCRATCH_FILES:
        path = _SCRATCH_FILES.pop()
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


atexit.register(_cleanup_scratch_files)


PEAK_CALLERS = {
    "PeakCallerSort": PeakCallerSort,
    "PeakCallerMaximumFilter": PeakCallerMaximumFilter,
    "PeakCallerFast": PeakCallerFast,
    "PeakCallerRecursiveMasking": PeakCallerRecursiveMasking,
    "PeakCallerScipy": PeakCallerScipy,
}


def add_postprocess_arguments(parser, batch_mode=False):
    input_group = parser.add_argument_group("Input")
    output_group = parser.add_argument_group("Output")
    peak_group = parser.add_argument_group("Peak Calling")
    additional_group = parser.add_argument_group("Additional Parameters")

    if not batch_mode:
        input_group.add_argument(
            "--input-file",
            "--input-files",
            required=True,
            nargs="+",
            help="Path to one or multiple runs of match_template.py.",
        )
        input_group.add_argument(
            "--background-file",
            "--background-files",
            nargs="+",
            default=[],
            help="Path to one or multiple runs of match_template.py for normalization. "
            "For instance from a phase-scrambled template "
            "(pytme preprocess --scramble-phases) or a different template.",
        )
        input_group.add_argument(
            "--target-mask",
            type=str,
            help="Path to an optional mask applied to template matching scores.",
        )
        output_group.add_argument(
            "--output-prefix",
            help="Output prefix. Defaults to basename of first input. Extension is "
            "added with respect to chosen output format.",
        )

    input_group.add_argument(
        "--template-mask",
        type=str,
        help="Override the template mask for peak calling with PeakCallerRecursiveMasking. "
        "This allows you to use a different (typically tighter) mask for peak calling than "
        "the one used during template matching.",
    )
    input_group.add_argument(
        "--orientations",
        type=str,
        help="Path to file generated using output_format orientations. Can be filtered "
        "to exclude false-positive peaks. If this file is provided, peak calling "
        "is skipped and corresponding parameters ignored.",
    )

    output_group.add_argument(
        "--output-format",
        choices=[
            "relion4",
            "relion5",
            "orientations",
            "alignment",
            "extraction",
            "pickle",
        ],
        default="relion4",
        help="Available output formats: "
        "relion4 (RELION 4 star format), "
        "relion5 (RELION 5 star format), "
        "orientations (translation, rotation, and score), "
        "alignment (aligned template to target based on orientations), "
        "extraction (extract regions around peaks from targets, i.e. subtomograms), "
        "pickle (results of applying mask and background correction for inspection).",
    )

    peak_group.add_argument(
        "--peak-caller",
        choices=list(PEAK_CALLERS.keys()),
        default="PeakCallerMaximumFilter",
        help="Peak caller for local maxima identification.",
    )
    peak_group.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score from which peaks will be considered. When also using "
        "--n-false-positive, the larger score cutoff is used.",
    )
    peak_group.add_argument(
        "--max-score",
        type=float,
        help="Maximum score until which peaks will be considered.",
    )
    peak_group.add_argument(
        "--min-distance",
        type=cli.check_bounded_dtype(float, 0),
        default=5,
        help="Minimum distance between peaks.",
    )
    peak_group.add_argument(
        "--min-boundary-distance",
        type=cli.check_bounded_dtype(float, 0),
        default=0,
        help="Minimum distance of peaks to target edges.",
    )
    peak_group.add_argument(
        "--mask-edges",
        action="store_true",
        help="Whether candidates should not be identified from scores that were "
        "computed from padded densities. Superseded by min_boundary_distance.",
    )
    peak_group.add_argument(
        "--num-peaks",
        type=cli.check_bounded_dtype(int, 1),
        default=1000,
        help="Upper limit of peaks to call, subject to filtering parameters. Default 1000. "
        "If minimum_score is provided all peaks scoring higher will be reported.",
    )
    peak_group.add_argument(
        "--peak-oversampling",
        type=cli.check_bounded_dtype(float, 0),
        help="1 / factor equals voxel precision, e.g. 2 detects half voxel "
        "translations. Useful for matching structures to electron density maps.",
    )

    additional_group.add_argument(
        "--extraction-box-size",
        type=cli.check_bounded_dtype(int, 1),
        help="Box size of extracted subtomograms, defaults to the centered template.",
    )
    additional_group.add_argument(
        "--invert-target-contrast",
        action="store_true",
        help="Whether to invert the target contrast.",
    )
    additional_group.add_argument(
        "--n-false-positives",
        type=cli.check_bounded_dtype(float, 0),
        help="Number of accepted false-positives picks to determine minimum score.",
    )
    additional_group.add_argument(
        "--local-optimization",
        action="store_true",
        help="[Experimental] Perform local optimization of candidates. Useful when the "
        "number of identified candidats is small (< 10).",
    )
    additional_group.add_argument(
        "--snr",
        action="store_true",
        help="Normalize scores of individual inputs to SNR-like values (z-scores).",
    )
    additional_group.add_argument(
        "--warp-compatible-names",
        action="store_true",
        help="Attempt to create warp compatible _rlnMicrographName for output formats "
        "'relion4' and 'relion5'.",
    )
    additional_group.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip statistics computation during postprocessing (faster for parameter tuning).",
    )
    additional_group.add_argument(
        "--use-memmap",
        action="store_true",
        help="Back the merge scratch buffers (scores, rotations, entities) with "
        "temporary memmaps under $TMPDIR. Useful for very large volumes on "
        "RAM-constrained nodes. Files are removed at exit.",
    )

    meta = cli.ArgumentMetadata()
    groups = [input_group, output_group, peak_group, additional_group]
    for group in groups:
        meta.update(group)
    return meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze template matching outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_postprocess_arguments(parser, batch_mode=False)
    args = parser.parse_args()

    if args.output_prefix is None:
        args.output_prefix = cli.strip_result_extensions(args.input_file[0])

    if args.orientations is not None:
        args.orientations = Orientations.from_file(filename=args.orientations)
    return args


def load_template(
    filepath: str,
    sampling_rate: NDArray,
    centering: bool = True,
):
    try:
        template = Density.from_file(filepath)
        template_is_density = True
    except Exception:
        template = Density.from_structure(filepath, sampling_rate=sampling_rate)
        template_is_density = False

    if centering:
        template = template.centered(0)
    center = np.divide(np.subtract(template.shape, 1), 2)
    return template, center, template_is_density


def load_matching_output(path: str, compute_snr: bool = False) -> List:
    data = serialization.deserialize(path)

    if compute_snr:
        data = [x[:] if hasattr(x, "shape") else x for x in data]
        mean = data[0].mean()
        std = data[0].std()
        std = np.where(std <= 1e-6, 1, std)
        data[0] -= mean
        data[0] /= std

        if len(data) == 6:
            data[4] = data[4] / std**2
    return data


def simple_stats(arr, decimals=3):
    if arr is None:
        return None

    return {
        "mean": round(float(arr.mean()), decimals),
        "std": round(float(arr.std()), decimals),
        "max": round(float(arr.max()), decimals),
    }


def _alloc_buffer(shape, dtype, fill_value, use_memmap=False):
    if use_memmap:
        # tempfile.mkstemp honors $TMPDIR when dir is None.
        fd, path = tempfile.mkstemp(suffix=".mm")
        os.close(fd)
        _SCRATCH_FILES.append(path)
        arr = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
        if fill_value != 0:
            arr[:] = fill_value
        return arr
    return np.full(shape, fill_value, dtype=dtype)


def _validate_shape(path, scores, expected_shape):
    if scores.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch in {path}: got {scores.shape}, expected {expected_shape}. "
            "All foregrounds, backgrounds and the target mask must share the same shape."
        )


def _merge_one(
    path,
    entity_index,
    scores_out,
    rotations_out,
    entities,
    rotation_mapping,
    local_to_global,
    mask_buffer,
    compute_snr,
    preloaded_data=None,
):
    """Load `path` (or use `preloaded_data`) and merge it into the running buffers.

    `rotation_mapping` maps global rotation index to rotation matrix (this
    becomes the merged `data[3]`). `local_to_global` maps a local key
    (as stored in the file's own rotation map) to the assigned global index;
    it is the bookkeeping used to translate this file's rotation indices.

    Returns the deserialized data tuple and the variance from `data[4]` when
    `len(data) == 6`, else None.
    """
    data = (
        preloaded_data
        if preloaded_data is not None
        else load_matching_output(path, compute_snr=compute_snr)
    )
    scores = data[0][:] if hasattr(data[0], "shape") else data[0]
    rotations = data[2][:] if hasattr(data[2], "shape") else data[2]
    local_rotation_map = data[3]

    _validate_shape(path, scores, scores_out.shape)

    for key, matrix in local_rotation_map.items():
        if key not in local_to_global:
            new_index = len(rotation_mapping)
            local_to_global[key] = new_index
            rotation_mapping[new_index] = matrix

    if len(rotation_mapping) > MAX_ROTATIONS:
        raise ValueError(
            f"Too many distinct rotations after merging {path}: "
            f"{len(rotation_mapping)} exceeds the {MAX_ROTATIONS} slots available in "
            f"{ROTATION_DTYPE.__name__}. Reduce the rotation sampling or split the run."
        )

    # +2 reserves a trailing NO_ROTATION slot so that -1 (no-rotation) entries
    # in `rotations` wrap to it via negative indexing.
    max_local = max(local_rotation_map.keys(), default=-1)
    lookup_table = np.full(max_local + 2, NO_ROTATION, dtype=ROTATION_DTYPE)
    for key in local_rotation_map:
        lookup_table[key] = local_to_global[key]

    np.greater(scores, scores_out, out=mask_buffer)
    np.copyto(scores_out, scores, where=mask_buffer)
    np.copyto(rotations_out, lookup_table[rotations], where=mask_buffer)
    if entities is not None:
        np.copyto(entities, entity_index, where=mask_buffer)

    variance = data[4] if len(data) == 6 else None
    return data, variance


def normalize_input(
    foregrounds: Tuple[str],
    backgrounds: Tuple[str],
    compute_snr: bool = False,
    compute_stats: bool = True,
    use_memmap: bool = False,
) -> Tuple:
    if not foregrounds:
        exit("No foreground inputs given.")

    first_data = load_matching_output(foregrounds[0], compute_snr=compute_snr)
    first_scores = (
        first_data[0][:] if hasattr(first_data[0], "shape") else first_data[0]
    )
    shape = first_scores.shape

    scores_out = _alloc_buffer(shape, np.float32, 0, use_memmap)
    rotations_out = _alloc_buffer(shape, ROTATION_DTYPE, NO_ROTATION, use_memmap)
    entities = _alloc_buffer(shape, np.int8, -1, use_memmap) if compute_stats else None
    mask_buffer = np.empty(shape, dtype=bool)

    rotation_mapping = {}
    local_to_global = {}
    max_var = None
    last_data = None

    for entity_index, foreground in enumerate(foregrounds):
        last_data, variance = _merge_one(
            path=foreground,
            entity_index=entity_index,
            scores_out=scores_out,
            rotations_out=rotations_out,
            entities=entities,
            rotation_mapping=rotation_mapping,
            local_to_global=local_to_global,
            mask_buffer=mask_buffer,
            compute_snr=compute_snr,
            preloaded_data=first_data if entity_index == 0 else None,
        )
        if variance is not None:
            variance_scalar = float(np.asarray(variance).reshape(()))
            max_var = (
                variance_scalar if max_var is None else max(max_var, variance_scalar)
            )

    data = list(last_data)
    data[0] = scores_out
    data[2] = rotations_out
    data[3] = rotation_mapping
    if len(data) == 6 and max_var is not None:
        data[4] = np.float32(max_var)

    if compute_stats:
        fg = simple_stats(data[0])
        logger.info(f"> Foreground {', '.join(f'{k} {v}' for k, v in fg.items())}.")

        if not backgrounds:
            logger.info("Score statistics per entity")
            for i in range(len(foregrounds)):
                mask = entities == i
                avg = "No occurences"
                if mask.sum() != 0:
                    fg = simple_stats(data[0][mask])
                    avg = ", ".join(f"{k} {v}" for k, v in fg.items())
                logger.info(f"> Entity {i}: {avg}.")
            return data, entities

    if not backgrounds:
        return data, entities

    logger.info("Computing and applying background correction.")
    scores_norm = _alloc_buffer(shape, np.float32, 0, use_memmap)
    bg_mask = np.empty(shape, dtype=bool)

    for background in backgrounds:
        bg_data = load_matching_output(background, compute_snr=compute_snr)
        bg_scores = bg_data[0][:] if hasattr(bg_data[0], "shape") else bg_data[0]
        _validate_shape(background, bg_scores, shape)

        np.greater(bg_scores, scores_norm, out=bg_mask)
        np.copyto(scores_norm, bg_scores, where=bg_mask)

    np.subtract(data[0], scores_norm, out=data[0])
    np.add(data[0], scores_norm.mean(), out=data[0])

    if compute_stats:
        fg = simple_stats(data[0])
        bg = simple_stats(scores_norm)
        logger.info(f"> Background {', '.join(f'{k} {v}' for k, v in bg.items())}.")
        logger.info(f"> Normalized {', '.join(f'{k} {v}' for k, v in fg.items())}.")

        logger.info("\nScore statistics per entity")
        for i in range(len(foregrounds)):
            mask = entities == i
            avg = "No occurences"
            if mask.sum() != 0:
                fg = simple_stats(data[0][mask])
                avg = ", ".join(f"{k} {v}" for k, v in fg.items())
            logger.info(f"> Entity {i}: {avg}.")

    return data, entities


def main():
    args = parse_args()
    logging.setup_logging()

    cli.print_entry(logger)

    cli_kwargs = {
        key: value
        for key, value in sorted(vars(args).items())
        if value is not None and key not in ("input_file", "background_file")
    }
    cli.print_block(
        name="Parameters",
        data={cli.sanitize_name(k): v for k, v in cli_kwargs.items()},
        label_width=25,
        logger=logger,
    )

    cli.print_block(
        name=cli.sanitize_name("Foreground entities"),
        data={i: k for i, k in enumerate(args.input_file)},
        label_width=25,
        logger=logger,
    )

    if len(args.background_file):
        cli.print_block(
            name=cli.sanitize_name("Background entities"),
            data={i: k for i, k in enumerate(args.background_file)},
            label_width=25,
            logger=logger,
        )
    logger.info("\n" + "-" * 80 + "\n")

    data, entities = normalize_input(
        args.input_file,
        args.background_file,
        compute_snr=args.snr,
        compute_stats=not args.no_stats,
        use_memmap=args.use_memmap,
    )

    if args.output_format == "pickle":
        serialization.serialize(data, f"{args.output_prefix}.pickle")
        exit(0)

    if args.target_mask:
        target_mask = Density.from_file(args.target_mask, use_memmap=True)
        if target_mask.data.shape != data[0].shape:
            raise ValueError(
                f"Shape mismatch in {args.target_mask}: got {target_mask.data.shape}, "
                f"expected {data[0].shape}. The target mask must share the foreground shape."
            )
        data[0] = np.multiply(data[0], target_mask.data, out=data[0])
    target_origin, _, sampling_rate, cli_args = data[-1]

    # Backwards compatibility with pre v0.3.0b
    if hasattr(cli_args, "no_centering"):
        cli_args.centering = not cli_args.no_centering

    if args.template_mask is not None:
        cli_args.template_mask = args.template_mask

    template, *_ = load_template(
        filepath=cli_args.template,
        sampling_rate=sampling_rate,
        centering=cli_args.centering,
    )

    template_mask = template.empty
    template_mask.data[:] = 1
    if cli_args.template_mask is not None:
        template_mask = Density.from_file(cli_args.template_mask)

    if cli_args.centering:
        template_mask.pad(template.shape, center=True)

    if args.mask_edges and args.min_boundary_distance == 0:
        max_shape = np.max(template.shape)
        args.min_boundary_distance = np.ceil(np.divide(max_shape, 2))

    # Do the actual peak calling
    orientations = args.orientations
    if orientations is None:
        translations, rotations, scores, details = [], [], [], []

        var = None
        scores, _, rotation_array, rotation_mapping, *_ = data
        if len(data) == 6:
            scores, _, rotation_array, rotation_mapping, var, *_ = data

        cropped_shape = np.subtract(
            scores.shape, np.multiply(args.min_boundary_distance, 2)
        ).astype(int)

        if args.min_boundary_distance > 0:
            from tme.matching_utils import center_slice

            _scores = np.zeros_like(scores)
            subset = center_slice(scores.shape, cropped_shape)
            _scores[subset] = scores[subset]
            scores = _scores

        if args.n_false_positives is not None:
            from tme.matching_utils import minimum_score_from_fp

            cropped_slice = tuple(
                slice(
                    int(args.min_boundary_distance), int(x - args.min_boundary_distance)
                )
                for x in scores.shape
            )
            n_correlations = np.size(scores[cropped_slice]) * len(rotation_mapping)
            if var is not None:
                std = float(np.sqrt(var).reshape(()))
            else:
                std = float(np.std(scores[cropped_slice]))

            minimum_score = minimum_score_from_fp(
                std, n_correlations, args.n_false_positives
            )
            logger.info(f"Determined cutoff --min-score {minimum_score}.")
            if args.min_score is None:
                args.min_score = minimum_score
            args.min_score = max(args.min_score, minimum_score)

        projection_dims = None
        batch_dims = getattr(cli_args, "batch_dims", None)
        if getattr(cli_args, "match_projection", False):
            projection_dims = batch_dims

        peak_caller_kwargs = {
            "shape": scores.shape,
            "num_peaks": args.num_peaks,
            "min_distance": args.min_distance,
            "min_boundary_distance": args.min_boundary_distance,
            "min_score": args.min_score,
            "max_score": args.max_score,
            "batch_dims": batch_dims,
            "projection_dims": projection_dims,
        }

        peak_caller = PEAK_CALLERS[args.peak_caller](**peak_caller_kwargs)
        state = peak_caller.init_state()
        state = peak_caller(
            state,
            scores,
            mask=template_mask.data,
            rotation_mapping=rotation_mapping,
            rotations=rotation_array,
            rotation_matrix=np.eye(template_mask.data.ndim),
        )
        candidates = peak_caller.merge(
            results=[peak_caller.result(state)], **peak_caller_kwargs
        )
        if len(candidates) == 0:
            candidates = [[], [], [], []]
            logger.info("Found no peaks, consider changing peak calling parameters.")
            exit(-1)

        for translation, _, score, detail in zip(*candidates):
            index = rotation_array[tuple(translation)]
            rotation = rotation_mapping.get(index, np.eye(template.data.ndim))
            rotations.append(euler_from_rotationmatrix(rotation, seq="ZYZ"))

        if len(rotations):
            rotations = np.vstack(rotations).astype(float)
        translations, scores, details = candidates[0], candidates[2], candidates[3]

        if entities is not None:
            details = entities[tuple(translations.T)]

        sampling = np.asarray(sampling_rate, dtype=float).ravel()
        pixel_size = float(np.mean(sampling)) if sampling.size else 1.0
        orientations = Orientations(
            translations=translations,
            rotations=rotations,
            metadata={
                "_pytmeScore": np.asarray(scores, dtype=np.float32),
                "_rlnClassNumber": np.asarray(details),
            },
            optics={
                "_rlnOpticsGroup": 1,
                "_rlnOpticsGroupName": "opticsGroup1",
                "_rlnImagePixelSize": pixel_size,
            },
        )

    if args.min_score is not None and len(orientations.scores):
        keep = orientations.scores >= args.min_score
        orientations = orientations[keep]

    if args.max_score is not None and len(orientations.scores):
        keep = orientations.scores <= args.max_score
        orientations = orientations[keep]

    if args.peak_oversampling:
        if data[0].ndim != data[2].ndim:
            exit(
                "Input pickle does not contain template matching scores."
                " Cannot oversample peaks."
            )
        peak_caller = PEAK_CALLERS[args.peak_caller](shape=scores.shape)
        orientations.translations = peak_caller.oversample_peaks(
            scores=data[0],
            peak_positions=orientations.translations,
            oversampling_factor=args.peak_oversampling,
        )

    if args.local_optimization:
        target = Density.from_file(cli_args.target, use_memmap=True)
        for index, (translation, angles, *_) in enumerate(orientations):
            score_object = create_score_object(
                score="FLC",
                target=target.data.copy(),
                template=template.data.copy(),
                template_mask=template_mask.data.copy(),
            )

            center = np.divide(template.shape, 2)
            init_translation = np.subtract(translation, center)
            bounds_translation = tuple((x - 5, x + 5) for x in init_translation)

            translation, rotation_matrix, score = optimize_match(
                score_object=score_object,
                optimization_method="basinhopping",
                bounds_translation=bounds_translation,
                maxiter=3,
                x0=[*init_translation, *angles],
            )
            orientations.translations[index] = np.add(translation, center)
            orientations.rotations[index] = euler_from_rotationmatrix(
                rotation_matrix, seq="ZYZ"
            )
            orientations.scores[index] = score * -1

    if args.output_format in ("orientations", "relion4", "relion5"):
        file_format, extension = "tsv", "tsv"

        version = None
        if args.output_format in ("relion4", "relion5"):
            version = "# version 40001"
            file_format, extension = "star", "star"

        if args.output_format == "relion5":
            version = "# version 50001"
            target = Density.from_file(cli_args.target, use_memmap=True)
            orientations.translations = np.subtract(
                orientations.translations, np.divide(target.shape, 2).astype(int)
            )
            orientations.translations = np.multiply(
                orientations.translations, target.sampling_rate
            )

        source_path = basename(cli_args.target)
        if args.warp_compatible_names:
            source_path = Path(source_path).stem
            source_path = re.sub(
                r"_\d+(\.\d+)?(Apx|bin\d*|dose_filt)$", "", source_path
            )
            source_path += ".tomostar"

        orientations.to_file(
            filename=f"{args.output_prefix}.{extension}",
            file_format=file_format,
            source_path=source_path,
            version=version,
        )
        exit(0)

    target = Density.from_file(cli_args.target)
    if args.invert_target_contrast:
        target.data = target.data * -1

    if args.output_format in ("extraction"):
        if not np.all(np.divide(target.shape, template.shape) > 2):
            logger.info(
                "Target might be too small relative to template to extract"
                " meaningful particles."
                f" Target : {target.shape}, Template : {template.shape}."
            )

        extraction_shape = template.shape
        if args.extraction_box_size is not None:
            extraction_shape = np.repeat(
                args.extraction_box_size, len(extraction_shape)
            )

        orientations, cand_slices, obs_slices = orientations.get_extraction_slices(
            target_shape=target.shape,
            extraction_shape=extraction_shape,
            drop_out_of_box=True,
            return_orientations=True,
        )

        working_directory = getcwd()

        observations = np.zeros((len(cand_slices), *extraction_shape))
        slices = zip(cand_slices, obs_slices)
        for idx, (cand_slice, obs_slice) in enumerate(slices):
            observations[idx][:] = np.mean(target.data[obs_slice])
            observations[idx][cand_slice] = target.data[obs_slice]

        for index in range(observations.shape[0]):
            cand_start = [x.start for x in cand_slices[index]]
            out_density = Density(
                data=observations[index],
                sampling_rate=sampling_rate,
                origin=np.multiply(cand_start, sampling_rate),
            )
            out_density.to_file(
                join(working_directory, f"{args.output_prefix}_{index}.mrc")
            )

        exit(0)

    template, center, template_is_density, *_ = load_template(
        filepath=cli_args.template,
        sampling_rate=sampling_rate,
        centering=cli_args.centering,
    )

    _, ext = splitext(cli_args.template)
    for index, (translation, angles, *_) in enumerate(orientations):
        rotation = euler_to_rotationmatrix(angles, seq="ZYZ")
        if template_is_density:
            transformed_template = template.rigid_transform(
                rotation_matrix=rotation, use_geometric_center=True
            )

            # Just adapting the coordinate system not the in-box position
            shift = np.multiply(np.subtract(translation, center), sampling_rate)
            transformed_template.origin = np.add(target_origin, shift)
        else:
            template = Structure.from_file(cli_args.template)
            shift = np.add(np.multiply(translation, sampling_rate), target_origin)
            translation = np.subtract(shift, template.center_of_mass())

            # Since we move the template's center of mass to the geometric center
            # during matching and analysis we use the center of mass
            # directly for rotating structures into the correct orientation
            transformed_template = template.rigid_transform(
                translation=translation,
                rotation_matrix=rotation,
                center="geometric",
            )

        transformed_template.to_file(f"{args.output_prefix}_{index}{ext}")


if __name__ == "__main__":
    main()
