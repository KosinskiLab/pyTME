#!python3
"""CLI to simplify analysing the output of match_template.py.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import argparse
from sys import exit
from os import getcwd
from typing import Tuple, List
from os.path import join, splitext, basename

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfcinv

from tme import Density, Structure, Orientations
from tme.cli import sanitize_name, print_block, print_entry
from tme.matching_utils import load_pickle, centered_mask, write_pickle
from tme.matching_optimization import create_score_object, optimize_match
from tme.rotations import euler_to_rotationmatrix, euler_from_rotationmatrix
from tme.analyzer import (
    PeakCallerSort,
    PeakCallerMaximumFilter,
    PeakCallerFast,
    PeakCallerRecursiveMasking,
    PeakCallerScipy,
)


PEAK_CALLERS = {
    "PeakCallerSort": PeakCallerSort,
    "PeakCallerMaximumFilter": PeakCallerMaximumFilter,
    "PeakCallerFast": PeakCallerFast,
    "PeakCallerRecursiveMasking": PeakCallerRecursiveMasking,
    "PeakCallerScipy": PeakCallerScipy,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze template matching outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_argument_group("Input")
    output_group = parser.add_argument_group("Output")
    peak_group = parser.add_argument_group("Peak Calling")
    additional_group = parser.add_argument_group("Additional Parameters")

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
        required=False,
        nargs="+",
        default=[],
        help="Path to one or multiple runs of match_template.py for normalization. "
        "For instance from --scramble_phases or a different template.",
    )
    input_group.add_argument(
        "--target-mask",
        required=False,
        type=str,
        help="Path to an optional mask applied to template matching scores.",
    )
    input_group.add_argument(
        "--orientations",
        required=False,
        type=str,
        help="Path to file generated using output_format orientations. Can be filtered "
        "to exclude false-positive peaks. If this file is provided, peak calling "
        "is skipped and corresponding parameters ignored.",
    )

    output_group.add_argument(
        "--output-prefix",
        required=False,
        default=None,
        help="Output prefix. Defaults to basename of first input. Extension is "
        "added with respect to chosen output format.",
    )
    output_group.add_argument(
        "--output-format",
        choices=[
            "orientations",
            "relion4",
            "relion5",
            "alignment",
            "extraction",
            "average",
            "pickle",
        ],
        default="orientations",
        help="Available output formats: "
        "orientations (translation, rotation, and score), "
        "relion4 (RELION 4 star format), "
        "relion5 (RELION 5 star format), "
        "alignment (aligned template to target based on orientations), "
        "extraction (extract regions around peaks from targets, i.e. subtomograms), "
        "average (extract matched regions from target and average them)."
        "pickle (results of applying mask and background correction for inspection).",
    )

    peak_group.add_argument(
        "--peak-caller",
        choices=list(PEAK_CALLERS.keys()),
        default="PeakCallerScipy",
        help="Peak caller for local maxima identification.",
    )
    peak_group.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score from which peaks will be considered.",
    )
    peak_group.add_argument(
        "--max-score",
        type=float,
        default=None,
        help="Maximum score until which peaks will be considered.",
    )
    peak_group.add_argument(
        "--min-distance",
        type=int,
        default=5,
        help="Minimum distance between peaks.",
    )
    peak_group.add_argument(
        "--min-boundary-distance",
        type=int,
        default=0,
        help="Minimum distance of peaks to target edges.",
    )
    peak_group.add_argument(
        "--mask-edges",
        action="store_true",
        default=False,
        help="Whether candidates should not be identified from scores that were "
        "computed from padded densities. Superseded by min_boundary_distance.",
    )
    peak_group.add_argument(
        "--num-peaks",
        type=int,
        default=1000,
        required=False,
        help="Upper limit of peaks to call, subject to filtering parameters. Default 1000. "
        "If minimum_score is provided all peaks scoring higher will be reported.",
    )
    peak_group.add_argument(
        "--peak-oversampling",
        type=int,
        default=1,
        help="1 / factor equals voxel precision, e.g. 2 detects half voxel "
        "translations. Useful for matching structures to electron density maps.",
    )

    additional_group.add_argument(
        "--extraction-box-size",
        type=int,
        default=None,
        help="Box size of extracted subtomograms, defaults to the centered template.",
    )
    additional_group.add_argument(
        "--mask-subtomograms",
        action="store_true",
        default=False,
        help="Whether to mask subtomograms using the template mask. The mask will be "
        "rotated according to determined angles.",
    )
    additional_group.add_argument(
        "--invert-target-contrast",
        action="store_true",
        default=False,
        help="Whether to invert the target contrast.",
    )
    additional_group.add_argument(
        "--n-false-positives",
        type=int,
        default=None,
        required=False,
        help="Number of accepted false-positives picks to determine minimum score.",
    )
    additional_group.add_argument(
        "--local-optimization",
        action="store_true",
        required=False,
        help="[Experimental] Perform local optimization of candidates. Useful when the "
        "number of identified candidats is small (< 10).",
    )

    args = parser.parse_args()

    if args.output_prefix is None:
        args.output_prefix = splitext(basename(args.input_file[0]))[0]

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
        center = np.divide(np.subtract(template.shape, 1), 2)
        template_is_density = True
    except Exception:
        template = Structure.from_file(filepath)
        center = template.center_of_mass()
        template = Density.from_structure(template, sampling_rate=sampling_rate)
        template_is_density = False

    translation = np.zeros_like(center)
    if centering and template_is_density:
        template, translation = template.centered(0)
        center = np.divide(np.subtract(template.shape, 1), 2)

    return template, center, translation, template_is_density


def load_matching_output(path: str) -> List:
    data = load_pickle(path)
    if data[0].ndim != data[2].ndim:
        data = _peaks_to_volume(data)
    return list(data)


def _peaks_to_volume(data):
    # Emulate the output of analyzer aggregators
    translations = data[0].astype(int)
    keep = (translations < 0).sum(axis=1) == 0

    translations = translations[keep]
    rotations = data[1][keep]

    unique_rotations, rotation_map = np.unique(rotations, axis=0, return_inverse=True)
    rotation_mapping = {
        i: unique_rotations[i] for i in range(unique_rotations.shape[0])
    }

    out_shape = np.max(translations, axis=0) + 1

    scores_out = np.full(out_shape, fill_value=0, dtype=np.float32)
    scores_out[tuple(translations.T)] = data[2][keep]

    rotations_out = np.full(out_shape, fill_value=-1, dtype=np.int32)
    rotations_out[tuple(translations.T)] = rotation_map

    offset = np.zeros((scores_out.ndim), dtype=int)
    return (scores_out, offset, rotations_out, rotation_mapping, data[-1])


def prepare_pickle_merge(paths):
    new_rotation_mapping, out_shape = {}, None
    for path in paths:
        data = load_matching_output(path)
        scores, _, rotations, rotation_mapping, *_ = data
        if np.allclose(scores.shape, 0):
            continue

        if out_shape is None:
            out_shape = scores.shape

        if out_shape is not None and not np.allclose(out_shape, scores.shape):
            print(
                f"\nScore spaces have different sizes {out_shape} and {scores.shape}. "
                "Assuming that both boxes are aligned at the origin, but please "
                "make sure this is intentional."
            )
        out_shape = np.maximum(out_shape, scores.shape)

        for key, value in rotation_mapping.items():
            if key not in new_rotation_mapping:
                new_rotation_mapping[key] = len(new_rotation_mapping)

    return new_rotation_mapping, out_shape


def simple_stats(arr, decimals=3):
    return {
        "mean": round(float(arr.mean()), decimals),
        "std": round(float(arr.std()), decimals),
        "max": round(float(arr.max()), decimals),
    }


def normalize_input(foregrounds: Tuple[str], backgrounds: Tuple[str]) -> Tuple:
    # Determine output array shape and create consistent rotation map
    new_rotation_mapping, out_shape = prepare_pickle_merge(foregrounds)

    if out_shape is None:
        exit("No valid score spaces found. Check messages aboves.")

    print("\nFinished conversion - Now aggregating over entities.")
    entities = np.full(out_shape, fill_value=-1, dtype=np.int32)
    scores_out = np.full(out_shape, fill_value=0, dtype=np.float32)
    rotations_out = np.full(out_shape, fill_value=-1, dtype=np.int32)

    # We reload to avoid potential memory bottlenecks
    for entity_index, foreground in enumerate(foregrounds):
        data = load_matching_output(foreground)
        scores, _, rotations, rotation_mapping, *_ = data

        # We could normalize to unit sdev, but that might lead to unexpected
        # results for flat background distributions
        scores -= scores.mean()
        indices = tuple(slice(0, x) for x in scores.shape)

        indices_update = scores > scores_out[indices]
        scores_out[indices][indices_update] = scores[indices_update]

        lookup_table = np.arange(len(rotation_mapping) + 1, dtype=rotations_out.dtype)

        # Maps rotation matrix to rotation index in rotations array
        for key, _ in rotation_mapping.items():
            lookup_table[key] = new_rotation_mapping[key]

        updated_rotations = rotations[indices_update].astype(int)
        if len(updated_rotations):
            rotations_out[indices][indices_update] = lookup_table[updated_rotations]

        entities[indices][indices_update] = entity_index

    data = list(data)
    data[0] = scores_out
    data[2] = rotations_out

    fg = simple_stats(data[0])
    print(f"> Foreground {', '.join(str(k) + ' ' + str(v) for k, v in fg.items())}.")

    if not len(backgrounds):
        print("\nScore statistics per entity")
        for i in range(len(foregrounds)):
            mask = entities == i
            avg = "No occurences"
            if mask.sum() != 0:
                fg = simple_stats(data[0][mask])
                avg = ", ".join(str(k) + " " + str(v) for k, v in fg.items())
            print(f"> Entity {i}: {avg}.")
        return data, entities

    print("\nComputing and applying background correction.")
    _, out_shape_norm = prepare_pickle_merge(backgrounds)

    if not np.allclose(out_shape, out_shape_norm):
        print(
            f"Foreground and background have different sizes {out_shape} and "
            f"{out_shape_norm}. Assuming that boxes are aligned at the origin and "
            "dropping scores beyond, but make sure this is intentional."
        )

    scores_norm = np.full(out_shape_norm, fill_value=0, dtype=np.float32)
    for background in backgrounds:
        data_norm = load_matching_output(background)

        scores = data_norm[0]
        scores -= scores.mean()
        indices = tuple(slice(0, x) for x in scores.shape)
        indices_update = scores > scores_norm[indices]
        scores_norm[indices][indices_update] = scores[indices_update]

    # Set translations to zero that do not have background distribution
    update = tuple(slice(0, int(x)) for x in np.minimum(out_shape, scores.shape))
    scores_out = np.full(out_shape, fill_value=0, dtype=np.float32)
    scores_out[update] = data[0][update] - scores_norm[update]
    scores_out[update] = np.divide(scores_out[update], 1 - scores_norm[update])
    scores_out = np.fmax(scores_out, 0, out=scores_out)
    data[0] = scores_out

    fg, bg = simple_stats(data[0]), simple_stats(scores_norm)
    print(f"> Background {', '.join(str(k) + ' ' + str(v) for k, v in bg.items())}.")
    print(f"> Normalized {', '.join(str(k) + ' ' + str(v) for k, v in fg.items())}.")

    print("\nScore statistics per entity")
    for i in range(len(foregrounds)):
        mask = entities == i
        avg = "No occurences"
        if mask.sum() != 0:
            fg = simple_stats(data[0][mask])
            avg = ", ".join(str(k) + " " + str(v) for k, v in fg.items())
        print(f"> Entity {i}: {avg}.")

    return data, entities


def main():
    args = parse_args()
    print_entry()

    cli_kwargs = {
        key: value
        for key, value in sorted(vars(args).items())
        if value is not None and key not in ("input_file", "background_file")
    }
    print_block(
        name="Parameters",
        data={sanitize_name(k): v for k, v in cli_kwargs.items()},
        label_width=25,
    )
    print("\n" + "-" * 80)

    print_block(
        name=sanitize_name("Foreground entities"),
        data={i: k for i, k in enumerate(args.input_file)},
        label_width=25,
    )

    if len(args.background_file):
        print_block(
            name=sanitize_name("Background entities"),
            data={i: k for i, k in enumerate(args.background_file)},
            label_width=25,
        )

    data, entities = normalize_input(args.input_file, args.background_file)

    if args.output_format == "pickle":
        write_pickle(data, f"{args.output_prefix}.pickle")
        exit(0)

    if args.target_mask:
        target_mask = Density.from_file(args.target_mask, use_memmap=True).data
        if target_mask.shape != data[0].shape:
            print(
                f"Shape of target mask and scores do not match {target_mask} "
                f"{data[0].shape}. Skipping mask application"
            )
        else:
            np.multiply(data[0], target_mask, out=data[0])

    target_origin, _, sampling_rate, cli_args = data[-1]

    # Backwards compatibility with pre v0.3.0b
    if hasattr(cli_args, "no_centering"):
        cli_args.centering = not cli_args.no_centering

    _, template_extension = splitext(cli_args.template)
    ret = load_template(
        filepath=cli_args.template,
        sampling_rate=sampling_rate,
        centering=cli_args.centering,
    )
    template, center_of_mass, translation, template_is_density = ret

    template_mask = template.empty
    template_mask.data[:] = 1
    if cli_args.template_mask is not None:
        template_mask = Density.from_file(cli_args.template_mask)
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

    if args.mask_edges and args.min_boundary_distance == 0:
        max_shape = np.max(template.shape)
        args.min_boundary_distance = np.ceil(np.divide(max_shape, 2))

    orientations = args.orientations
    if orientations is None:
        translations, rotations, scores, details = [], [], [], []

        # Data processed by normalize_input is guaranteed to have this shape
        scores, offset, rotation_array, rotation_mapping, meta = data

        cropped_shape = np.subtract(
            scores.shape, np.multiply(args.min_boundary_distance, 2)
        ).astype(int)

        if args.min_boundary_distance > 0:
            scores = centered_mask(scores, new_shape=cropped_shape)

        if args.n_false_positives is not None:
            # Rickgauer et al. 2017
            cropped_slice = tuple(
                slice(
                    int(args.min_boundary_distance),
                    int(x - args.min_boundary_distance),
                )
                for x in scores.shape
            )
            args.n_false_positives = max(args.n_false_positives, 1)
            n_correlations = np.size(scores[cropped_slice]) * len(rotation_mapping)
            minimum_score = np.multiply(
                erfcinv(2 * args.n_false_positives / n_correlations),
                np.sqrt(2) * np.std(scores[cropped_slice]),
            )
            print(f"Determined minimum score cutoff: {minimum_score}.")
            minimum_score = max(minimum_score, 0)
            args.min_score = minimum_score

        args.batch_dims = None
        if hasattr(cli_args, "target_batch"):
            args.batch_dims = cli_args.target_batch

        peak_caller_kwargs = {
            "shape": scores.shape,
            "num_peaks": args.num_peaks,
            "min_distance": args.min_distance,
            "min_boundary_distance": args.min_boundary_distance,
            "batch_dims": args.batch_dims,
            "minimum_score": args.min_score,
            "maximum_score": args.max_score,
        }

        peak_caller = PEAK_CALLERS[args.peak_caller](**peak_caller_kwargs)
        state = peak_caller.init_state()
        state = peak_caller(
            state,
            scores,
            rotation_matrix=np.eye(template.data.ndim),
            mask=template_mask.data,
            rotation_mapping=rotation_mapping,
            rotations=rotation_array,
        )
        candidates = peak_caller.merge(
            results=[peak_caller.result(state)], **peak_caller_kwargs
        )
        if len(candidates) == 0:
            candidates = [[], [], [], []]
            print("Found no peaks, consider changing peak calling parameters.")
            exit(-1)

        for translation, _, score, detail in zip(*candidates):
            rotation_index = rotation_array[tuple(translation)]
            rotation = rotation_mapping.get(
                rotation_index, np.zeros(template.data.ndim, int)
            )
            if rotation.ndim == 2:
                rotation = euler_from_rotationmatrix(rotation)
            rotations.append(rotation)

        if len(rotations):
            rotations = np.vstack(rotations).astype(float)
        translations, scores, details = candidates[0], candidates[2], candidates[3]

        if entities is not None:
            details = entities[tuple(translations.T)]

        orientations = Orientations(
            translations=translations,
            rotations=rotations,
            scores=scores,
            details=details,
        )

    if args.min_score is not None and len(orientations.scores):
        keep = orientations.scores >= args.min_score
        orientations = orientations[keep]

    if args.max_score is not None and len(orientations.scores):
        keep = orientations.scores <= args.max_score
        orientations = orientations[keep]

    if args.peak_oversampling > 1:
        if data[0].ndim != data[2].ndim:
            print(
                "Input pickle does not contain template matching scores."
                " Cannot oversample peaks."
            )
            exit(-1)
        peak_caller = peak_caller = PEAK_CALLERS[args.peak_caller](shape=scores.shape)
        orientations.translations = peak_caller.oversample_peaks(
            scores=data[0],
            peak_positions=orientations.translations,
            oversampling_factor=args.peak_oversampling,
        )

    if args.local_optimization:
        target = Density.from_file(cli_args.target, use_memmap=True)
        orientations.translations = orientations.translations.astype(np.float32)
        orientations.rotations = orientations.rotations.astype(np.float32)
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
            orientations.rotations[index] = angles
            orientations.scores[index] = score * -1

    if args.output_format in ("orientations", "relion4", "relion5"):
        file_format, extension = "text", "tsv"

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

        orientations.to_file(
            filename=f"{args.output_prefix}.{extension}",
            file_format=file_format,
            source_path=basename(cli_args.target),
            version=version,
        )
        exit(0)

    target = Density.from_file(cli_args.target)
    if args.invert_target_contrast:
        target.data = target.data * -1

    if args.output_format in ("extraction"):
        if not np.all(np.divide(target.shape, template.shape) > 2):
            print(
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
            if args.mask_subtomograms:
                rotation_matrix = euler_to_rotationmatrix(orientations.rotations[index])
                mask_transfomed = template_mask.rigid_transform(
                    rotation_matrix=rotation_matrix, order=1
                )
                out_density.data = out_density.data * mask_transfomed.data
            out_density.to_file(
                join(working_directory, f"{args.output_prefix}_{index}.mrc")
            )

        exit(0)

    if args.output_format == "average":
        orientations, cand_slices, obs_slices = orientations.get_extraction_slices(
            target_shape=target.shape,
            extraction_shape=template.shape,
            drop_out_of_box=True,
            return_orientations=True,
        )
        out = np.zeros_like(template.data)
        for index in range(len(cand_slices)):
            subset = Density(target.data[obs_slices[index]])
            rotation_matrix = euler_to_rotationmatrix(orientations.rotations[index])

            subset = subset.rigid_transform(
                rotation_matrix=np.linalg.inv(rotation_matrix),
                order=1,
                use_geometric_center=True,
            )
            np.add(out, subset.data, out=out)

        out /= len(cand_slices)
        ret = Density(out, sampling_rate=template.sampling_rate, origin=0)
        ret.pad(template.shape, center=True)
        ret.to_file(f"{args.output_prefix}.mrc")
        exit(0)

    template, center, *_ = load_template(
        filepath=cli_args.template,
        sampling_rate=sampling_rate,
        centering=cli_args.centering,
    )

    for index, (translation, angles, *_) in enumerate(orientations):
        rotation_matrix = euler_to_rotationmatrix(angles)
        if template_is_density:
            transformed_template = template.rigid_transform(
                rotation_matrix=rotation_matrix, use_geometric_center=True
            )
            # Just adapting the coordinate system not the in-box position
            shift = np.multiply(np.subtract(translation, center), sampling_rate)
            transformed_template.origin = np.add(target_origin, shift)

        else:
            template = Structure.from_file(cli_args.template)
            new_center_of_mass = np.add(
                np.multiply(translation, sampling_rate), target_origin
            )
            translation = np.subtract(new_center_of_mass, center)
            transformed_template = template.rigid_transform(
                translation=translation,
                rotation_matrix=rotation_matrix,
            )
        # template_extension should contain '.'
        transformed_template.to_file(
            f"{args.output_prefix}_{index}{template_extension}"
        )
        index += 1


if __name__ == "__main__":
    main()
