#!python3
""" CLI to simplify analysing the output of match_template.py.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import argparse
from sys import exit
from os import getcwd
from typing import List, Tuple
from os.path import join, abspath, splitext

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfcinv

from tme import Density, Structure, Orientations
from tme.matching_utils import load_pickle, centered_mask
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
    parser = argparse.ArgumentParser(description="Analyze Template Matching Outputs")

    input_group = parser.add_argument_group("Input")
    output_group = parser.add_argument_group("Output")
    peak_group = parser.add_argument_group("Peak Calling")
    additional_group = parser.add_argument_group("Additional Parameters")

    input_group.add_argument(
        "--input_file",
        required=True,
        nargs="+",
        help="Path to the output of match_template.py.",
    )
    input_group.add_argument(
        "--background_file",
        required=False,
        nargs="+",
        help="Path to an output of match_template.py used for normalization. "
        "For instance from --scramble_phases or a different template.",
    )
    input_group.add_argument(
        "--target_mask",
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
        "--output_prefix",
        required=True,
        help="Output filename, extension will be added based on output_format.",
    )
    output_group.add_argument(
        "--output_format",
        choices=[
            "orientations",
            "relion4",
            "relion5",
            "alignment",
            "extraction",
            "average",
        ],
        default="orientations",
        help="Available output formats: "
        "orientations (translation, rotation, and score), "
        "relion4 (RELION 4 star format), "
        "relion5 (RELION 5 star format), "
        "alignment (aligned template to target based on orientations), "
        "extraction (extract regions around peaks from targets, i.e. subtomograms), "
        "average (extract matched regions from target and average them).",
    )

    peak_group.add_argument(
        "--peak_caller",
        choices=list(PEAK_CALLERS.keys()),
        default="PeakCallerScipy",
        help="Peak caller for local maxima identification.",
    )
    peak_group.add_argument(
        "--minimum_score",
        type=float,
        default=None,
        help="Minimum score from which peaks will be considered.",
    )
    peak_group.add_argument(
        "--maximum_score",
        type=float,
        default=None,
        help="Maximum score until which peaks will be considered.",
    )
    peak_group.add_argument(
        "--min_distance",
        type=int,
        default=5,
        help="Minimum distance between peaks.",
    )
    peak_group.add_argument(
        "--min_boundary_distance",
        type=int,
        default=0,
        help="Minimum distance of peaks to target edges.",
    )
    peak_group.add_argument(
        "--mask_edges",
        action="store_true",
        default=False,
        help="Whether candidates should not be identified from scores that were "
        "computed from padded densities. Superseded by min_boundary_distance.",
    )
    peak_group.add_argument(
        "--num_peaks",
        type=int,
        default=1000,
        required=False,
        help="Upper limit of peaks to call, subject to filtering parameters. Default 1000. "
        "If minimum_score is provided all peaks scoring higher will be reported.",
    )
    peak_group.add_argument(
        "--peak_oversampling",
        type=int,
        default=1,
        help="1 / factor equals voxel precision, e.g. 2 detects half voxel "
        "translations. Useful for matching structures to electron density maps.",
    )

    additional_group.add_argument(
        "--subtomogram_box_size",
        type=int,
        default=None,
        help="Subtomogram box size, by default equal to the centered template. Will be "
        "padded to even values if output_format is relion.",
    )
    additional_group.add_argument(
        "--mask_subtomograms",
        action="store_true",
        default=False,
        help="Whether to mask subtomograms using the template mask. The mask will be "
        "rotated according to determined angles.",
    )
    additional_group.add_argument(
        "--invert_target_contrast",
        action="store_true",
        default=False,
        help="Whether to invert the target contrast.",
    )
    additional_group.add_argument(
        "--n_false_positives",
        type=int,
        default=None,
        required=False,
        help="Number of accepted false-positives picks to determine minimum score.",
    )
    additional_group.add_argument(
        "--local_optimization",
        action="store_true",
        required=False,
        help="[Experimental] Perform local optimization of candidates. Useful when the "
        "number of identified candidats is small (< 10).",
    )

    args = parser.parse_args()

    if args.output_format == "relion" and args.subtomogram_box_size is not None:
        args.subtomogram_box_size += args.subtomogram_box_size % 2

    if args.orientations is not None:
        args.orientations = Orientations.from_file(filename=args.orientations)

    if args.background_file is None:
        args.background_file = [None]
    if len(args.background_file) == 1:
        args.background_file = args.background_file * len(args.input_file)
    elif len(args.background_file) not in (0, len(args.input_file)):
        raise ValueError(
            "--background_file needs to be specified once or for each --input_file."
        )

    return args


def load_template(
    filepath: str,
    sampling_rate: NDArray,
    centering: bool = True,
    target_shape: Tuple[int] = None,
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


def merge_outputs(data, foreground_paths: List[str], background_paths: List[str], args):
    if len(foreground_paths) == 0:
        return data, 1

    if data[0].ndim != data[2].ndim:
        return data, 1

    from tme.matching_exhaustive import normalize_under_mask

    def _norm_scores(data, args):
        target_origin, _, sampling_rate, cli_args = data[-1]

        _, template_extension = splitext(cli_args.template)
        ret = load_template(
            filepath=cli_args.template,
            sampling_rate=sampling_rate,
            centering=not cli_args.no_centering,
        )
        template, center_of_mass, translation, template_is_density = ret

        if args.mask_edges and args.min_boundary_distance == 0:
            max_shape = np.max(template.shape)
            args.min_boundary_distance = np.ceil(np.divide(max_shape, 2))

        target_mask = 1
        if args.target_mask is not None:
            target_mask = Density.from_file(args.target_mask).data
        elif cli_args.target_mask is not None:
            target_mask = Density.from_file(args.target_mask).data

        mask = np.ones_like(data[0])
        np.multiply(mask, target_mask, out=mask)

        cropped_shape = np.subtract(
            mask.shape, np.multiply(args.min_boundary_distance, 2)
        ).astype(int)
        mask[cropped_shape] = 0
        normalize_under_mask(template=data[0], mask=mask, mask_intensity=mask.sum())
        return data[0]

    entities = np.zeros_like(data[0])
    data[0] = _norm_scores(data=data, args=args)
    for index, filepath in enumerate(foreground_paths):
        new_scores = _norm_scores(
            data=load_match_template_output(filepath, background_paths[index]),
            args=args,
        )
        indices = new_scores > data[0]
        entities[indices] = index + 1
        data[0][indices] = new_scores[indices]

    return data, entities


def load_match_template_output(foreground_path, background_path):
    data = load_pickle(foreground_path)
    if background_path is not None:
        data_background = load_pickle(background_path)
        data[0] = (data[0] - data_background[0]) / (1 - data_background[0])
        np.fmax(data[0], 0, out=data[0])
    return data


def main():
    args = parse_args()
    data = load_match_template_output(args.input_file[0], args.background_file[0])

    target_origin, _, sampling_rate, cli_args = data[-1]

    _, template_extension = splitext(cli_args.template)
    ret = load_template(
        filepath=cli_args.template,
        sampling_rate=sampling_rate,
        centering=not cli_args.no_centering,
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

    entities = None
    if len(args.input_file) > 1:
        data, entities = merge_outputs(
            data=data,
            foreground_paths=args.input_file,
            background_paths=args.background_file,
            args=args,
        )

    orientations = args.orientations
    if orientations is None:
        translations, rotations, scores, details = [], [], [], []
        # Output is MaxScoreOverRotations
        if data[0].ndim == data[2].ndim:
            scores, offset, rotation_array, rotation_mapping, meta = data

            if args.target_mask is not None:
                target_mask = Density.from_file(args.target_mask)
                scores = scores * target_mask.data

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
                args.minimum_score = minimum_score

            args.batch_dims = None
            if hasattr(cli_args, "target_batch"):
                args.batch_dims = cli_args.target_batch

            peak_caller_kwargs = {
                "shape": scores.shape,
                "num_peaks": args.num_peaks,
                "min_distance": args.min_distance,
                "min_boundary_distance": args.min_boundary_distance,
                "batch_dims": args.batch_dims,
                "minimum_score": args.minimum_score,
                "maximum_score": args.maximum_score,
            }

            peak_caller = PEAK_CALLERS[args.peak_caller](**peak_caller_kwargs)
            peak_caller(
                scores,
                rotation_matrix=np.eye(template.data.ndim),
                mask=template.data,
                rotation_mapping=rotation_mapping,
                rotations=rotation_array,
            )
            candidates = peak_caller.merge(
                candidates=[tuple(peak_caller)], **peak_caller_kwargs
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

        else:
            candidates = data
            translation, rotation, *_ = data
            for i in range(translation.shape[0]):
                rotations.append(euler_from_rotationmatrix(rotation[i]))

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

    if args.minimum_score is not None and len(orientations.scores):
        keep = orientations.scores >= args.minimum_score
        orientations = orientations[keep]

    if args.maximum_score is not None and len(orientations.scores):
        keep = orientations.scores <= args.maximum_score
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
            source_path=cli_args.target,
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
        if args.subtomogram_box_size is not None:
            extraction_shape = np.repeat(
                args.subtomogram_box_size, len(extraction_shape)
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
        centering=not cli_args.no_centering,
        target_shape=target.shape,
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
