#!python3
""" CLI to simplify analysing the output of match_template.py.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import warnings
import argparse
from sys import exit
from os import getcwd
from os.path import join, abspath
from typing import List
from os.path import splitext

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfcinv

from tme import Density, Structure, Orientations
from tme.analyzer import (
    PeakCallerSort,
    PeakCallerMaximumFilter,
    PeakCallerFast,
    PeakCallerRecursiveMasking,
    PeakCallerScipy,
)
from tme.matching_utils import (
    load_pickle,
    euler_to_rotationmatrix,
    euler_from_rotationmatrix,
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
        description="Peak Calling for Template Matching Outputs"
    )

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
            "alignment",
            "extraction",
            "relion",
            "backmapping",
            "average",
        ],
        default="orientations",
        help="Available output formats:"
        "orientations (translation, rotation, and score), "
        "alignment (aligned template to target based on orientations), "
        "extraction (extract regions around peaks from targets, i.e. subtomograms), "
        "relion (perform extraction step and generate corresponding star files), "
        "backmapping (map template to target using identified peaks),"
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
        "--number_of_peaks",
        type=int,
        default=None,
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
        "--wedge_mask",
        type=str,
        default=None,
        help="Path to file used as ctf_mask for output_format relion.",
    )
    additional_group.add_argument(
        "--n_false_positives",
        type=int,
        default=None,
        required=False,
        help="Number of accepted false-positives picks to determine minimum score.",
    )

    args = parser.parse_args()

    if args.wedge_mask is not None:
        args.wedge_mask = abspath(args.wedge_mask)

    if args.output_format == "relion" and args.subtomogram_box_size is not None:
        args.subtomogram_box_size += args.subtomogram_box_size % 2

    if args.orientations is not None:
        args.orientations = Orientations.from_file(filename=args.orientations)

    if args.minimum_score is not None or args.n_false_positives is not None:
        args.number_of_peaks = np.iinfo(np.int64).max
    elif args.number_of_peaks is None:
        args.number_of_peaks = 1000

    return args


def load_template(filepath: str, sampling_rate: NDArray, center: bool = True):
    try:
        template = Density.from_file(filepath)
        center_of_mass = template.center_of_mass(template.data)
        template_is_density = True
    except Exception:
        template = Structure.from_file(filepath)
        center_of_mass = template.center_of_mass()[::-1]
        template = Density.from_structure(template, sampling_rate=sampling_rate)
        template_is_density = False

    translation = np.zeros_like(center_of_mass)
    if center:
        template, translation = template.centered(0)

    return template, center_of_mass, translation, template_is_density


def merge_outputs(data, filepaths: List[str], args):
    if len(filepaths) == 0:
        return data, 1

    if data[0].ndim != data[2].ndim:
        return data, 1

    from tme.matching_exhaustive import _normalize_under_mask

    def _norm_scores(data, args):
        target_origin, _, sampling_rate, cli_args = data[-1]

        _, template_extension = splitext(cli_args.template)
        ret = load_template(
            filepath=cli_args.template,
            sampling_rate=sampling_rate,
            center=not cli_args.no_centering,
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
        _normalize_under_mask(template=data[0], mask=mask, mask_intensity=mask.sum())
        return data[0]

    entities = np.zeros_like(data[0])
    data[0] = _norm_scores(data=data, args=args)
    for index, filepath in enumerate(filepaths):
        new_scores = _norm_scores(data=load_pickle(filepath), args=args)
        indices = new_scores > data[0]
        entities[indices] = index + 1
        data[0][indices] = new_scores[indices]

    return data, entities


def main():
    args = parse_args()
    data = load_pickle(args.input_file[0])

    target_origin, _, sampling_rate, cli_args = data[-1]

    _, template_extension = splitext(cli_args.template)
    ret = load_template(
        filepath=cli_args.template,
        sampling_rate=sampling_rate,
        center=not cli_args.no_centering,
    )
    template, center_of_mass, translation, template_is_density = ret

    if args.output_format == "relion" and args.subtomogram_box_size is None:
        new_shape = np.add(template.shape, np.mod(template.shape, 2))
        new_shape = np.repeat(new_shape.max(), new_shape.size).astype(int)
        print(f"Padding template from {template.shape} to {new_shape} for RELION.")
        template.pad(new_shape)

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

    # data, entities = merge_outputs(data=data, filepaths=args.input_file[1:], args=args)

    orientations = args.orientations
    if orientations is None:
        translations, rotations, scores, details = [], [], [], []
        # Output is MaxScoreOverRotations
        if data[0].ndim == data[2].ndim:
            scores, offset, rotation_array, rotation_mapping, meta = data

            if args.target_mask is not None:
                target_mask = Density.from_file(args.target_mask)
                scores = scores * target_mask.data

            if args.n_false_positives is not None:
                args.n_false_positives = max(args.n_false_positives, 1)
                cropped_shape = np.subtract(
                    scores.shape, np.multiply(args.min_boundary_distance, 2)
                ).astype(int)

                cropped_shape = tuple(
                    slice(
                        int(args.min_boundary_distance),
                        int(x - args.min_boundary_distance),
                    )
                    for x in scores.shape
                )
                # Rickgauer et al. 2017
                n_correlations = np.size(scores[cropped_shape]) * len(rotation_mapping)
                minimum_score = np.multiply(
                    erfcinv(2 * args.n_false_positives / n_correlations),
                    np.sqrt(2) * np.std(scores[cropped_shape]),
                )
                print(f"Determined minimum score cutoff: {minimum_score}.")
                minimum_score = max(minimum_score, 0)
                args.minimum_score = minimum_score

            peak_caller = PEAK_CALLERS[args.peak_caller](
                number_of_peaks=args.number_of_peaks,
                min_distance=args.min_distance,
                min_boundary_distance=args.min_boundary_distance,
            )

            peak_caller(
                scores,
                rotation_matrix=np.eye(3),
                mask=template.data,
                rotation_mapping=rotation_mapping,
                rotation_array=rotation_array,
                minimum_score=args.minimum_score,
            )
            candidates = peak_caller.merge(
                candidates=[tuple(peak_caller)],
                number_of_peaks=args.number_of_peaks,
                min_distance=args.min_distance,
                min_boundary_distance=args.min_boundary_distance,
            )
            if len(candidates) == 0:
                candidates = [[], [], [], []]
                warnings.warn(
                    "Found no peaks, consider changing peak calling parameters."
                )

            for translation, _, score, detail in zip(*candidates):
                rotations.append(rotation_mapping[rotation_array[tuple(translation)]])

        else:
            candidates = data
            translation, rotation, *_ = data
            for i in range(translation.shape[0]):
                rotations.append(euler_from_rotationmatrix(rotation[i]))

        if len(rotations):
            rotations = np.vstack(rotations).astype(float)
        translations, scores, details = candidates[0], candidates[2], candidates[3]
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
        peak_caller = peak_caller = PEAK_CALLERS[args.peak_caller]()
        if data[0].ndim != data[2].ndim:
            print(
                "Input pickle does not contain template matching scores."
                " Cannot oversample peaks."
            )
            exit(-1)
        orientations.translations = peak_caller.oversample_peaks(
            score_space=data[0],
            peak_positions=orientations.translations,
            oversampling_factor=args.peak_oversampling,
        )

    if args.output_format == "orientations":
        orientations.to_file(filename=f"{args.output_prefix}.tsv", file_format="text")
        exit(0)

    target = Density.from_file(cli_args.target)
    if args.invert_target_contrast:
        if args.output_format == "relion":
            target.data = target.data * -1
            target.data = np.divide(
                np.subtract(target.data, target.data.mean()), target.data.std()
            )
        else:
            target.data = (
                -np.divide(
                    np.subtract(target.data, target.data.min()),
                    np.subtract(target.data.max(), target.data.min()),
                )
                + 1
            )

    if args.output_format in ("extraction", "relion"):
        if not np.all(np.divide(target.shape, template.shape) > 2):
            print(
                "Target might be too small relative to template to extract"
                " meaningful particles."
                f" Target : {target.shape}, template : {template.shape}."
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
        if args.output_format == "relion":
            orientations.to_file(
                filename=f"{args.output_prefix}.star",
                file_format="relion",
                name_prefix=join(working_directory, args.output_prefix),
                ctf_image=args.wedge_mask,
                sampling_rate=target.sampling_rate.max(),
                subtomogram_size=extraction_shape[0],
            )

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

    if args.output_format == "backmapping":
        orientations, cand_slices, obs_slices = orientations.get_extraction_slices(
            target_shape=target.shape,
            extraction_shape=template.shape,
            drop_out_of_box=True,
            return_orientations=True,
        )
        ret, template_sum = target.empty, template.data.sum()
        for index in range(len(cand_slices)):
            rotation_matrix = euler_to_rotationmatrix(orientations.rotations[index])

            transformed_template = template.rigid_transform(
                rotation_matrix=rotation_matrix
            )
            transformed_template.data = np.multiply(
                transformed_template.data,
                np.divide(template_sum, transformed_template.data.sum()),
            )
            cand_slice, obs_slice = cand_slices[index], obs_slices[index]
            ret.data[obs_slice] += transformed_template.data[cand_slice]
        ret.to_file(f"{args.output_prefix}_backmapped.mrc")
        exit(0)

    if args.output_format == "average":
        orientations, cand_slices, obs_slices = orientations.get_extraction_slices(
            target_shape=target.shape,
            extraction_shape=np.multiply(template.shape, 2),
            drop_out_of_box=True,
            return_orientations=True,
        )
        out = np.zeros_like(template.data)
        out = np.zeros(np.multiply(template.shape, 2).astype(int))
        for index in range(len(cand_slices)):
            from scipy.spatial.transform import Rotation

            rotation = Rotation.from_euler(
                angles=orientations.rotations[index], seq="zyx", degrees=True
            )
            rotation_matrix = rotation.inv().as_matrix()

            # rotation_matrix = euler_to_rotationmatrix(orientations.rotations[index])
            subset = Density(target.data[obs_slices[index]])
            subset = subset.rigid_transform(rotation_matrix=rotation_matrix, order=1)

            np.add(out, subset.data, out=out)
        out /= len(cand_slices)
        ret = Density(out, sampling_rate=template.sampling_rate, origin=0)
        ret.pad(template.shape, center=True)
        ret.to_file(f"{args.output_prefix}_average.mrc")
        exit(0)

    for index, (translation, angles, *_) in enumerate(orientations):
        rotation_matrix = euler_to_rotationmatrix(angles)
        if template_is_density:
            translation = np.subtract(translation, center_of_mass)
            transformed_template = template.rigid_transform(
                rotation_matrix=rotation_matrix
            )
            new_origin = np.add(target_origin / sampling_rate, translation)
            transformed_template.origin = np.multiply(new_origin, sampling_rate)
        else:
            template = Structure.from_file(cli_args.template)
            new_center_of_mass = np.add(
                np.multiply(translation, sampling_rate), target_origin
            )
            translation = np.subtract(new_center_of_mass, center_of_mass)
            transformed_template = template.rigid_transform(
                translation=translation[::-1],
                rotation_matrix=rotation_matrix[::-1, ::-1],
            )
        # template_extension should contain '.'
        transformed_template.to_file(
            f"{args.output_prefix}_{index}{template_extension}"
        )
        index += 1


if __name__ == "__main__":
    main()
