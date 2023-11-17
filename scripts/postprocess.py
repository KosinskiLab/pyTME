#!python3
""" CLI to simplify analysing the output of match_template.py.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import argparse
from sys import exit
from os.path import splitext

import numpy as np

from tme import Density, Structure
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
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the output of match_template.py.",
    )
    parser.add_argument(
        "--output_prefix",
        required=True,
        help="Prefix for the output file name. Extension depends on output_format.",
    )
    parser.add_argument(
        "--number_of_peaks", type=int, default=1000, help="Number of peaks to consider."
    )
    parser.add_argument(
        "--min_distance", type=int, default=5, help="Minimum distance between peaks."
    )
    parser.add_argument(
        "--peak_caller",
        choices=list(PEAK_CALLERS.keys()),
        default="PeakCallerScipy",
        help="Peak caller to use for analysis. Ignored if input_file contains peaks.",
    )
    parser.add_argument(
        "--orientations",
        default=None,
        help="Path to orientations file to overwrite orientations computed from"
        " match_template.py output.",
    )
    parser.add_argument(
        "--output_format",
        choices=["orientations", "alignment", "extraction"],
        default="orientations",
        help="Choose the output format. Available formats are: "
        "orientations (translation, rotation, and score), "
        "alignment (aligned template to target based on orientations), "
        "extraction (extract regions around peaks from targets, i.e. subtomograms).",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    data = load_pickle(args.input_file)

    meta = data[-1]

    orientations = []
    if args.orientations is None:
        if data[0].ndim == data[2].ndim:
            scores, offset, rotations, rotation_mapping, meta = data
            peak_caller = PEAK_CALLERS[args.peak_caller](
                number_of_peaks=args.number_of_peaks, min_distance=args.min_distance
            )
            peak_caller(scores, rotation_matrix=np.eye(3))
            candidates = tuple(peak_caller)
            for translation, _, score, detail in zip(*candidates):
                angles = rotation_mapping[rotations[tuple(translation)]]
                orientations.append((translation, angles, score, detail))
        else:
            candidates = data
            translation, rotation, score, detail, *_ = data
            for i in range(translation.shape[0]):
                angles = euler_from_rotationmatrix(rotation[i])
                orientations.append(
                    (np.array(translation[i]), angles, score[i], detail[i])
                )
    else:
        with open(args.orientations, mode="r", encoding="utf-8") as infile:
            data = [x.strip().split("\t") for x in infile.read().split("\n")]
            _ = data.pop(0)
        translation, rotation, score, detail = [], [], [], []
        for candidate in data:
            if len(candidate) <= 1:
                continue
            if len(candidate) != 8:
                candidate.append(-1)

            candidate = [float(x) for x in candidate]
            translation.append((candidate[0], candidate[1], candidate[2]))
            rotation.append(
                euler_to_rotationmatrix((candidate[3], candidate[4], candidate[5]))
            )
            score.append(candidate[6])
            detail.append(candidate[7])
            orientations.append(
                (
                    translation[-1],
                    (candidate[3], candidate[4], candidate[5]),
                    score[-1],
                    detail[-1],
                )
            )

        candidates = (
            np.vstack(translation).astype(int),
            np.vstack(rotation).astype(float),
            np.array(score).astype(float),
            np.array(detail).astype(float),
        )

    if args.output_format == "orientations":
        header = "\t".join(
            ["z", "y", "x", "euler_z", "euler_y", "euler_x", "score", "detail"]
        )
        output_file = f"{args.output_prefix}.tsv"
        with open(output_file, mode="w", encoding="utf-8") as ofile:
            _ = ofile.write(f"{header}\n")
            for translation, angles, score, detail in orientations:
                translation_string = "\t".join([str(x) for x in translation])
                angle_string = "\t".join([str(x) for x in angles])
                _ = ofile.write(
                    f"{translation_string}\t{angle_string}\t{score}\t{detail}\n"
                )
        exit(0)

    target_origin, _, sampling_rate, cli_args = meta

    template_is_density, index = True, 0
    _, template_extension = splitext(cli_args.template)
    try:
        template = Density.from_file(cli_args.template)
        template, _ = template.centered(0)
        center_of_mass = template.center_of_mass(template.data)
    except ValueError:
        template_is_density = False
        template = Structure.from_file(cli_args.template)
        center_of_mass = template.center_of_mass()[::-1]

    if args.output_format == "extraction":
        target = Density.from_file(cli_args.target)

        if not np.all(np.divide(target.shape, template.shape) > 2):
            print(
                "Target might be too small relative to template to extract"
                " meaningful particles."
                f" Target : {target.shape}, template : {template.shape}."
            )

        peaks = candidates[0].astype(int)
        max_shape = np.max(template.shape).astype(int)
        half_shape = max_shape // 2

        left_pad = half_shape
        right_pad = np.add(half_shape, max_shape % 2)
        starts = np.subtract(peaks, left_pad)
        stops = np.add(peaks, right_pad)

        candidate_starts = np.maximum(starts, 0).astype(int)
        candidate_stops = np.minimum(stops, target.shape).astype(int)
        keep_peaks = (
            np.sum(
                np.multiply(starts == candidate_starts, stops == candidate_stops),
                axis=1,
            )
            == peaks.shape[1]
        )

        peaks = peaks[keep_peaks,]
        starts = starts[keep_peaks,]
        stops = stops[keep_peaks,]
        candidate_starts = candidate_starts[keep_peaks,]
        candidate_stops = candidate_stops[keep_peaks,]

        if not len(peaks):
            print(
                "No peak remaining after filtering. Started with"
                f" {candidates[0].shape[0]} filtered to {peaks.shape[0]}."
                " Consider reducing min_distance, increase num_peaks or use"
                " a different peak caller."
            )
            exit(-1)

        observation_starts = np.subtract(candidate_starts, starts).astype(int)
        observation_stops = np.subtract(np.add(max_shape, candidate_stops), stops)
        observation_stops = observation_stops.astype(int)

        candidate_slices = [
            tuple(slice(s, e) for s, e in zip(start_row, stop_row))
            for start_row, stop_row in zip(candidate_starts, candidate_stops)
        ]

        observation_slices = [
            tuple(slice(s, e) for s, e in zip(start_row, stop_row))
            for start_row, stop_row in zip(observation_starts, observation_stops)
        ]
        observations = np.zeros(
            (len(candidate_slices), max_shape, max_shape, max_shape)
        )

        slices = zip(candidate_slices, observation_slices)
        for idx, (cand_slice, obs_slice) in enumerate(slices):
            observations[idx][:] = np.mean(target.data[cand_slice])
            observations[idx][obs_slice] = target.data[cand_slice]

        for index in range(observations.shape[0]):
            Density(
                data=observations[index],
                sampling_rate=sampling_rate,
                origin=candidate_starts[index] * sampling_rate,
            ).to_file(f"{args.output_prefix}{index}.mrc")
        exit(0)

    for translation, angles, *_ in orientations:
        rotation_matrix = euler_to_rotationmatrix(angles)

        if template_is_density:
            translation = np.subtract(translation, center_of_mass)
            transformed_template = template.rigid_transform(
                rotation_matrix=rotation_matrix
            )
            new_origin = np.add(target_origin / sampling_rate, translation)
            transformed_template.origin = np.multiply(new_origin, sampling_rate)
        else:
            new_center_of_mass = np.add(
                np.multiply(translation, sampling_rate), target_origin
            )
            translation = np.subtract(new_center_of_mass, center_of_mass)
            transformed_template = template.rigid_transform(
                translation=translation[::-1],
                rotation_matrix=rotation_matrix[::-1, ::-1],
            )
        transformed_template.to_file(f"{args.output_prefix}{index}{template_extension}")
        index += 1


if __name__ == "__main__":
    main()
