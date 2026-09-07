#!python3
"""Evaluate predicted peaks against ground truth positions.

Copyright (c) 2026 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import argparse

import yaml
import numpy as np

from tme import Orientations
from tme.utils.evaluation import compute_metrics
from tme.rotations import euler_to_rotationmatrix
from tme.utils.logging import get_logger, setup_logging

logger = get_logger("evaluate")


def main():
    parser = argparse.ArgumentParser(
        prog="pytme utils evaluate",
        description="Evaluate predicted peaks against ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("pytme utils evaluate -i peaks.star --gt gt.star --distance 10\n"),
    )

    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument(
        "-i",
        "--input",
        required=True,
        help="Predicted peaks (.star, .tsv).",
    )
    io_group.add_argument(
        "--gt",
        required=True,
        help="Ground truth positions (.star, .tsv).",
    )
    io_group.add_argument(
        "-o",
        "--output-prefix",
        type=str,
        default=None,
        help="If given, write <prefix>_metrics.yaml and " "<prefix>_annotations.star.",
    )

    match_group = parser.add_argument_group("Matching")
    match_group.add_argument(
        "--distance",
        type=float,
        required=True,
        help="Maximum match distance, in the units of --gt after --scale is applied.",
    )
    match_group.add_argument(
        "--angle",
        type=float,
        default=None,
        help="Maximum angular error in degrees for a match.",
    )
    match_group.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiply predicted coordinates by this factor before "
        "matching (e.g. to convert voxels to angstrom).",
    )

    args = parser.parse_args()
    setup_logging()

    predicted = Orientations.from_file(args.input)
    ground_truth = Orientations.from_file(args.gt)

    pred_translations = predicted.translations.copy()
    if args.scale != 1.0:
        pred_translations *= args.scale

    gt_rotations, pred_rotations = None, None
    if args.angle is not None:
        pred_rotations = np.array(
            [euler_to_rotationmatrix(r) for r in predicted.rotations]
        )
        gt_rotations = np.array(
            [euler_to_rotationmatrix(r) for r in ground_truth.rotations]
        )

    result = compute_metrics(
        predicted_translations=pred_translations,
        predicted_scores=predicted.scores,
        gt_translations=ground_truth.translations,
        distance_threshold=args.distance,
        angle_threshold=args.angle,
        predicted_rotations=pred_rotations,
        gt_rotations=gt_rotations,
    )

    summary = {
        "f1": result["f1"],
        "precision": result["precision"],
        "recall": result["recall"],
        "threshold": result["threshold"],
        "n_predicted": result["n_predicted"],
        "n_ground_truth": result["n_ground_truth"],
        "n_tp": result["n_tp"],
        "n_fp": result["n_fp"],
        "n_gt_matched": result["n_gt_matched"],
        "n_gt_missed": result["n_gt_missed"],
    }

    logger.info(
        f"F1={summary['f1']:.4f}  precision={summary['precision']:.4f}  "
        f"recall={summary['recall']:.4f}  threshold={summary['threshold']:.4f}"
    )
    logger.info(
        f"TP={summary['n_tp']}  FP={summary['n_fp']}  "
        f"GT matched={summary['n_gt_matched']}/{summary['n_ground_truth']}"
    )

    if args.output_prefix is not None:
        metrics_path = f"{args.output_prefix}_metrics.yaml"
        with open(metrics_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

        matching = result["matching"]
        annotated = Orientations(
            translations=pred_translations,
            rotations=predicted.rotations,
            metadata={
                **predicted.metadata,
                "_rlnClassNumber": matching["tp_mask"].astype(np.float32),
                "_pytmeMatchDistance": matching["distances"].astype(np.float32),
                "_pytmeMatchGTIndex": matching["gt_indices"],
                "_pytmeMatchAngleError": matching["angle_errors"].astype(np.float32),
            },
        )
        star_path = f"{args.output_prefix}_annotations.star"
        annotated.to_file(star_path)
        logger.info(f"Written: {metrics_path}, {star_path}")
