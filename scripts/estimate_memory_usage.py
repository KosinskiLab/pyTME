#!python3
"""Estimate memory requirements for template matching jobs.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import numpy as np
import argparse
from tme import Density
from tme.memory import estimate_memory_usage
from tme.matching_exhaustive import MATCHING_EXHAUSTIVE_REGISTER


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate memory usage for template matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--target",
        type=str,
        required=True,
        help="Path to a target in CCP4/MRC format.",
    )
    parser.add_argument(
        "-i",
        "--template",
        type=str,
        required=True,
        help="Path to a template in PDB/MMCIF or CCP4/MRC format.",
    )
    parser.add_argument(
        "-s",
        "--score",
        type=str,
        default="FLCSphericalMask",
        help="Template matching scoring function.",
        choices=MATCHING_EXHAUSTIVE_REGISTER.keys(),
    )
    parser.add_argument(
        "--ncores", type=int, help="Number of cores for parallelization.", required=True
    )
    parser.add_argument(
        "--pad-edges",
        action="store_true",
        default=False,
        help="Whether to pad the edges of the target. Useful if the target does not "
        "a well-defined bounding box. Defaults to True if splitting is required.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    target = Density.from_file(args.target, use_memmap=True)
    template = Density.from_file(args.template, use_memmap=True)

    template_box = template.shape
    if not args.pad_edges:
        template_box = np.ones(len(template_box), dtype=int)

    result = estimate_memory_usage(
        shape1=target.shape,
        shape2=template_box,
        matching_method=args.score,
        ncores=args.ncores,
        analyzer_method="MaxScoreOverRotations",
    )
    print(result)


if __name__ == "__main__":
    main()
