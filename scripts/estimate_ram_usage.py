#!python3
""" Estimate RAM requirements for template matching jobs.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import numpy as np
import argparse
from tme import Density
from tme.matching_utils import estimate_ram_usage
from tme.matching_exhaustive import MATCHING_EXHAUSTIVE_REGISTER


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate RAM usage for template matching."
    )
    parser.add_argument(
        "-m",
        "--target",
        dest="target",
        type=str,
        required=True,
        help="Path to a target in CCP4/MRC format.",
    )
    parser.add_argument(
        "-i",
        "--template",
        dest="template",
        type=str,
        required=True,
        help="Path to a template in PDB/MMCIF or CCP4/MRC format.",
    )
    parser.add_argument(
        "--matching_method",
        required=False,
        default=None,
        help="Analyzer method to use.",
    )
    parser.add_argument(
        "-s",
        dest="score",
        type=str,
        default="FLCSphericalMask",
        help="Template matching scoring function.",
        choices=MATCHING_EXHAUSTIVE_REGISTER.keys(),
    )
    parser.add_argument(
        "--ncores", type=int, help="Number of cores for parallelization.", required=True
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
        " for numerical stability. When working with very large targets such as"
        " tomograms it is safe to use this flag and benefit from the performance gain.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    target = Density.from_file(args.target)
    template = Density.from_file(args.template)

    target_box = target.shape
    if not args.no_edge_padding:
        target_box = np.add(target_box, template.shape)

    template_box = template.shape
    if args.no_fourier_padding:
        template_box = np.ones(len(template_box), dtype=int)

    result = estimate_ram_usage(
        shape1=target_box,
        shape2=template_box,
        matching_method=args.score,
        ncores=args.ncores,
        analyzer_method="MaxScoreOverRotations"
    )
    print(result)


if __name__ == "__main__":
    main()
