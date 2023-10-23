#!python3
""" Estimate RAM requirements for template matching jobs.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import numpy as np
import argparse
from tme import Density
from tme.matching_utils import estimate_ram_usage


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
        "--analyzer_method",
        required=False,
        default=None,
        help="Analyzer method to use.",
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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    target = Density.from_file(args.target)
    template = Density.from_file(args.template)

    target_box = target.shape
    if not args.no_edge_padding:
        target_box = np.add(target_box, template.shape)

    result = estimate_ram_usage(
        shape1=target_box,
        shape2=template.shape,
        matching_method=args.matching_method,
        ncores=args.ncores,
        analyzer_method=args.analyzer_method,
    )
    print(result)


if __name__ == "__main__":
    main()
