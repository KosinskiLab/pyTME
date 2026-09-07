#!python3
"""Compute a simple subtomogram average.

Copyright (c) 2026 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import os
import argparse
from sys import exit

import numpy as np

from tme import Density, Orientations
from tme.rotations import euler_to_rotationmatrix
from tme.utils.logging import get_logger, setup_logging

logger = get_logger("average")


def _extract_box(data, cand_slice, obs_slice, shape, normalize):
    box = np.zeros(shape, dtype=np.float64)
    box[cand_slice] = data[obs_slice]
    if normalize:
        std = box.std()
        if std > 0:
            box = (box - box.mean()) / std
    return box


def main():
    parser = argparse.ArgumentParser(
        description="Compute a simple subtomogram average.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "pytme utils average -t tomo1.mrc tomo2.mrc \\\n"
            "    -s tomo1.star tomo2.star -b 64 -o average.mrc\n"
        ),
    )

    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument(
        "-t",
        "--tomograms",
        nargs="+",
        required=True,
        help="Tomogram files. Pair with --starfiles by position.",
    )
    io_group.add_argument(
        "-s",
        "--starfiles",
        nargs="+",
        required=True,
        help="Star files with particle coordinates and orientations.",
    )
    io_group.add_argument(
        "-b",
        "--box-size",
        type=int,
        required=True,
        help="Edge length of the averaging box in voxels.",
    )
    io_group.add_argument(
        "-o",
        "--output",
        default="average.mrc",
        help="Output average path. Default average.mrc.",
    )

    proc_group = parser.add_argument_group("Processing")
    proc_group.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiply star-file coordinates by this factor before extraction.",
    )
    proc_group.add_argument(
        "--pad",
        action="store_true",
        help="Zero-pad particles extending past tomogram edges. Default drops them.",
    )
    proc_group.add_argument(
        "--normalize",
        action="store_true",
        help="Standardize each particle to zero mean, unit variance before summing.",
    )
    proc_group.add_argument(
        "--invert",
        action="store_true",
        help="Negate the average. Use when protein density is dark in the tomogram.",
    )

    args = parser.parse_args()
    setup_logging()

    if len(args.tomograms) != len(args.starfiles):
        parser.error("Provide one star file per tomogram.")

    out, sampling_rate, total = None, None, 0
    for tomo_path, star_path in zip(args.tomograms, args.starfiles):
        tomogram = Density.from_file(tomo_path)
        orientations = Orientations.from_file(star_path)

        if args.scale != 1.0:
            orientations.translations = orientations.translations * args.scale

        shape = np.repeat(args.box_size, tomogram.data.ndim)
        if out is None:
            out = np.zeros(shape, dtype=np.float32)
            sampling_rate = tomogram.sampling_rate

        orientations, cand, obs = orientations.get_extraction_slices(
            target_shape=tomogram.shape,
            extraction_shape=shape,
            drop_out_of_box=not args.pad,
            return_orientations=True,
        )

        n = len(cand)
        logger.info(f"{tomo_path}: {n} particles")
        for i in range(n):
            box = _extract_box(tomogram.data, cand[i], obs[i], shape, args.normalize)
            matrix = euler_to_rotationmatrix(orientations.rotations[i]).T
            rotated = Density(box).rigid_transform(
                rotation_matrix=matrix,
                order=1,
                use_geometric_center=True,
            )
            np.add(out, rotated.data, out=out)
        total += n

    if total == 0:
        exit("No valid particles found.")

    out /= total
    if args.invert:
        out = -out

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    Density(out, sampling_rate=sampling_rate, origin=0).to_file(args.output)
    logger.info(f"Wrote average of {total} particles to {args.output}")


if __name__ == "__main__":
    main()
