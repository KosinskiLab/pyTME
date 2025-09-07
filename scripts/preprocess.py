#!python3
"""Preprocessing routines for template matching.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import argparse
import numpy as np

from tme import Density, Structure
from tme.cli import print_entry
from tme.backends import backend as be
from tme.filters import BandPassReconstructed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform template matching preprocessing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument(
        "-m",
        "--data",
        type=str,
        required=True,
        help="Path to a file in PDB/MMCIF, CCP4/MRC, EM, H5 or a format supported by "
        "tme.density.Density.from_file "
        "https://kosinskilab.github.io/pyTME/reference/api/tme.density.Density.from_file.html",
    )
    io_group.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path the output should be written to.",
    )

    box_group = parser.add_argument_group("Box")
    box_group.add_argument(
        "--box-size",
        type=int,
        required=False,
        help="Box size of the output. Defaults to twice the required box size.",
    )
    box_group.add_argument(
        "--sampling-rate",
        type=float,
        required=True,
        help="Sampling rate of the output file.",
    )
    box_group.add_argument(
        "--input-sampling-rate",
        type=float,
        required=False,
        help="Sampling rate of the input file. Defaults to header for volume "
        "and to --sampling_rate for atomic structures.",
    )

    modulation_group = parser.add_argument_group("Modulation")
    modulation_group.add_argument(
        "--invert-contrast",
        action="store_true",
        required=False,
        help="Inverts the template contrast.",
    )
    modulation_group.add_argument(
        "--lowpass",
        type=float,
        required=False,
        default=None,
        help="Lowpass filter the template to the given resolution. Nyquist by default. "
        "A value of 0 disables the filter.",
    )
    modulation_group.add_argument(
        "--no-centering",
        action="store_true",
        help="Assumes the template is already centered and omits centering.",
    )
    modulation_group.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=be.available_backends(),
        help="Determines more suitable box size for the given compute backend.",
    )

    alignment_group = parser.add_argument_group("Modulation")
    alignment_group.add_argument(
        "--align-axis",
        type=int,
        required=False,
        help="Align template to given axis, e.g. 2 for z-axis.",
    )
    alignment_group.add_argument(
        "--align-eigenvector",
        type=int,
        required=False,
        default=0,
        help="Eigenvector to use for alignment. Defaults to 0, i.e. the eigenvector "
        "with numerically largest eigenvalue.",
    )
    alignment_group.add_argument(
        "--flip-axis",
        action="store_true",
        required=False,
        help="Align the template to -axis instead of axis.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print_entry()

    try:
        data = Structure.from_file(args.data)
        sampling_rate = args.sampling_rate
        if args.input_sampling_rate is not None:
            sampling_rate = args.input_sampling_rate

        if args.align_axis is not None:
            rmat = data.align_to_axis(axis=args.align_axis, flip=args.flip_axis)
            data = data.rigid_transform(
                rotation_matrix=rmat.T, use_geometric_center=True
            )
        data = Density.from_structure(data, sampling_rate=sampling_rate)

    except NotImplementedError:
        data = Density.from_file(args.data)
        if args.input_sampling_rate is not None:
            data.sampling_rate = args.input_sampling_rate

        if args.align_axis is not None:
            rmat = data.align_to_axis(axis=args.align_axis, flip=args.flip_axis)
            data = data.rigid_transform(
                rotation_matrix=rmat.T, use_geometric_center=True
            )

    if not args.no_centering:
        data = data.centered(0)

    if args.box_size is None:
        scale = np.divide(data.sampling_rate, args.sampling_rate)
        args.box_size = int(np.ceil(2 * np.max(np.multiply(scale, data.shape)))) + 1

    default_backend = be._backend_name
    for name in be.available_backends():
        be.change_backend(name, device="cpu")
        box = be.compute_convolution_shapes([args.box_size], [1])[1][0]
        if box != args.box_size and args.backend is None:
            print(f"Consider --box_size {box} instead of {args.box_size} for {name}.")

    if args.backend is not None:
        be.change_backend(args.backend, device="cpu")

        box = be.compute_convolution_shapes([args.box_size], [1])[1][0]
        if box != args.box_size:
            print(f"Changed --box_size from {args.box_size} to {box}.")
        args.box_size = box

    be.change_backend(default_backend)

    data.pad(
        np.multiply(args.box_size, np.divide(args.sampling_rate, data.sampling_rate)),
        center=True,
    )

    bpf_mask = 1
    lowpass = 2 * args.sampling_rate if args.lowpass is None else args.lowpass
    if args.lowpass != 0:
        bpf_mask = BandPassReconstructed(
            lowpass=lowpass,
            highpass=None,
            use_gaussian=True,
            sampling_rate=data.sampling_rate,
        )(shape=data.shape, return_real_fourier=True)["data"]
        bpf_mask = be.to_backend_array(bpf_mask)

        data_ft = be.rfftn(be.to_backend_array(data.data), s=data.shape)
        data_ft = be.multiply(data_ft, bpf_mask, out=data_ft)
        data.data = be.to_numpy_array(be.irfftn(data_ft, s=data.shape).real)

    data = data.resample(args.sampling_rate, method="spline", order=3)

    if args.invert_contrast:
        data.data = data.data * -1

    data.data = data.data.astype(np.float32)
    data.to_file(args.output)


if __name__ == "__main__":
    main()
