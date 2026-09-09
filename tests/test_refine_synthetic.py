"""
Synthetic validation of refine_matches.py

Embed a template with a certain target filter manipulation together
with the original template without modification (negative particle).
By identifying the correct filter combination, the two populations
become discriminable via their template matching scores.
"""

import os
import sys
import tempfile
import subprocess

import numpy as np
import yaml

from tme import Density
from tme.filters import BandPassReconstructed


def create_sphere(shape, radius, center=None):
    if center is None:
        center = tuple(s // 2 for s in shape)
    coords = np.mgrid[tuple(slice(0, s) for s in shape)]
    dist = sum((c - cn) ** 2 for c, cn in zip(coords, center))
    return (dist <= radius**2).astype(np.float32)


def apply_bandpass(data, lowpass, highpass, sampling_rate):
    bpf = BandPassReconstructed(
        lowpass=lowpass, highpass=highpass, sampling_rate=sampling_rate
    )
    result = bpf(shape=data.shape)
    mask = np.asarray(result["data"])
    ft = np.fft.fftn(data)
    return np.real(np.fft.ifftn(ft * mask)).astype(np.float32)


def write_star_file(filepath, positions, labels, tomo_path):
    with open(filepath, "w") as f:
        f.write("data_particles\n\nloop_\n")
        for col in (
            "_rlnCoordinateX",
            "_rlnCoordinateY",
            "_rlnCoordinateZ",
            "_rlnAngleRot",
            "_rlnAngleTilt",
            "_rlnAnglePsi",
            "_rlnClassNumber",
            "_rlnMicrographName",
            "_pytmeScore",
        ):
            f.write(f"{col}\n")
        for pos, label in zip(positions, labels):
            f.write(
                f"{pos[0]}\t{pos[1]}\t{pos[2]}\t0\t0\t0\t" f"{label}\t{tomo_path}\t0\n"
            )


def main():
    SAMPLING_RATE = 10.0
    TRUE_LOWPASS = 60.0
    TRUE_HIGHPASS = 300.0

    TEMPLATE_RADIUS = 5
    TEMPLATE_SIZE = 14
    TOMO_SIZE = 100

    rng = np.random.default_rng(0)

    positive_positions = np.array(
        [
            [25, 25, 25],
            [25, 25, 70],
            [25, 70, 25],
            [25, 70, 70],
            [70, 25, 25],
            [70, 25, 70],
            [70, 70, 25],
            [70, 70, 70],
        ]
    )
    negative_positions = np.array(
        [
            [48, 25, 48],
            [48, 70, 48],
            [25, 48, 48],
            [70, 48, 48],
            [48, 48, 25],
            [48, 48, 70],
            [48, 25, 25],
            [48, 70, 70],
        ]
    )

    all_positions = np.vstack([positive_positions, negative_positions])
    labels = np.array([1] * 8 + [0] * 8)

    print("=== Synthetic Refine Matches Test ===")
    print(f"Ground truth:   lowpass={TRUE_LOWPASS} Å, highpass={TRUE_HIGHPASS} Å")
    print(f"Sampling rate:  {SAMPLING_RATE} Å/voxel")
    print(f"Template:       sphere radius {TEMPLATE_RADIUS}")
    print(
        f"Particles:      {len(positive_positions)} pos, {len(negative_positions)} neg\n"
    )

    template_shape = (TEMPLATE_SIZE,) * 3
    template = create_sphere(template_shape, TEMPLATE_RADIUS)

    tomo = np.zeros((TOMO_SIZE,) * 3, dtype=np.float32)
    half = TEMPLATE_SIZE // 2

    template_bp = apply_bandpass(template, TRUE_LOWPASS, TRUE_HIGHPASS, SAMPLING_RATE)
    for pos in positive_positions:
        slc = tuple(slice(int(p - half), int(p - half + TEMPLATE_SIZE)) for p in pos)
        tomo[slc] += template_bp

    confused_bp = apply_bandpass(template, 30, 400, SAMPLING_RATE)
    for pos in negative_positions:
        slc = tuple(slice(int(p - half), int(p - half + TEMPLATE_SIZE)) for p in pos)
        tomo[slc] += confused_bp

    noise = rng.normal(0, 0.5, tomo.shape).astype(np.float32)
    tomo_noisy = tomo + noise

    print(f"Tomogram shape: {tomo_noisy.shape}")
    print(f"Signal range:   [{tomo.min():.3f}, {tomo.max():.3f}]")
    print("Noise std:      0.5\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        template_path = os.path.join(tmpdir, "template.mrc")
        tomo_path = os.path.join(tmpdir, "tomogram.mrc")
        star_path = os.path.join(tmpdir, "particles.star")
        output_path = os.path.join(tmpdir, "best_params.yaml")

        Density(
            data=template,
            origin=np.zeros(3),
            sampling_rate=np.full(3, SAMPLING_RATE),
        ).to_file(template_path)

        Density(
            data=tomo_noisy,
            origin=np.zeros(3),
            sampling_rate=np.full(3, SAMPLING_RATE),
        ).to_file(tomo_path)

        write_star_file(star_path, all_positions, labels, tomo_path)

        cmd = [
            sys.executable,
            "-m",
            "tme.scripts.refine_matches",
            "--template",
            template_path,
            "--orientations",
            star_path,
            "--angular-sampling",
            "360",
            "--lowpass-range",
            "15,60",
            "--highpass-range",
            "80,400",
            "--translation-uncertainty",
            "3",
            "--margin",
            "4",
            "--maxiter",
            "5",
            "--processes",
            "1",
            "--objective",
            "pairwise_logistic",
            "-s",
            "FLCSphericalMask",
            "--no-filter-target",
            "--background-correction",
            "phase-scrambling",
            "--output",
            output_path,
        ]

        print("Running optimizer …")
        print(f"  {' '.join(cmd)}\n")

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            sys.exit(f"Script failed with return code {ret.returncode}")

        with open(output_path) as f:
            params = yaml.safe_load(f)

        lp_err = abs(params["lowpass"] - TRUE_LOWPASS)
        hp_err = abs(params["highpass"] - TRUE_HIGHPASS)

        print("=== Results ===")
        print(
            f"Ground truth:   lowpass={TRUE_LOWPASS:.1f}  highpass={TRUE_HIGHPASS:.1f}"
        )
        print(
            f"Recovered:      lowpass={params['lowpass']:.1f}  highpass={params['highpass']:.1f}"
        )
        print(f"Errors:         Δlp={lp_err:.1f} Å   Δhp={hp_err:.1f} Å")
        print(f"F1 score:       {params['f1_score']:.4f}")
        print(f"Threshold:      {params['threshold']:.4f}")


if __name__ == "__main__":
    main()
