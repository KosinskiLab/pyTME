#!python3
"""Generate masks from template densities.

Copyright (c) 2026 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import os
import argparse
from itertools import product

import numpy as np

from tme import Density
from tme.matching_utils import create_mask


def _parse_radius(value):
    """Parse a radius string into a scalar or tuple of floats.

    Comma-separated values define per-axis radii (e.g. '5,8,10').
    A single number gives an isotropic radius.
    """
    parts = value.split(",")
    floats = tuple(float(p) for p in parts)
    return floats[0] if len(floats) == 1 else floats


def _parse_sweep(value):
    """Parse a single sweep token: 'start:stop:step' range or scalar."""
    if ":" in value:
        parts = value.split(":")
        start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
        return list(np.arange(start, stop + step / 2, step))
    return [float(value)]


def main():
    parser = argparse.ArgumentParser(
        prog="pytme utils mask",
        description="Generate template mask (libraries).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sweep parameters accept ranges (6:12:2) or multiple space-\n"
            "separated values (6 8 10). When multiple values are given the\n"
            "cartesian product is written to --output-prefix.\n"
            "\n"
            "Per-axis radii use commas: --radius 5,8,10 gives one mask with\n"
            "radii (5, 8, 10). To sweep radii: --radius 5 8 10 gives three\n"
            "isotropic masks.\n"
            "\n"
            "If --threshold is given, a threshold mask is created regardless\n"
            "of --shape.\n"
            "\n"
            "Examples:\n"
            "  pytme utils mask --template t.mrc --radius 8 -o out\n"
            "  pytme utils mask --template t.mrc --radius 6 8 10 \\\n"
            "      --soft-edge-width 0 2 -o masks/screen\n"
        ),
    )

    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument(
        "--template",
        required=True,
        help="Template density file (provides shape and sampling rate).",
    )
    io_group.add_argument(
        "-o",
        "--output-prefix",
        type=str,
        required=True,
        help="Output path prefix. Single masks are written as "
        "<prefix>.mrc, libraries as <prefix>_<shape>_<params>.mrc.",
    )
    io_group.add_argument(
        "--threshold",
        type=str,
        nargs="+",
        default=None,
        help="Binarize template at this density value. Presence of this "
        "flag selects threshold mode.",
    )

    geom_group = parser.add_argument_group(
        "Geometric masks (ellipse, box, tube, membrane)"
    )
    geom_group.add_argument(
        "--shape",
        type=str,
        default="ellipse",
        choices=["ellipse", "box", "tube", "membrane"],
        help="Geometric mask shape (default: ellipse).",
    )
    geom_group.add_argument(
        "--radius",
        type=str,
        nargs="+",
        default=None,
        help="Radius in voxels. Use commas for per-axis radii (5,8,10). "
        "Multiple space-separated values sweep over radii.",
    )
    geom_group.add_argument(
        "--inner-radius",
        type=str,
        nargs="+",
        default=None,
        help="Inner radius for tube masks (default: 0).",
    )
    geom_group.add_argument(
        "--height",
        type=str,
        nargs="+",
        default=None,
        help="Height in voxels for tube masks.",
    )
    geom_group.add_argument(
        "--thickness",
        type=str,
        nargs="+",
        default=None,
        help="Leaflet thickness for membrane masks.",
    )
    geom_group.add_argument(
        "--separation",
        type=str,
        nargs="+",
        default=None,
        help="Leaflet separation for membrane masks.",
    )
    geom_group.add_argument(
        "--symmetry-axis",
        type=int,
        default=2,
        help="Symmetry axis for tube/membrane masks (default: 2).",
    )
    geom_group.add_argument(
        "--center",
        type=str,
        nargs="+",
        default=None,
        help="Mask center in voxels as comma-separated per-axis values "
        "(e.g. 23.5,29,29) in the same axis order as the template. "
        "Defaults to the center of the template box.",
    )

    edge_group = parser.add_argument_group("Soft edge")
    edge_group.add_argument(
        "--soft-edge-width",
        type=str,
        nargs="+",
        default=["0"],
        help="Cosine soft-edge width in voxels (default: 0, hard edge). "
        "Width equals the full extent of the falloff.",
    )
    edge_group.add_argument(
        "--extend",
        type=str,
        nargs="+",
        default=["0"],
        help="Isotropic dilation in voxels before soft edge (default: 0).",
    )

    args = parser.parse_args()

    template = Density.from_file(args.template)
    is_threshold = args.threshold is not None

    # Build sweep grid.
    sweep = {}

    # Soft edge params.
    for attr in ("soft_edge_width", "extend"):
        raw = getattr(args, attr)
        values = []
        for v in raw:
            values.extend(_parse_sweep(v))
        sweep[attr] = values

    if is_threshold:
        values = []
        for v in args.threshold:
            values.extend(_parse_sweep(v))
        sweep["threshold"] = values
    else:
        if args.radius is not None:
            sweep["radius"] = [_parse_radius(v) for v in args.radius]
        if args.center is not None:
            sweep["center"] = [_parse_radius(v) for v in args.center]
        for attr in ("inner_radius", "height", "thickness", "separation"):
            raw = getattr(args, attr, None)
            if raw is None:
                continue
            values = []
            for v in raw:
                values.extend(_parse_sweep(v))
            sweep[attr] = values

    keys = list(sweep.keys())
    configs = list(product(*sweep.values())) if keys else [()]

    prefix_dir = os.path.dirname(args.output_prefix)
    if prefix_dir:
        os.makedirs(prefix_dir, exist_ok=True)

    mask_type = "threshold" if is_threshold else args.shape
    is_single = len(configs) == 1

    masks_written = 0
    for values in configs:
        params = dict(zip(keys, values))

        for k in params:
            if k != "radius":
                try:
                    params[k] = float(params[k])
                except (TypeError, ValueError):
                    pass

        params.setdefault("shape", template.data.shape)
        params.setdefault("center", tuple(s // 2 for s in template.data.shape))
        params.setdefault("size", template.data.shape)

        if is_threshold:
            params["data"] = template.data

        if mask_type in ("tube", "membrane"):
            params.setdefault("symmetry_axis", args.symmetry_axis)

        if mask_type == "tube" and "radius" in params:
            params.setdefault("outer_radius", params.pop("radius"))
            params.setdefault("inner_radius", 0)

        try:
            mask_data = create_mask(mask_type=mask_type, method="cosine", **params)
        except (TypeError, ValueError) as exc:
            hint_keys = {
                "radius",
                "threshold",
                "thickness",
                "separation",
                "inner_radius",
                "height",
            }
            missing = [f"--{k.replace('_', '-')}" for k in hint_keys if k not in params]
            if missing:
                parser.error(f"{exc}\n\nPerhaps specify one of: {', '.join(missing)}")
            raise

        if is_single:
            path = f"{args.output_prefix}.mrc"
        else:
            _short = {
                "radius": "r",
                "soft_edge_width": "se",
                "threshold": "t",
                "extend": "e",
                "thickness": "th",
                "separation": "sep",
                "inner_radius": "ir",
                "height": "h",
                "center": "c",
            }
            tag = "_".join(f"{_short.get(k, k)}{v}" for k, v in zip(keys, values))
            path = f"{args.output_prefix}_{mask_type}_{tag}.mrc"

        out_density = Density(
            data=mask_data,
            origin=template.origin,
            sampling_rate=template.sampling_rate,
        )
        out_density.to_file(path)
        masks_written += 1

    print(f"Wrote {masks_written} mask(s).")
