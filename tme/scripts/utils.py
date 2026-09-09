#!python3
"""Dispatcher for pytme utils subcommands.

Copyright (c) 2026 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import sys
import argparse
import importlib


_UTILS_SUBCOMMANDS = {
    "mask": {
        "module": "tme.scripts.mask",
        "help": "Generate template masks.",
    },
    "memory": {
        "module": "tme.scripts.estimate_memory_usage",
        "help": "Estimate matching memory requirements.",
    },
    "evaluate": {
        "module": "tme.scripts.evaluate",
        "help": "Evaluate predicted picks against ground truth.",
    },
    "average": {
        "module": "tme.scripts.average",
        "help": "Create simple average from picks.",
    },
    "mesh": {
        "module": "tme.scripts.mesh",
        "help": "Build a surface mesh from a segmentation.",
    },
}


def main():
    parser = argparse.ArgumentParser(
        prog="pytme utils",
        description="Utility commands.",
    )
    subparsers = parser.add_subparsers(dest="util_command")
    for name, info in _UTILS_SUBCOMMANDS.items():
        subparsers.add_parser(name, help=info["help"], add_help=False)

    args, remaining = parser.parse_known_args()

    if args.util_command is None:
        parser.print_help()
        sys.exit(0)

    info = _UTILS_SUBCOMMANDS[args.util_command]
    module = importlib.import_module(info["module"])

    sys.argv = [f"pytme utils {args.util_command}"] + (remaining or ["--help"])
    module.main()
