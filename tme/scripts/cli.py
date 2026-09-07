#!python3
"""CLI entry point for PyTME.

Copyright (c) 2026 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import sys
import warnings
import argparse
import importlib


_SUBCOMMANDS = {
    "template": {
        "module": "tme.scripts.preprocess",
        "help": "Prepare templates for matching",
    },
    "match": {
        "module": "tme.scripts.match_template",
        "help": "Run template matching",
    },
    "postprocess": {
        "module": "tme.scripts.postprocess",
        "help": "Analyze matching results",
    },
    "batch": {
        "module": "tme.scripts.batch",
        "help": "Run batches of 'match' or 'postprocess' jobs",
    },
    "gui": {
        "module": "tme.scripts.gui",
        "help": "Interactive GUI",
    },
    "utils": {
        "module": "tme.scripts.utils",
        "help": "Auxiliary tools",
    },
}


def main(argv=None):
    from tme.__version__ import __version__

    parser = argparse.ArgumentParser(
        prog="pytme",
        description="PyTME - Python Template Matching Engine",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="subcommand")
    for name, info in _SUBCOMMANDS.items():
        subparsers.add_parser(name, help=info["help"], add_help=False)

    args, remaining = parser.parse_known_args(argv)

    if args.subcommand is None:
        parser.print_help()
        sys.exit(0)

    info = _SUBCOMMANDS[args.subcommand]
    sys.argv = [f"pytme {args.subcommand}"] + (remaining or ["--help"])

    try:
        module = importlib.import_module(info["module"])
    except ImportError as exc:
        if args.subcommand == "gui":
            print(
                f"pytme gui requires the [gui] extra: pip install pytme[gui]\n  ({exc})",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    module.main()


def _make_deprecated_main(legacy_name, subcommand_name, module_path):
    def deprecated_main():
        warnings.warn(
            f"'{legacy_name}' is deprecated. Use 'pytme {subcommand_name}' instead.",
            FutureWarning,
            stacklevel=2,
        )
        module = importlib.import_module(module_path)
        module.main()

    return deprecated_main


def preprocessor_gui_deprecated():
    warnings.warn(
        "'preprocessor_gui' is deprecated. Use 'pytme gui' instead.",
        FutureWarning,
        stacklevel=2,
    )
    main(["gui"])


match_template_deprecated = _make_deprecated_main(
    "match_template", "match", "tme.scripts.match_template"
)
estimate_memory_usage_deprecated = _make_deprecated_main(
    "estimate_memory_usage", "utils memory", "tme.scripts.estimate_memory_usage"
)
preprocess_deprecated = _make_deprecated_main(
    "preprocess", "template", "tme.scripts.preprocess"
)
postprocess_deprecated = _make_deprecated_main(
    "postprocess", "postprocess", "tme.scripts.postprocess"
)
pytme_runner_deprecated = _make_deprecated_main(
    "pytme_runner", "batch", "tme.scripts.pytme_runner"
)
