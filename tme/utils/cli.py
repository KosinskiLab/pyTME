#!python3
"""
CLI utility functions.

Copyright (c) 2025 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import re
import argparse
import warnings
from typing import Optional, Set
from pathlib import Path
from os.path import exists, abspath
from dataclasses import dataclass, field

from ..types import BackendArray
from ..__version__ import __version__

__all__ = [
    "match_template",
    "sanitize_name",
    "print_entry",
    "get_func_fullname",
    "print_block",
    "check_positive",
    "check_bounded_dtype",
    "check_symmetry",
    "existing_file",
    "load_and_validate_mask",
    "strip_result_extensions",
    "DeprecatedAction",
    "ArgumentMetadata",
]

RESULT_EXTENSIONS = (".pickle.gz", ".pickle", ".hdf5", ".h5")


def strip_result_extensions(filename: str) -> str:
    """Return the basename of a result file without its format extension."""
    name = Path(filename).name
    for ext in RESULT_EXTENSIONS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


@dataclass
class ArgumentMetadata:
    """Tracks argument destinations and flags added to a parser.

    Attributes
    ----------
    args : Set[str]
        Argument destination names (with underscores).
    flags : Set[str]
        Flag names in CLI format (with hyphens).
    """

    args: Set[str] = field(default_factory=set)
    flags: Set[str] = field(default_factory=set)

    def update(self, argparse_group):
        for action in argparse_group._group_actions:
            if action.dest in (argparse.SUPPRESS, "help"):
                continue
            self.args.add(action.dest)

            # Capture argparse._StoreTrueAction, argparse._StoreFalseAction
            is_flag = action.nargs == 0 or action.const is not None
            if action.option_strings and is_flag:
                self.flags.update(action.option_strings)


def match_template(
    target: BackendArray,
    template: BackendArray,
    template_mask: BackendArray = None,
    score="FLCSphericalMask",
    rotations=None,
    target_filter=None,
    template_filter=None,
):
    """
    Simple template matching run.

    Parameters
    ----------
    target : BackendArray
        Target array.
    template : BackendArray
        Template to be matched against target.
    template_mask : BackendArray, optional
        Template mask for normalization, defaults to None.
    score : str, optional
        Scoring method to use, defaults to 'FLCSphericalMask'.
    rotations: BackendArray, optional
        Rotation matrices with shape (n, d, d), where d is the dimension
        of the target. Defaults to the identity rotation matrix.

    Returns
    -------
    tuple
        scores : BackendArray
            Computed cross-correlation scores.
        offset : BackendArray
            Offset in target, defaults to 0.
        rotations : BackendArray
            Map between translations and rotation indices
        rotation_mapping : dict
            Map between rotation indices and rotation matrices
    """
    import numpy as np

    from ..matching_data import MatchingData
    from ..analyzer import MaxScoreOverRotations
    from ..matching_exhaustive import match_exhaustive, MATCHING_EXHAUSTIVE_REGISTER

    if rotations is None:
        rotations = np.eye(target.ndim).reshape(1, target.ndim, target.ndim)

    if rotations.shape[-1] != target.ndim:
        print(
            f"Dimension of rotation matrix {rotations.shape[-1]} does not "
            "match target dimension."
        )

    matching_data = MatchingData(
        target=target,
        template=template,
        template_mask=template_mask,
        rotations=rotations,
    )
    matching_data.template_mask = template_mask
    matching_setup, matching_score = MATCHING_EXHAUSTIVE_REGISTER[score]

    if target_filter is not None:
        matching_data.target_filter = target_filter

    if template_filter is not None:
        matching_data.template_filter = template_filter

    candidates = list(
        match_exhaustive(
            matching_data=matching_data,
            matching_score=matching_score,
            matching_setup=matching_setup,
            callback_class=MaxScoreOverRotations,
            callback_class_args={
                "score_threshold": -1,
            },
            pad_target_edges=True,
            job_schedule=(1, 1),
        )
    )
    return candidates


def sanitize_name(name: str):
    return name.title().replace("_", " ").replace("-", " ")


def _emit(message: str, logger=None) -> None:
    """Route a message through a logger if given, else to stdout."""
    if logger is None:
        print(message)
    else:
        logger.info(message)


def print_entry(logger=None) -> None:
    width = 80
    text = f" pytme v{__version__} "
    padding_total = width - len(text) - 2
    padding_left = padding_total // 2
    padding_right = padding_total - padding_left

    lines = [
        "*" * width,
        f"*{ ' ' * padding_left }{text}{ ' ' * padding_right }*",
        "*" * width,
    ]
    _emit("\n".join(lines), logger)


def get_func_fullname(func) -> str:
    """Returns the full name of the given function, including its module."""
    return f"<function '{func.__module__}.{func.__name__}'>"


def print_block(name: str, data: dict, label_width=20, logger=None) -> None:
    """Format a block of information, routed via logger if given, else stdout."""
    import numpy as np

    lines = [f"\n> {name}"]
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            value = value.shape
        lines.append(f"  - {str(key) + ':':<{label_width}} {str(value)}")
    _emit("\n".join(lines), logger)


def check_positive(value):
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive float." % value)
    return ivalue


def check_bounded_dtype(dtype, min_val=None, max_val=None):
    min_val = dtype(min_val) if min_val is not None else min_val
    max_val = dtype(max_val) if max_val is not None else max_val

    def validator(value):
        fvalue = dtype(value)

        if min_val is not None and max_val is not None:
            if not min_val <= fvalue <= max_val:
                raise argparse.ArgumentTypeError(
                    f"Value must be between {min_val} and {max_val}, got {fvalue}"
                )
        elif min_val is not None and fvalue < min_val:
            raise argparse.ArgumentTypeError(
                f"Value must be >= {min_val}, got {fvalue}"
            )
        elif max_val is not None and fvalue > max_val:
            raise argparse.ArgumentTypeError(
                f"Value must be <= {max_val}, got {fvalue}"
            )

        return fvalue

    return validator


def check_symmetry(value: str) -> str:
    """Validate a point-group symmetry string of the form C<n> or D<n>."""
    if not re.fullmatch(r"[CDcd][1-9][0-9]*", value or ""):
        raise argparse.ArgumentTypeError(
            f"Invalid symmetry '{value}'. Use the form C<n> or D<n>, e.g. C4 or D2."
        )
    return value.upper()


def existing_file(path: str) -> str:
    """Validate that file exists."""
    if not exists(path):
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return abspath(path)


def load_and_validate_mask(
    mask_target: "Density", mask_path: Optional[str], **kwargs
) -> Optional["Density"]:
    """
    Load and validate a mask against a target density.

    Parameters
    ----------
    mask_target : Density
        Target density the mask will be applied to.
    mask_path : str, optional
        Path to mask file.
    **kwargs
        Keyword arguments passed to Density.from_file.

    Returns
    -------
    Density or None
        Loaded mask with validated shape and sampling rate, or None if no path provided.

    Raises
    ------
    ValueError
        If shape does not match between mask and target.
    """
    from .. import Density
    import numpy as np

    if mask_path is None:
        return None

    mask = Density.from_file(mask_path, **kwargs)
    mask.origin = np.asarray(mask_target.origin).copy()

    if not np.allclose(mask.shape, mask_target.shape):
        raise ValueError(
            f"Mask shape mismatch: expected {mask_target.shape}, "
            f"got {mask.shape} in {mask_path}"
        )

    if not np.allclose(
        np.round(mask.sampling_rate, 2), np.round(mask_target.sampling_rate, 2)
    ):
        warnings.warn(
            f"Mask sampling rate mismatch: expected {mask_target.sampling_rate}, "
            f"got {mask.sampling_rate} in {mask_path}"
        )

    return mask


class DeprecatedAction(argparse.Action):
    def __init__(
        self, option_strings, dest, nargs=None, help=None, replacement=None, **kwargs
    ):
        kwargs["default"] = None
        self.replacement = replacement
        super().__init__(option_strings, dest, nargs=nargs, help=help, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        message = f"{option_string} is deprecated and ignored."
        if self.replacement:
            message += f" {self.replacement}"
        warnings.warn(message, FutureWarning)
        setattr(namespace, self.dest, None)
