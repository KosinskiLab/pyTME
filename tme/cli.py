#!python3
"""
CLI utility functions.

Copyright (c) 2025 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import argparse

import numpy as np
from . import __version__
from .types import BackendArray


def match_template(
    target: BackendArray,
    template: BackendArray,
    template_mask: BackendArray = None,
    score="FLCSphericalMask",
    rotations=None,
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
    from .matching_data import MatchingData
    from .analyzer import MaxScoreOverRotations
    from .matching_exhaustive import match_exhaustive, MATCHING_EXHAUSTIVE_REGISTER

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


def print_entry() -> None:
    width = 80
    text = f" pytme v{__version__} "
    padding_total = width - len(text) - 2
    padding_left = padding_total // 2
    padding_right = padding_total - padding_left

    print("*" * width)
    print(f"*{ ' ' * padding_left }{text}{ ' ' * padding_right }*")
    print("*" * width)


def get_func_fullname(func) -> str:
    """Returns the full name of the given function, including its module."""
    return f"<function '{func.__module__}.{func.__name__}'>"


def print_block(name: str, data: dict, label_width=20) -> None:
    """Prints a formatted block of information."""
    print(f"\n> {name}")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            value = value.shape
        formatted_value = str(value)
        print(f"  - {str(key) + ':':<{label_width}} {formatted_value}")


def check_positive(value):
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive float." % value)
    return ivalue
