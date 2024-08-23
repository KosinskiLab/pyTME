""" Defines a specification for filters that can be used with
    :py:class:`tme.preprocessing.compose.Compose`.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
from typing import Dict
from abc import ABC, abstractmethod


class ComposableFilter(ABC):
    """
    Strategy class for composable filters.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Dict:
        """

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        Dict
            A dictionary representing the result of the filtering operation.
        """
