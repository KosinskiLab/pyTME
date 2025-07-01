"""
Implements abstract base class for template matching analyzers.

Copyright (c) 2025 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, List
from abc import ABC, abstractmethod

__all__ = ["AbstractAnalyzer"]


class AbstractAnalyzer(ABC):
    """
    Abstract base class for template matching analyzers.
    """

    @property
    def shareable(self):
        """
        Indicate whether the analyzer can be shared across processes.

        Returns
        -------
        bool
            True if the analyzer supports shared memory operations
            and can be safely used across multiple processes, False
            if it should only be used within a single process.
        """
        return False

    @abstractmethod
    def init_state(self, *args, **kwargs) -> Tuple:
        """
        Initialize the analyzer state.

        Returns
        -------
        state : tuple
            Initial state tuple of the analyzer instance. The exact structure
            depends on the specific implementation.

        Notes
        -----
        This method creates the initial state that will be passed to
        :py:meth:`AbstractAnalyzer.__call__` and finally to
        :py:meth:`AbstractAnalyzer.result`. The state should contain all necessary
        data structures for accumulating analysis results.
        """

    @abstractmethod
    def __call__(self, state, scores, rotation_matrix, **kwargs) -> Tuple:
        """
        Update the analyzer state with new scoring data.

        Parameters
        ----------
        state : tuple
            Current analyzer state as returned :py:meth:`AbstractAnalyzer.init_state`
            or previous invocations of :py:meth:`AbstractAnalyzer.__call__`.
        scores : BackendArray
            Array of new scores with dimensionality d.
        rotation_matrix : BackendArray
            Rotation matrix used to generate scores with shape (d,d).
        **kwargs : dict
            Keyword arguments used by specific implementations.

        Returns
        -------
        tuple
            Updated analyzer state incorporating the new data.
        """

    @abstractmethod
    def result(self, state: Tuple, **kwargs) -> Tuple:
        """
        Finalize the analysis by performing potential post processing.

        Parameters
        ----------
        state : tuple
            Analyzer state containing accumulated data.
        **kwargs : dict
            Keyword arguments used by specific implementations.

        Returns
        -------
        result
            Final analysis result. The exact struccture depends on the
            analyzer implementation.

        Notes
        -----
        This method converts the internal analyzer state into the
        final output format expected by the template matching pipeline.
        It may apply postprocessing operations like convolution mode
        correction or coordinate transformations.
        """

    @classmethod
    @abstractmethod
    def merge(cls, results: List[Tuple], **kwargs) -> Tuple:
        """
        Merge multiple analyzer results.

        Parameters
        ----------
        results : list of tuple
            List of tuple objects returned by :py:meth:`AbstractAnalyzer.result`
            from different instances of the same analyzer class.
        **kwargs : dict
            Keyword arguments used by specific implementations.

        Returns
        -------
        tuple
            Single result object combining all input results.

        Notes
        -----
        This method enables parallel processing by allowing results
        from different processes or splits to be combined into a
        unified result. The merge operation should handle overlapping
        data appropriately and maintain consistency.
        """
