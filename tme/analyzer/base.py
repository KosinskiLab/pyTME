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
        state
            Initial state tuple containing the analyzer's internal data
            structures. The exact structure depends on the specific
            implementation.

        Notes
        -----
        This method creates the initial state that will be passed to
        subsequent calls to __call__. The state should contain all
        necessary data structures for accumulating analysis results.
        """

    @abstractmethod
    def __call__(self, state, scores, rotation_matrix, **kwargs) -> Tuple:
        """
        Update the analyzer state with new scoring data.

        Parameters
        ----------
        state : object
            Current analyzer state as returned by init_state() or
            previous calls to __call__.
        scores : BackendArray
            Array of scores computed for the current rotation.
        rotation_matrix : BackendArray
            Rotation matrix used to generate the scores.
        **kwargs : dict
            Additional keyword arguments specific to the analyzer
            implementation.

        Returns
        -------
        state
            Updated analyzer state with the new scoring data incorporated.

        Notes
        -----
        This method should be pure functional - it should not modify
        the input state but return a new state with the updates applied.
        The exact signature may vary between implementations.
        """
        pass

    @abstractmethod
    def result(self, state: Tuple, **kwargs) -> Tuple:
        """
        Finalize the analysis and produce the final result.

        Parameters
        ----------
        state : tuple
            Final analyzer state containing all accumulated data.
        **kwargs : dict
            Additional keyword arguments for result processing,
            such as postprocessing parameters.

        Returns
        -------
        result
            Final analysis result. The exact format depends on the
            analyzer implementation but typically includes processed
            scores, rotation information, and metadata.

        Notes
        -----
        This method converts the internal analyzer state into the
        final output format expected by the template matching pipeline.
        It may apply postprocessing operations like convolution mode
        correction or coordinate transformations.
        """
        pass

    @classmethod
    @abstractmethod
    def merge(cls, results: List[Tuple], **kwargs) -> Tuple:
        """
        Merge results from multiple analyzer instances.

        Parameters
        ----------
        results : list
            List of result objects as returned by the result() method
            from multiple analyzer instances.
        **kwargs : dict
            Additional keyword arguments for merge configuration.

        Returns
        -------
        merged_result
            Single result object combining all input results.

        Notes
        -----
        This method enables parallel processing by allowing results
        from different processes or splits to be combined into a
        unified result. The merge operation should handle overlapping
        data appropriately and maintain consistency.
        """
