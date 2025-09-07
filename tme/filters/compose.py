"""
Combine filters using an interface analogous to pytorch's Compose.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, Dict
from abc import ABC, abstractmethod

from ._utils import crop_real_fourier
from ..backends import backend as be

__all__ = ["Compose", "ComposableFilter"]


class ComposableFilter(ABC):
    """
    Base class for composable filters.

    This class provides a standard interface for filters used in template matching
    and reconstruction. It automatically handles:

    - Parameter merging between instance attributes and runtime arguments
    - Fourier space shifting when needed
    - Real Fourier transform cropping for efficiency
    - Standardized result dictionary formatting

    Subclasses need to implement :py:meth:`ComposableFilter._evaluate` which
    contains the core filter computation logic.

    By default, all filters are assumed to be multiplicative in Fourier space,
    which covers the vast majority of use cases (bandpass, CTF, wedge, whitening, etc.).
    Only explicitly specify non-multiplicative behavior when needed.
    """

    @abstractmethod
    def _evaluate(self, **kwargs) -> Dict:
        """
        Compute the actual filter given a set of keyword parameters.

        Parameters
        ----------
        **kwargs : dict
            Merged parameters from instance attributes and runtime arguments
            passed to :py:meth:`__call__`. This includes both the filter's
            configuration parameters and any runtime overrides.

        Returns
        -------
        Dict
            Dictionary containing the filter result and metadata. Required keys:

            - data : BackendArray or array-like
                The computed filter data
            - shape : tuple of int
                Input shape the filter was built for.

            Optional keys:
            - is_multiplicative_filter : bool
                Whether the filter is multiplicative in Fourier space (default True)
        """

    def __call__(self, return_real_fourier: bool = False, **kwargs) -> Dict:
        """
        This method provides the standard interface for creating of composable
        filter masks. It merges instance attributes with runtime parameters,
        and ensures Fourier conventions are consistent across filters.

        Parameters
        ----------
        return_real_fourier : bool, optional
            Whether to crop the filter mask for compatibility with real input
            FFTs (i.e., :py:func:`numpy.fft.rfft`). When True, only the
            positive frequency components are returned, reducing memory usage
            and computation time for real-valued inputs. Default is False.
        **kwargs : dict
            Additional keyword arguments passed to :py:meth:`_evaluate`.
            These will override any matching instance attributes during
            parameter merging.

        Returns
        -------
        Dict
            - data : BackendArray
                The processed filter data, converted to the appropriate backend
                array type and with fourier operations applied as needed
            - shape : tuple of int or None
                Shape for which the filter was created
            - return_real_fourier : bool
                The value of the return_real_fourier parameter
            - is_multiplicative_filter : bool
                Whether the filter is multiplicative in Fourier space
            - Additional metadata from the filter implementation
        """
        ret = self._evaluate(**(vars(self) | kwargs))

        # This parameter is only here to allow for using Composable filters outside
        # the context of a Compose operation. Internally, we require return_real_fourier
        # to be False, e.g., for filters that require reconstruction.
        if return_real_fourier:
            ret["data"] = crop_real_fourier(ret["data"])

        ret["data"] = be.to_backend_array(ret["data"])
        ret["return_real_fourier"] = return_real_fourier
        return ret


class Compose:
    """
    Compose a series of filters.

    Parameters
    ----------
    transforms : tuple of :py:class:`ComposableFilter`.
        Tuple of filter instances.
    """

    def __init__(self, transforms: Tuple[ComposableFilter, ...]):
        for transform in transforms:
            if not isinstance(transform, ComposableFilter):
                raise ValueError(f"{transform} is not a child of {ComposableFilter}.")

        self.transforms = transforms

    def __call__(self, return_real_fourier: bool = False, **kwargs) -> Dict:
        """
        Apply the sequence of filters in order, chaining their outputs.

        Parameters
        ----------
        return_real_fourier : bool, optional
            Whether to crop the filter mask for compatibility with real input
            FFTs (i.e., :py:func:`numpy.fft.rfft`). When True, only the
            positive frequency components are returned, reducing memory usage
            and computation time for real-valued inputs. Default is False.
        **kwargs : dict
            Keyword arguments passed to the first filter and propagated through
            the pipeline.

        Returns
        -------
        Dict
            Result dictionary from the final filter in the composition, containing:

            - data : BackendArray
                The final composite filter data. For multiplicative filters, this is
                the element-wise product of all individual filter outputs.
            - shape : tuple of int
                Shape of the filter data
            - return_real_fourier : bool
                Whether the output is compatible with real FFTs
            - Additional metadata from the filter pipeline
        """
        meta = {}
        if not len(self.transforms):
            return meta

        meta = self.transforms[0](**kwargs)
        for transform in self.transforms[1:]:
            kwargs.update(meta)
            ret = transform(**kwargs)

            if "data" not in ret:
                continue

            if ret.get("is_multiplicative_filter", True):
                prev_data = meta.pop("data")
                ret["data"] = be.multiply(ret["data"], prev_data)
                ret["merge"], prev_data = None, None
            meta = ret

        if return_real_fourier:
            meta["data"] = crop_real_fourier(meta["data"])
        return meta
