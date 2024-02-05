""" pyTME backend manager.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Dict, List

from .matching_backend import MatchingBackend
from .npfftw_backend import NumpyFFTWBackend
from .pytorch_backend import PytorchBackend
from .cupy_backend import CupyBackend
from .mlx_backend import MLXBackend


class BackendManager:
    """
    Manager for template matching backends.

    This class serves as an interface to various computational backends (e.g.,
    CPU, GPU). It allows users to seamlessly swap between different backend
    implementations while preserving the consistency and functionality of the API.
    Direct attribute and method calls to the manager are delegated to the current
    active backend.

    Attributes
    ----------
    _BACKEND_REGISTRY : dict
        A dictionary mapping backend names to their respective classes or instances.
    _backend : instance of MatchingBackend
        An instance of the currently active backend.
    _backend_name : str
        Name of the current backend.
    _backend_args : Dict
        Arguments passed to create current backend.

    Examples
    --------
    >>> from tme.backends import backend
    >>> backend.multiply(arr1, arr2)
    # This will use the default NumpyFFTWBackend's multiply method

    >>> backend.change_backend("pytorch")
    >>> backend.multiply(arr1, arr2)
    # This will use the GPUBackend's multiply method

    Notes
    -----
    To add custom backends, use the `add_backend` method. To switch
    between backends, use the `change_backend` method. Note that the backend
    has to be reinitialzed when using fork-based parallelism.
    """

    def __init__(self):
        self._BACKEND_REGISTRY = {
            "cpu_backend": NumpyFFTWBackend,
            "pytorch": PytorchBackend,
            "cupy": CupyBackend,
        }
        self._backend = NumpyFFTWBackend()
        self._backend_name = "cpu_backend"
        self._backend_args = {}

    def __repr__(self):
        return f"<BackendManager: using {self._backend_name}>"

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def __dir__(self) -> List:
        """
        Return a list of attributes available in this object,
        including those from the backend.

        Returns
        -------
        list
            Sorted list of attributes.
        """
        base_attributes = []
        base_attributes.extend(dir(self.__class__))
        base_attributes.extend(self.__dict__.keys())
        base_attributes.extend(dir(self._backend))
        return sorted(base_attributes)

    def add_backend(self, backend_name: str, backend_class: type):
        """
        Adds a custom backend to the registry.

        Parameters
        ----------
        backend_name : str
            Name by which the backend can be referenced.
        backend_class : :py:class:`MatchingBackend`
            An instance of the backend to be added.

        Raises
        ------
        ValueError
            If the provided backend_instance does not inherit from MatchingBackend.
        """
        if not issubclass(backend_class, MatchingBackend):
            raise ValueError("backend_class needs to inherit from MatchingBackend.")
        self._BACKEND_REGISTRY[backend_name] = backend_class

    def change_backend(self, backend_name: str, **backend_kwargs: Dict) -> None:
        """
        Change the backend.

        Parameters
        ----------
        backend_name : str
            Name of the new backend that should be used.
        **backend_kwargs : Dict, optional
            Parameters passed to __init__ method of backend.

        Raises
        ------
        NotImplementedError
            If no backend is found with the provided name.
        """
        if backend_name not in self._BACKEND_REGISTRY:
            available_backends = ", ".join(
                [str(x) for x in self._BACKEND_REGISTRY.keys()]
            )
            raise NotImplementedError(
                f"Available backends are {available_backends} - not {backend_name}."
            )
        self._backend = self._BACKEND_REGISTRY[backend_name](**backend_kwargs)
        self._backend_name = backend_name
        self._backend_args = backend_kwargs


backend = BackendManager()
