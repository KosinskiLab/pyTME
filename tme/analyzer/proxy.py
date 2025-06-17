"""
Implements SharedAnalyzerProxy to managed shared memory of Analyzer instances
across different tasks.

This is primarily useful for CPU template matching, where parallelization can
be performed over rotations, rather than subsections of a large input volume.

Copyright (c) 2025 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple
from multiprocessing import Manager
from multiprocessing.shared_memory import SharedMemory

from ..backends import backend as be

__all__ = ["StatelessSharedAnalyzerProxy", "SharedAnalyzerProxy"]


class StatelessSharedAnalyzerProxy:
    """
    Proxy that wraps functional analyzers for concurrent access via shared memory.

    Enables multiple processes/threads to safely update the same analyzer
    while preserving the functional interface of the underlying analyzer.
    """

    def __init__(self, analyzer_class: type, analyzer_params: dict):
        self._shared = False
        self._process = self._direct_call

        self._analyzer = analyzer_class(**analyzer_params)

    def __call__(self, state, *args, **kwargs):
        return self._process(state, *args, **kwargs)

    def init_state(self, shm_handler=None, *args, **kwargs) -> Tuple:
        state = self._analyzer.init_state()
        if shm_handler is not None:
            self._shared = True
            state = self._to_shared(state, shm_handler)

            self._lock = Manager().Lock()
            self._process = self._thread_safe_call
        return state

    def _to_shared(self, state: Tuple, shm_handler):
        backend_arr = type(be.zeros((1), dtype=be._float_dtype))

        ret = []
        for v in state:
            if isinstance(v, backend_arr):
                v = be.to_sharedarr(v, shm_handler)
            elif isinstance(v, dict):
                v = Manager().dict(**v)
            ret.append(v)
        return tuple(ret)

    def _shared_to_object(self, shared: type):
        if not self._shared:
            return shared
        if isinstance(shared, tuple) and len(shared):
            if isinstance(shared[0], SharedMemory):
                return be.from_sharedarr(shared)
        return shared

    def _thread_safe_call(self, state, *args, **kwargs):
        """Thread-safe call to analyzer"""
        with self._lock:
            state = tuple(self._shared_to_object(x) for x in state)
            return self._direct_call(state, *args, **kwargs)

    def _direct_call(self, state, *args, **kwargs):
        """Direct call to analyzer without locking"""
        return self._analyzer(state, *args, **kwargs)

    def result(self, state, **kwargs):
        """Extract final result"""
        final_state = state
        if self._shared:
            # Convert shared arrays back to regular arrays and copy to
            # avoid array invalidation by shared memory handler
            final_state = tuple(self._shared_to_object(x) for x in final_state)
        return self._analyzer.result(final_state, **kwargs)

    def merge(self, *args, **kwargs):
        return self._analyzer.merge(*args, **kwargs)


class SharedAnalyzerProxy(StatelessSharedAnalyzerProxy):
    """
    Child of :py:class:`StatelessSharedAnalyzerProxy` that is aware
    of the current analyzer state to emulate the previous analyzer interface.
    """

    def __init__(
        self,
        analyzer_class: type,
        analyzer_params: dict,
        shm_handler: type = None,
        **kwargs,
    ):
        super().__init__(
            analyzer_class=analyzer_class,
            analyzer_params=analyzer_params,
        )
        if not self._analyzer.shareable:
            shm_handler = None
        self.init_state(shm_handler)

    def init_state(self, shm_handler=None, *args, **kwargs) -> Tuple:
        self._state = super().init_state(shm_handler, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        state = super().__call__(self._state, *args, **kwargs)
        if not self._shared:
            self._state = state

    def result(self, **kwargs):
        """Extract final result"""
        return super().result(self._state, **kwargs)
