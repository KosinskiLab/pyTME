import pytest
import numpy as np

from tme.backends import backend as be
from tme.analyzer import (
    MaxScoreOverRotations,
    PeakCaller,
    PeakCallerSort,
    PeakCallerMaximumFilter,
    PeakCallerFast,
    PeakCallerRecursiveMasking,
    PeakCallerScipy,
    PeakClustering,
)


PEAK_CALLER_CHILDREN = [
    PeakCallerSort,
    PeakCallerMaximumFilter,
    PeakCallerFast,
    PeakCallerRecursiveMasking,
    PeakCallerScipy,
    PeakClustering,
]
np.random.seed(123)


class TestPeakCallers:
    def setup_method(self):
        self.num_peaks = 100
        self.min_distance = 5
        self.data = np.random.rand(100, 100, 100)
        self.rotation_matrix = np.eye(3)

    @pytest.mark.parametrize("peak_caller", PEAK_CALLER_CHILDREN)
    def test_initialization(self, peak_caller):
        _ = peak_caller(shape=self.data.shape, num_peaks=100, min_distance=5)

    def test_initialization_error(self):
        with pytest.raises(TypeError):
            _ = PeakCaller(shape=self.data.shape, num_peaks=100, min_distance=5)

    @pytest.mark.parametrize("peak_caller", PEAK_CALLER_CHILDREN)
    def test_initialization_error_parameter(self, peak_caller):
        with pytest.raises(ValueError):
            _ = peak_caller(shape=self.data.shape, num_peaks=0, min_distance=5)
        with pytest.raises(ValueError):
            _ = peak_caller(shape=self.data.shape, num_peaks=-1, min_distance=5)
        with pytest.raises(ValueError):
            _ = peak_caller(shape=self.data.shape, num_peaks=-1, min_distance=-1)

    @pytest.mark.parametrize("peak_caller", PEAK_CALLER_CHILDREN)
    @pytest.mark.parametrize("num_peaks", (1, 100))
    @pytest.mark.parametrize("minimum_score", (None, 0.5))
    def test__call__(self, peak_caller, num_peaks, minimum_score):
        kwargs = {
            "shape": self.data.shape,
            "num_peaks": num_peaks,
            "min_distance": self.min_distance,
            "min_score": minimum_score,
        }
        peak_caller = peak_caller(**kwargs)
        state = peak_caller(
            peak_caller.init_state(),
            self.data.copy(),
            rotation_matrix=self.rotation_matrix,
        )
        state = peak_caller.result(state)
        if minimum_score is None:
            assert len(state[0] <= num_peaks)
        else:
            peaks = state[0].astype(int)
            print(self.data[tuple(peaks.T)])
            assert np.all(self.data[tuple(peaks.T)] >= minimum_score)

    @pytest.mark.parametrize("peak_caller", PEAK_CALLER_CHILDREN)
    @pytest.mark.parametrize("num_peaks", (1, 100))
    def test_merge(self, peak_caller, num_peaks):
        peak_caller1 = peak_caller(
            shape=self.data.shape, num_peaks=num_peaks, min_distance=self.min_distance
        )
        state1 = peak_caller1.init_state()
        state1 = peak_caller1(state1, self.data, rotation_matrix=self.rotation_matrix)

        peak_caller2 = peak_caller(
            shape=self.data.shape, num_peaks=num_peaks, min_distance=self.min_distance
        )
        state2 = peak_caller2.init_state()
        state2 = peak_caller2(state2, self.data, rotation_matrix=self.rotation_matrix)

        states = [peak_caller1.result(state1), peak_caller2.result(state2)]
        result = peak_caller.merge(
            results=states,
            num_peaks=num_peaks,
            min_distance=self.min_distance,
        )
        assert [len(res) == 2 for res in result]


class TestRecursiveMasking:
    def setup_method(self):
        self.num_peaks = 100
        self.min_distance = 5
        self.data = np.random.rand(100, 100, 100)
        self.rotation_matrix = np.eye(3)
        self.mask = np.random.rand(20, 20, 20)
        self.rotation_space = np.zeros_like(self.data)
        self.rotation_mapping = {0: (0, 0, 0)}

    @pytest.mark.parametrize("num_peaks", (1, 100))
    @pytest.mark.parametrize("compute_rotation", (True, False))
    @pytest.mark.parametrize("minimum_score", (None, 0.5))
    def test__call__(self, num_peaks, compute_rotation, minimum_score):
        peak_caller = PeakCallerRecursiveMasking(
            shape=self.data.shape,
            num_peaks=num_peaks,
            min_distance=self.min_distance,
            min_score=minimum_score,
        )
        rotation_space, rotation_mapping = None, None
        if compute_rotation:
            rotation_space = self.rotation_space
            rotation_mapping = self.rotation_mapping

        state = peak_caller.init_state()
        state = peak_caller(
            state,
            self.data.copy(),
            rotation_matrix=self.rotation_matrix,
            mask=self.mask,
            rotation_space=rotation_space,
            rotation_mapping=rotation_mapping,
        )

        if minimum_score is None:
            assert len(state[0] <= num_peaks)
        else:
            peaks = state[0].astype(int)
            assert np.all(self.data[tuple(peaks.T)] >= minimum_score)


class TestMaxScoreOverRotations:
    def setup_method(self):
        self.num_peaks = 100
        self.min_distance = 5
        self.data = np.random.rand(100, 100, 100)
        self.rotation_matrix = np.eye(3)

    def test_initialization(self):
        _ = MaxScoreOverRotations(
            shape=self.data.shape,
            translation_offset=np.zeros(self.data.ndim, dtype=int),
        )

    @pytest.mark.parametrize("use_memmap", [False, True])
    def test__iter__(self, use_memmap: bool):
        score_analyzer = MaxScoreOverRotations(
            shape=self.data.shape,
            use_memmap=use_memmap,
        )
        state = score_analyzer.init_state()
        state = score_analyzer(state, self.data, rotation_matrix=self.rotation_matrix)
        res = score_analyzer.result(state)
        assert np.allclose(res[0].shape, self.data.shape)
        assert res[0].dtype == be._float_dtype
        assert res[1].size == self.data.ndim
        assert np.allclose(res[2].shape, self.data.shape)
        assert len(res) == 4

    @pytest.mark.parametrize("use_memmap", [False, True])
    @pytest.mark.parametrize("score_threshold", [0, 1e10, -1e10])
    def test__call__(self, use_memmap: bool, score_threshold: float):
        score_analyzer = MaxScoreOverRotations(
            shape=self.data.shape,
            score_threshold=score_threshold,
            translation_offset=np.zeros(self.data.ndim, dtype=int),
            use_memmap=use_memmap,
        )
        state = score_analyzer.init_state()
        state = score_analyzer(state, self.data, rotation_matrix=self.rotation_matrix)

        data2 = self.data * 2
        score_analyzer(state, data2, rotation_matrix=self.rotation_matrix)
        scores, translation_offset, rotations, mapping = score_analyzer.result(state)

        assert np.all(scores >= score_threshold)
        max_scores = np.maximum(self.data, data2)
        max_scores = np.maximum(max_scores, score_threshold)
        assert np.allclose(scores, max_scores)

    @pytest.mark.parametrize("use_memmap", [False, True])
    @pytest.mark.parametrize("score_threshold", [0, 1e10, -1e10])
    def test_merge(self, use_memmap: bool, score_threshold: float):
        score_analyzer = MaxScoreOverRotations(
            shape=self.data.shape,
            score_threshold=score_threshold,
            translation_offset=np.zeros(self.data.ndim, dtype=int),
            use_memmap=use_memmap,
        )
        state1 = score_analyzer.init_state()
        state1, score_analyzer(state1, self.data, rotation_matrix=self.rotation_matrix)

        data2 = self.data * 2
        score_analyzer2 = MaxScoreOverRotations(
            shape=self.data.shape,
            score_threshold=score_threshold,
            translation_offset=np.zeros(self.data.ndim, dtype=int),
            use_memmap=use_memmap,
        )
        state2 = score_analyzer2.init_state()
        state2 = score_analyzer2(state2, data2, rotation_matrix=self.rotation_matrix)
        states = [score_analyzer.result(state1), score_analyzer2.result(state2)]

        ret = MaxScoreOverRotations.merge(
            results=states, use_memmap=use_memmap, score_threshold=score_threshold
        )
        scores, translation, rotations, mapping = ret
        assert np.all(scores >= score_threshold)
        max_scores = np.maximum(self.data, data2)
        max_scores = np.maximum(max_scores, score_threshold)
        assert np.allclose(scores, max_scores)
