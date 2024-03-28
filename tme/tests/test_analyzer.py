from tempfile import mkstemp

import pytest
import numpy as np

from tme.analyzer import (
    ScoreStatistics,
    MaxScoreOverRotations,
    PeakCaller,
    PeakCallerSort,
    PeakCallerMaximumFilter,
    PeakCallerFast,
    PeakCallerRecursiveMasking,
    PeakCallerScipy,
    PeakClustering,
    MemmapHandler,
)


PEAK_CALLER_CHILDREN = [
    PeakCallerSort,
    PeakCallerMaximumFilter,
    PeakCallerFast,
    PeakCallerRecursiveMasking,
    PeakCallerScipy,
    PeakClustering,
]


class TestPeakCallers:
    def setup_method(self):
        self.number_of_peaks = 100
        self.min_distance = 5
        self.data = np.random.rand(100, 100, 100)
        self.rotation_matrix = np.eye(3)

    @pytest.mark.parametrize("peak_caller", PEAK_CALLER_CHILDREN)
    def test_initialization(self, peak_caller):
        _ = peak_caller(number_of_peaks=100, min_distance=5)

    def test_initialization_error(self):
        with pytest.raises(TypeError):
            _ = PeakCaller(number_of_peaks=100, min_distance=5)

    @pytest.mark.parametrize("peak_caller", PEAK_CALLER_CHILDREN)
    def test_initialization_error_parameter(self, peak_caller):
        with pytest.raises(ValueError):
            _ = peak_caller(number_of_peaks=0, min_distance=5)
        with pytest.raises(ValueError):
            _ = peak_caller(number_of_peaks=-1, min_distance=5)
        with pytest.raises(ValueError):
            _ = peak_caller(number_of_peaks=-1, min_distance=-1)

    @pytest.mark.parametrize("peak_caller", PEAK_CALLER_CHILDREN)
    @pytest.mark.parametrize("number_of_peaks", (1, 100))
    @pytest.mark.parametrize("minimum_score", (None, 0.5))
    def test__call__(self, peak_caller, number_of_peaks, minimum_score):
        peak_caller = peak_caller(
            number_of_peaks=number_of_peaks, min_distance=self.min_distance
        )
        peak_caller(
            score_space=self.data.copy(),
            rotation_matrix=self.rotation_matrix,
            minimum_score=minimum_score,
        )
        candidates = tuple(peak_caller)
        if minimum_score is None:
            assert len(candidates[0] <= number_of_peaks)
        else:
            peaks = candidates[0].astype(int)
            print(self.data[tuple(peaks.T)])
            assert np.all(self.data[tuple(peaks.T)] >= minimum_score)

    @pytest.mark.parametrize("peak_caller", PEAK_CALLER_CHILDREN)
    @pytest.mark.parametrize("number_of_peaks", (1, 100))
    def test_merge(self, peak_caller, number_of_peaks):
        peak_caller1 = peak_caller(
            number_of_peaks=number_of_peaks, min_distance=self.min_distance
        )
        peak_caller1(score_space=self.data, rotation_matrix=self.rotation_matrix)

        peak_caller2 = peak_caller(
            number_of_peaks=number_of_peaks, min_distance=self.min_distance
        )
        peak_caller2(score_space=self.data, rotation_matrix=self.rotation_matrix)

        parameters = [tuple(peak_caller1), tuple(peak_caller2)]

        result = tuple(
            peak_caller.merge(
                candidates=parameters,
                number_of_peaks=number_of_peaks,
                min_distance=self.min_distance,
            )
        )
        assert [len(res) == 2 for res in result]


class TestRecursiveMasking:
    def setup_method(self):
        self.number_of_peaks = 100
        self.min_distance = 5
        self.data = np.random.rand(100, 100, 100)
        self.rotation_matrix = np.eye(3)
        self.mask = np.random.rand(20, 20, 20)
        self.rotation_space = np.zeros_like(self.data)
        self.rotation_mapping = {0: (0, 0, 0)}

    @pytest.mark.parametrize("number_of_peaks", (1, 100))
    @pytest.mark.parametrize("compute_rotation", (True, False))
    @pytest.mark.parametrize("minimum_score", (None, 0.5))
    def test__call__(self, number_of_peaks, compute_rotation, minimum_score):
        peak_caller = PeakCallerRecursiveMasking(
            number_of_peaks=number_of_peaks, min_distance=self.min_distance
        )
        rotation_space, rotation_mapping = None, None
        if compute_rotation:
            rotation_space = self.rotation_space
            rotation_mapping = self.rotation_mapping

        peak_caller(
            score_space=self.data.copy(),
            rotation_matrix=self.rotation_matrix,
            mask=self.mask,
            rotation_space=rotation_space,
            rotation_mapping=rotation_mapping,
        )

        candidates = tuple(peak_caller)
        if minimum_score is None:
            assert len(candidates[0] <= number_of_peaks)
        else:
            peaks = candidates[0].astype(int)
            print(self.data[tuple(peaks.T)])
            assert np.all(self.data[tuple(peaks.T)] >= minimum_score)


class TestMaxScoreOverRotations:
    def setup_method(self):
        self.number_of_peaks = 100
        self.min_distance = 5
        self.data = np.random.rand(100, 100, 100)
        self.rotation_matrix = np.eye(3)

    def test_initialization(self):
        _ = MaxScoreOverRotations(
            score_space_shape=self.data.shape,
            score_space_dtype=self.data.dtype,
            translation_offset=np.zeros(self.data.ndim, dtype=int),
            rotation_space_dtype=np.int32,
        )

    @pytest.mark.parametrize("use_memmap", [False, True])
    def test__iter__(self, use_memmap: bool):
        score_analyzer = MaxScoreOverRotations(
            score_space_shape=self.data.shape,
            score_space_dtype=self.data.dtype,
            rotation_space_dtype=np.int32,
            use_memmap=use_memmap,
        )
        score_analyzer(score_space=self.data, rotation_matrix=self.rotation_matrix)
        res = tuple(score_analyzer)
        assert np.allclose(res[0].shape, self.data.shape)
        assert res[0].dtype == self.data.dtype
        assert res[1].size == self.data.ndim
        assert np.allclose(res[2].shape, self.data.shape)
        assert len(res) == 4

    @pytest.mark.parametrize("use_memmap", [False, True])
    def test__call__(self, use_memmap: bool):
        score_analyzer = MaxScoreOverRotations(
            score_space_shape=self.data.shape,
            score_space_dtype=self.data.dtype,
            translation_offset=np.zeros(self.data.ndim, dtype=int),
            rotation_space_dtype=np.int32,
            use_memmap=use_memmap,
        )
        score_analyzer(score_space=self.data, rotation_matrix=self.rotation_matrix)

        data2 = self.data * 2
        score_analyzer(score_space=data2, rotation_matrix=self.rotation_matrix)
        scores, translation_offset, rotations, mapping = tuple(score_analyzer)
        assert np.allclose(scores, np.maximum(self.data, data2))

    @pytest.mark.parametrize("use_memmap", [False, True])
    def test_merge(self, use_memmap: bool):
        score_analyzer = MaxScoreOverRotations(
            score_space_shape=self.data.shape,
            score_space_dtype=self.data.dtype,
            translation_offset=np.zeros(self.data.ndim, dtype=int),
            rotation_space_dtype=np.int32,
            use_memmap=use_memmap,
        )
        score_analyzer(score_space=self.data, rotation_matrix=self.rotation_matrix)

        data2 = self.data * 2
        score_analyzer2 = MaxScoreOverRotations(
            score_space_shape=self.data.shape,
            score_space_dtype=self.data.dtype,
            translation_offset=np.zeros(self.data.ndim, dtype=int),
            rotation_space_dtype=np.int32,
            use_memmap=use_memmap,
        )
        score_analyzer2(score_space=data2, rotation_matrix=self.rotation_matrix)

        parameters = [tuple(score_analyzer), tuple(score_analyzer2)]

        ret = MaxScoreOverRotations.merge(parameters, use_memmap=use_memmap)
        scores, translation, rotations, mapping = ret
        assert np.allclose(scores, np.maximum(self.data, data2))


class TestMemmapHandler:
    def setup_method(self):
        self.number_of_peaks = 100
        self.min_distance = 5
        self.data = np.random.rand(100, 100, 100)
        self.indices = tuple(np.indices(self.data.shape))

        self.rotation_matrix = np.eye(3)
        rotation_matrix2 = np.eye(3)
        rotation_matrix2[0, 0] = -1

        rotation_matrix = "_".join(self.rotation_matrix.ravel().astype(str))
        rotation_matrix2 = "_".join(rotation_matrix2.ravel().astype(str))

        self.path_translation = {
            rotation_matrix: mkstemp()[1],
            rotation_matrix2: mkstemp()[1],
        }

    def test_initialization(self):
        _ = MemmapHandler(
            path_translation=self.path_translation,
            shape=self.data.shape,
            dtype=self.data.dtype,
            indices=self.indices,
        )

    def test__call__(self):
        score_analyzer = MemmapHandler(
            path_translation=self.path_translation,
            shape=self.data.shape,
            dtype=self.data.dtype,
            indices=self.indices,
        )
        score_analyzer(score_space=self.data, rotation_matrix=self.rotation_matrix)
        rotation_filepath = score_analyzer._rotation_matrix_to_filepath(
            rotation_matrix=self.rotation_matrix
        )
        array = np.memmap(
            rotation_filepath,
            mode="r+",
            shape=score_analyzer.shape,
            dtype=score_analyzer.dtype,
        )
        assert np.allclose(array, self.data)

    def test__iter__(self):
        score_analyzer = MemmapHandler(
            path_translation=self.path_translation,
            shape=self.data.shape,
            dtype=self.data.dtype,
            indices=self.indices,
        )
        res = tuple(score_analyzer)
        assert res == (None,)

    def test_merge(self):
        score_analyzer = MemmapHandler(
            path_translation=self.path_translation,
            shape=self.data.shape,
            dtype=self.data.dtype,
            indices=self.indices,
        )
        res = MemmapHandler.merge(score_analyzer)
        assert res is None

    def test_update_indices(self):
        score_analyzer = MemmapHandler(
            path_translation=self.path_translation,
            shape=self.data.shape,
            dtype=self.data.dtype,
            indices=self.indices,
        )
        new_indices = np.random.rand(3)
        score_analyzer.update_indices(new_indices)
        assert np.allclose(score_analyzer._indices, new_indices)

    def test__rotation_matrix_to_filepath(self):
        score_analyzer = MemmapHandler(
            path_translation=self.path_translation,
            shape=self.data.shape,
            dtype=self.data.dtype,
            indices=self.indices,
        )

        rotation_matrix = list(self.path_translation.keys())[0]
        rotation_filepath = score_analyzer._rotation_matrix_to_filepath(
            rotation_matrix=self.rotation_matrix
        )
        assert rotation_filepath == self.path_translation.get(rotation_matrix)


class TestScoreStatistics:
    def setup_method(self):
        self.data = np.random.rand(100, 100, 100).astype(np.float32)
        self.rotation_matrix = np.eye(self.data.ndim)
        self.reference_position = np.array([5, 5, 5], dtype=int)
        self.min_score = 0
        self.min_distance = 5

    def test_initialization(self):
        _ = ScoreStatistics()

    def test__call__(self):
        score_statistics = ScoreStatistics(
            min_distance=self.min_distance, reference_position=self.reference_position
        )

        score_statistics(score_space=self.data, rotation_matrix=self.rotation_matrix)

    def test__iter__(self):
        score_statistics = ScoreStatistics(
            min_distance=self.min_distance, reference_position=self.reference_position
        )

        score_statistics(score_space=self.data, rotation_matrix=self.rotation_matrix)
        res = tuple(score_statistics)
        assert len(res) == 13

    def test_merge(self):
        score_statistics = ScoreStatistics(
            min_distance=self.min_distance, reference_position=self.reference_position
        )

        score_statistics(score_space=self.data, rotation_matrix=self.rotation_matrix)

        score_statistics2 = ScoreStatistics(
            min_distance=self.min_distance, reference_position=self.reference_position
        )

        score_statistics2(score_space=self.data, rotation_matrix=self.rotation_matrix)

        parameters = [tuple(score_statistics), tuple(score_statistics2)]

        res = ScoreStatistics.merge(parameters)
        assert len(res) == 13

    def test_set_reference(self):
        score_statistics = ScoreStatistics(
            min_distance=self.min_distance, reference_position=self.reference_position
        )

        score_statistics(score_space=self.data, rotation_matrix=self.rotation_matrix)

        score_statistics.set_reference(self.data, rotation_matrix=self.rotation_matrix)

        assert score_statistics.has_reference.value == 1
