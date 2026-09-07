import pytest
import numpy as np
from dataclasses import dataclass

from tme.analyzer import (
    MaxScoreOverRotations,
    MaxScoreOverTranslations,
    MaxScoreOverRotationsConstrained,
    PeakCallerSort,
    PeakCallerMaximumFilter,
    PeakCallerFast,
    PeakCallerRecursiveMasking,
    PeakCallerScipy,
    PeakClustering,
)
from tme.analyzer._utils import cart_to_score, score_to_cart
from tme.backends import backend as be


@dataclass
class AnalyzerTestContext:
    data: np.ndarray
    rotation_matrix: np.ndarray
    config: dict = None

    def __post_init__(self):
        if self.config is None:
            self.config = {}

    def create_object(self, object_class, **overrides):
        full_config = {**self.config, **overrides}
        return object_class(**full_config)

    def run_object(self, obj, additional_kwargs=None, data=None):
        additional_kwargs = additional_kwargs or {}
        data = data if data is not None else self.data

        state = obj.init_state()
        state = obj(
            state,
            scores=data.copy(),
            rotation_matrix=self.rotation_matrix,
            **additional_kwargs,
        )
        return state

    def get_result(self, obj, state):
        return obj.result(state)

    def create_and_run(
        self, object_class, additional_kwargs=None, data=None, **overrides
    ):
        obj = self.create_object(object_class, **overrides)
        return self.run_object(obj, additional_kwargs, data)

    def create_run_and_result(
        self, object_class, additional_kwargs=None, data=None, **overrides
    ):
        obj = self.create_object(object_class, **overrides)
        state = self.run_object(obj, additional_kwargs, data)
        return self.get_result(obj, state)

    def create_multiple_results(self, object_class, count=2, **overrides):
        return [
            self.create_run_and_result(object_class, **overrides) for _ in range(count)
        ]


def create_test_context(**kwargs):
    np.random.seed(123)

    shape = (50, 50, 50)
    defaults = {
        "data": np.random.rand(*shape),
        "rotation_matrix": np.eye(3),
        "config": {"shape": shape} | kwargs,
    }
    return AnalyzerTestContext(**defaults)


@pytest.fixture
def peak_context():
    return create_test_context(num_peaks=100, min_distance=5)


@pytest.fixture
def aggregation_context():
    return create_test_context(
        n_rotations=30,
        translation_offset=(0, 0, 0),
        positions=np.array([[15, 30, 15], [90, 60, 30], [45, 50, 10]]),
        rotations=np.array([np.eye(3), np.eye(3) * -1, np.eye(3)]),
        cone_angle=20,
    )


class TestPeakCaller:

    PEAK_CALLER_CHILDREN = [
        PeakCallerSort,
        PeakCallerMaximumFilter,
        PeakCallerFast,
        PeakCallerRecursiveMasking,
        PeakCallerScipy,
        PeakClustering,
    ]

    @pytest.mark.parametrize("analyzer", PEAK_CALLER_CHILDREN)
    def test_initialization(self, analyzer, peak_context):
        caller = peak_context.create_object(analyzer)
        assert isinstance(caller, analyzer)

    @pytest.mark.parametrize("analyzer", PEAK_CALLER_CHILDREN)
    @pytest.mark.parametrize("num_peaks", [1, 100])
    @pytest.mark.parametrize("min_score", [None, 0.5])
    def test_result(self, analyzer, num_peaks, min_score, peak_context):
        result = peak_context.create_and_run(
            analyzer, num_peaks=num_peaks, min_score=min_score
        )
        translations, rotations, scores, details = result

        if min_score is not None:
            # Filler peaks for GPU computation are marked as -1
            peak_scores = scores[scores != -1]
            assert np.all(peak_scores >= min_score)
        assert len(translations) <= num_peaks

    @pytest.mark.parametrize("analyzer", PEAK_CALLER_CHILDREN)
    def test_merge(self, analyzer, peak_context, num_peaks=100):
        results = peak_context.create_multiple_results(
            analyzer, count=2, num_peaks=num_peaks
        )
        merged = analyzer.merge(
            results=results,
            num_peaks=100,
            min_distance=peak_context.config["min_distance"],
        )

        for result in results:
            assert len(result) == len(merged)
        assert len(merged[0]) <= num_peaks

    @pytest.mark.parametrize("compute_rotation", [True, False])
    def test_recursive_mask(self, compute_rotation, peak_context, num_peaks=100):
        peak_context.config.update(
            {
                "mask": np.random.rand(20, 20, 20),
                "rotation_space": np.zeros_like(peak_context.data),
                "rotation_mapping": {0: (0, 0, 0)},
            }
        )
        caller = peak_context.create_object(
            PeakCallerRecursiveMasking, num_peaks=num_peaks
        )
        additional_kwargs = {"mask": peak_context.config["mask"]}
        if compute_rotation:
            additional_kwargs["rotation_space"] = peak_context.config["rotation_space"]
            additional_kwargs["rotation_mapping"] = peak_context.config[
                "rotation_mapping"
            ]

        state = peak_context.run_object(caller, additional_kwargs)
        result = peak_context.get_result(caller, state)
        assert len(result[0]) <= num_peaks

    @pytest.mark.parametrize("analyzer", PEAK_CALLER_CHILDREN)
    def test_correct_background(self, analyzer, aggregation_context):
        instance = aggregation_context.create_object(analyzer)
        state = aggregation_context.run_object(instance)
        new_state = instance.correct_background(
            state, mean=aggregation_context.data, inv_std=1
        )
        assert np.all(state[2] >= new_state[2])


class TestAggregation:
    AGGREGATION_CHILDREN = [
        MaxScoreOverRotations,
        MaxScoreOverTranslations,
        MaxScoreOverRotationsConstrained,
    ]

    @pytest.mark.parametrize("analyzer", AGGREGATION_CHILDREN)
    def test_initialization(self, analyzer, aggregation_context):
        instance = aggregation_context.create_object(analyzer)
        assert isinstance(instance, analyzer)

    @pytest.mark.parametrize("use_memmap", [False, True])
    @pytest.mark.parametrize("analyzer", AGGREGATION_CHILDREN)
    def test_result(self, analyzer, use_memmap: bool, aggregation_context):
        result = aggregation_context.create_run_and_result(
            analyzer, use_memmap=use_memmap
        )
        scores, offset, rotations, mapping, *_ = result
        assert scores.ndim == rotations.ndim
        assert all([x in mapping for x in np.unique(rotations).astype(int) if x != -1])

    @pytest.mark.parametrize("use_memmap", [False, True])
    @pytest.mark.parametrize("analyzer", AGGREGATION_CHILDREN)
    def test_merge(self, analyzer, use_memmap, aggregation_context):
        results = aggregation_context.create_multiple_results(
            analyzer,
            count=2,
            use_memmap=use_memmap,
        )
        merged = analyzer.merge(
            results,
            use_memmap=use_memmap,
        )
        scores, offset, rotations, mapping, *_ = merged
        assert scores.ndim == rotations.ndim
        assert all([x in mapping for x in np.unique(rotations).astype(int) if x != -1])

    def test_constrained(self, aggregation_context):
        instance = aggregation_context.create_object(MaxScoreOverRotationsConstrained)
        result = aggregation_context.run_object(instance)

        mask = instance._get_score_mask(
            mask=np.ones((instance._index_grid[0].shape[0]), dtype=bool),
            scores=aggregation_context.data,
        )

        scores, rotations, *_ = result
        assert np.allclose(scores, scores * mask)
        assert np.all(scores[np.invert(mask)] == 0)

    @pytest.mark.parametrize("analyzer", AGGREGATION_CHILDREN)
    def test_correct_background(self, analyzer, aggregation_context):
        instance = aggregation_context.create_object(analyzer)
        state = aggregation_context.run_object(instance)
        new_state = instance.correct_background(
            state, mean=aggregation_context.data, inv_std=1
        )
        assert np.all(state[0] >= new_state[0])


def fourier_padding(target_shape, template_shape):
    """
    Computes efficient shape for Fourier transforms and potential associated shifts.

    Returns
    -------
    Tuple[tuple of int, tuple of int, tuple of int, tuple of int]
        Tuple with convolution, forward FT, inverse FT shape and corresponding shift.
        When batched, shapes are prefixed with (target_batch, template_batch).
    """
    batch_prefix = ()

    pad_shape = np.maximum(target_shape, template_shape)
    conv, fwd, inv = be.compute_convolution_shapes(pad_shape, np.ones_like(pad_shape))

    fourier_shift = (
        1 - np.divide(template_shape, 2).astype(int) - np.mod(template_shape, 2)
    )

    shape_diff = np.subtract(target_shape, template_shape)
    if np.sum(shape_diff < 0):
        shape_shift = np.divide(shape_diff, 2)
        offset = np.mod(shape_diff, 2)
        shape_shift = np.multiply(np.add(shape_shift, offset), shape_diff < 0)
        fourier_shift = np.subtract(fourier_shift, shape_shift).astype(int)

    fourier_shift = tuple(int(x) for x in fourier_shift)

    conv = batch_prefix + tuple(conv)
    fwd = batch_prefix + tuple(fwd)
    inv = batch_prefix + tuple(inv)
    fourier_shift = (0,) * len(batch_prefix) + fourier_shift

    return conv, fwd, inv, fourier_shift


@pytest.mark.parametrize(
    "target_shape,template_shape,convolution_mode,split_factor",
    [
        (np.array([100, 100, 100]), np.array([30, 30, 30]), "same", 1),
        (np.array([100, 100, 100]), np.array([30, 30, 30]), "valid", 2),
    ],
)
def test_coordinate_roundtrip(
    target_shape, template_shape, convolution_mode, split_factor
):
    np.random.seed(42)

    conv_shape, fast_shape, fast_ft_shape, fourier_shift = fourier_padding(
        target_shape, template_shape
    )

    split_target_shape = target_shape.copy()
    split_target_shape[0] = target_shape[0] // split_factor

    n_positions = 50
    positions_cart = np.random.randint(0, split_target_shape, size=(n_positions, 3))

    positions_score, valid1 = cart_to_score(
        positions_cart,
        fast_shape=fast_shape,
        targetshape=split_target_shape,
        templateshape=template_shape,
        convolution_shape=conv_shape,
        fourier_shift=fourier_shift,
        convolution_mode=convolution_mode,
    )

    positions_recovered, valid2 = score_to_cart(
        positions_score,
        fast_shape=fast_shape,
        targetshape=split_target_shape,
        templateshape=template_shape,
        convolution_shape=conv_shape,
        fourier_shift=fourier_shift,
        convolution_mode=convolution_mode,
    )

    assert valid1.dtype == bool
    assert valid1.size == positions_cart.shape[0]

    valid = np.logical_and(valid1, valid2)
    assert np.allclose(positions_cart[valid], positions_recovered[valid])
