import pytest
import numpy as np
from scipy.signal import correlate

from tme.backends import backend as be
from tme.mask import soft_edge, threshold_mask
from tme.matching_utils import (
    create_mask,
    scramble_phases,
    apply_convolution_mode,
    sliding_window_slices,
    _standardize_safe,
)


class TestSlidingWindowSlices:
    @pytest.mark.parametrize("ndim", (1, 2, 3, 4))
    def test_windows_are_uniform_and_cover_edges(self, ndim):
        shape = tuple(range(10, 10 + ndim))  # distinct, non-divisible extents
        windows = list(sliding_window_slices(shape, length=4, step=3))
        assert all(len(w) == ndim for w in windows)
        # Every window has the requested edge length along every axis.
        assert all(s.stop - s.start == 4 for w in windows for s in w)
        # The last window along each axis is snapped to the array edge.
        for axis, extent in enumerate(shape):
            starts = sorted({w[axis].start for w in windows})
            assert starts[0] == 0
            assert starts[-1] == extent - 4

    def test_raises_when_window_larger_than_extent(self):
        with pytest.raises(ValueError):
            list(sliding_window_slices((3, 10), length=4, step=2))


@pytest.fixture
def shape_context():
    return (10, 20, 15), (10, 10, 10)


@pytest.fixture
def data_context(shape_context):
    target_shape, template_shape = shape_context
    return np.random.rand(*target_shape), np.random.rand(*template_shape)


@pytest.mark.parametrize("mask_type", ["ellipse", "box", "tube", "membrane"])
def test_create_mask(mask_type: str, shape_context):
    target_shape, template_shape = shape_context
    create_mask(
        mask_type=mask_type,
        shape=target_shape,
        radius=5,
        center=np.divide(target_shape, 2),
        height=np.max(target_shape) // 2 - 1,
        size=np.divide(target_shape, 2).astype(int),
        thickness=2,
        separation=2,
        symmetry_axis=1,
        inner_radius=5,
        outer_radius=10,
    )


def test_create_mask_threshold():
    data = np.zeros((20, 20, 20), dtype=np.float32)
    data[5:15, 5:15, 5:15] = 1.0
    mask = create_mask(mask_type="threshold", data=data, threshold=0.5)
    assert mask.shape == data.shape
    assert mask[10, 10, 10] == 1.0
    assert mask[0, 0, 0] == 0.0


def test_create_mask_error():
    with pytest.raises(ValueError):
        create_mask(mask_type=None)


def test_soft_edge_zero_width():
    binary = np.zeros((20, 20), dtype=np.float32)
    binary[5:15, 5:15] = 1.0
    result = soft_edge(binary, soft_edge_width=0)
    np.testing.assert_array_equal(result, binary)


@pytest.mark.parametrize("method", ["gaussian", "cosine"])
def test_soft_edge_interior_preserved(method):
    binary = np.zeros((30, 30), dtype=np.float32)
    binary[10:20, 10:20] = 1.0
    result = soft_edge(binary, soft_edge_width=3, method=method)
    # Interior should be exactly 1.0
    np.testing.assert_array_equal(result[10:20, 10:20], 1.0)


@pytest.mark.parametrize("method", ["gaussian", "cosine"])
def test_soft_edge_decay(method):
    binary = np.zeros((30, 30), dtype=np.float32)
    binary[10:20, 10:20] = 1.0
    result = soft_edge(binary, soft_edge_width=3, method=method)
    # Voxels just outside should have values between 0 and 1
    assert 0 < result[9, 15] < 1.0
    # Far-away voxels should be 0
    assert result[0, 0] == 0.0


@pytest.mark.parametrize("method", ["gaussian", "cosine"])
def test_soft_edge_3d(method):
    binary = np.zeros((20, 20, 20), dtype=np.float32)
    binary[5:15, 5:15, 5:15] = 1.0
    result = soft_edge(binary, soft_edge_width=2, method=method)
    assert result.shape == binary.shape
    assert result[10, 10, 10] == 1.0
    assert result[0, 0, 0] == 0.0
    assert 0 < result[4, 10, 10] < 1.0


def test_soft_edge_invalid_method():
    binary = np.ones((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="method must be"):
        soft_edge(binary, soft_edge_width=1, method="invalid")


def test_threshold_mask_basic():
    data = np.zeros((20, 20), dtype=np.float32)
    data[5:15, 5:15] = 2.0
    mask = threshold_mask(data, threshold=1.0)
    assert mask[10, 10] == 1.0
    assert mask[0, 0] == 0.0


def test_threshold_mask_extend():
    data = np.zeros((30, 30), dtype=np.float32)
    data[10:20, 10:20] = 1.0
    mask = threshold_mask(data, threshold=0.5, extend=2)
    # Original interior
    assert mask[15, 15] == 1.0
    # Extended region (within 2 voxels of boundary)
    assert mask[8, 15] == 1.0
    # Far outside
    assert mask[0, 0] == 0.0


@pytest.mark.parametrize("method", ["gaussian", "cosine"])
def test_threshold_mask_soft_edge(method):
    data = np.zeros((30, 30), dtype=np.float32)
    data[10:20, 10:20] = 1.0
    mask = threshold_mask(data, threshold=0.5, soft_edge_width=3, method=method)
    assert mask[15, 15] == 1.0
    assert 0 < mask[9, 15] < 1.0
    assert mask[0, 0] == 0.0


@pytest.mark.parametrize("mask_type", ["ellipse", "box", "tube"])
def test_create_mask_hard_edge_binary(mask_type, shape_context):
    target_shape, _ = shape_context
    mask = create_mask(
        mask_type=mask_type,
        shape=target_shape,
        radius=5,
        center=np.divide(target_shape, 2),
        height=np.max(target_shape) // 2 - 1,
        size=np.divide(target_shape, 2).astype(int),
        thickness=2,
        separation=2,
        symmetry_axis=1,
        inner_radius=5,
        outer_radius=10,
        sigma_decay=0,
    )
    # Hard edge should produce only 0s and 1s
    unique = np.unique(mask)
    assert all(v in [0.0, 1.0] for v in unique)


@pytest.mark.parametrize("mask_type", ["ellipse", "box", "tube"])
@pytest.mark.parametrize("method", ["gaussian", "cosine"])
def test_create_mask_soft_edge(mask_type, method, shape_context):
    target_shape, _ = shape_context
    mask = create_mask(
        mask_type=mask_type,
        shape=target_shape,
        radius=5,
        center=np.divide(target_shape, 2),
        height=np.max(target_shape) // 2 - 1,
        size=np.divide(target_shape, 2).astype(int),
        thickness=2,
        separation=2,
        symmetry_axis=1,
        inner_radius=5,
        outer_radius=10,
        sigma_decay=2,
        method=method,
    )
    # Soft edge should produce values between 0 and 1
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
    # Should have some intermediate values
    unique = np.unique(mask)
    assert len(unique) > 2


def test_scramble_phases(data_context):
    scramble_phases(arr=data_context[0])


@pytest.mark.parametrize("convolution_mode", ["full", "valid", "same"])
def test_apply_convolution_mode(convolution_mode, data_context):
    target, template = data_context
    correlation = correlate(target, template, method="direct", mode="full")
    ret = apply_convolution_mode(
        arr=correlation,
        convolution_mode=convolution_mode,
        s1=target.shape,
        s2=template.shape,
    )
    if convolution_mode == "full":
        expected_size = correlation.shape
    elif convolution_mode == "same":
        expected_size = target.shape
    else:
        expected_size = np.subtract(target.shape, template.shape)
        expected_size += 1
    assert np.allclose(ret.shape, expected_size)


def test_apply_convolution_mode_error(data_context):
    target, template = data_context
    correlation = correlate(target, template, method="direct", mode="full")
    with pytest.raises(ValueError):
        _ = apply_convolution_mode(
            arr=correlation,
            convolution_mode=None,
            s1=target.shape,
            s2=template.shape,
        )


def test_standardize_safe(data_context):
    _, template = data_context

    mask = be.ones_like(template)
    n_observations = mask.sum()

    result = _standardize_safe(template, mask, n_observations)
    assert result.shape == template.shape
    assert result.dtype == template.dtype
    assert np.allclose(result.mean(), 0, atol=0.1)
    assert np.allclose(result.std(), 1, atol=0.1)


def test_minimum_score_from_fp_matches_rickgauer():
    from scipy.special import erfcinv
    import numpy as np
    from tme.matching_utils import minimum_score_from_fp

    std, n_corr, n_fp = 0.05, 10_000_000, 10.0
    expected = float(erfcinv(2 * n_fp / n_corr) * np.sqrt(2) * std)
    assert minimum_score_from_fp(std, n_corr, n_fp) == pytest.approx(expected)
