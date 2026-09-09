import pytest

from tme.memory import (
    MATCHING_MEMORY_REGISTRY,
    estimate_memory_usage,
    register_memory,
    MemoryProfile,
    compute_schedule,
)


@pytest.fixture
def test_shapes():
    return {
        "target": (100, 100, 100),
        "template": (20, 20, 20),
    }


@pytest.mark.parametrize("memory", list(MATCHING_MEMORY_REGISTRY.values()))
@pytest.mark.parametrize("ncores", (1, 8))
def test_registered(test_shapes, memory, ncores):
    instance = memory(
        fast_shape=test_shapes["target"],
        ft_shape=test_shapes["template"],
        float_nbytes=4,
        complex_nbytes=8,
        integer_nbytes=4,
    )
    nbytes = instance.base_usage() + instance.per_fork() * ncores
    assert nbytes >= 0 and isinstance(nbytes, int)


def test_register_single_name():
    """Test registering a memory estimator with one name."""

    @register_memory("TEST_SINGLE")
    class TestProfile(MemoryProfile):
        base_float = 1
        base_complex = 1

    assert "TEST_SINGLE" in MATCHING_MEMORY_REGISTRY
    assert MATCHING_MEMORY_REGISTRY["TEST_SINGLE"] == TestProfile
    del MATCHING_MEMORY_REGISTRY["TEST_SINGLE"]


def test_register_multiple_names():
    """Test registering a memory estimator with multiple names."""

    @register_memory("TEST_A", "TEST_B", "TEST_C")
    class TestProfile(MemoryProfile):
        base_float = 2

    for name in ["TEST_A", "TEST_B", "TEST_C"]:
        assert name in MATCHING_MEMORY_REGISTRY
        assert MATCHING_MEMORY_REGISTRY[name] == TestProfile
    for name in ["TEST_A", "TEST_B", "TEST_C"]:
        del MATCHING_MEMORY_REGISTRY[name]


def test_basic_estimation(test_shapes):
    """Test basic memory estimation."""
    nbytes = estimate_memory_usage(
        shape1=test_shapes["target"],
        shape2=test_shapes["template"],
        matching_method="CC",
        ncores=1,
    )
    assert nbytes >= 0 and isinstance(nbytes, int)


def test_unsupported_method_raises(test_shapes):
    """Test that unsupported methods raise ValueError."""
    with pytest.raises(ValueError, match="Supported are"):
        estimate_memory_usage(
            shape1=test_shapes["target"],
            shape2=test_shapes["template"],
            matching_method="NONEXISTENT",
            ncores=1,
        )


@pytest.mark.parametrize("ncores", (1, 8))
def test_adds_memory(test_shapes, ncores):
    """Test that adding an analyzer increases memory."""
    base = estimate_memory_usage(
        shape1=test_shapes["target"],
        shape2=test_shapes["template"],
        matching_method="CC",
        ncores=ncores,
    )
    with_analyzer = estimate_memory_usage(
        shape1=test_shapes["target"],
        shape2=test_shapes["template"],
        matching_method="CC",
        analyzer_method="MaxScoreOverRotations",
        ncores=ncores,
    )

    with_backend = estimate_memory_usage(
        shape1=test_shapes["target"],
        shape2=test_shapes["template"],
        matching_method="CC",
        backend="cupy",
        ncores=ncores,
    )

    total = estimate_memory_usage(
        shape1=test_shapes["target"],
        shape2=test_shapes["template"],
        matching_method="CC",
        analyzer_method="MaxScoreOverRotations",
        backend="cupy",
        ncores=ncores,
    )
    assert (with_backend + with_analyzer - base) == total


@pytest.mark.parametrize("max_workers", [1, 8])
@pytest.mark.parametrize("split_only_outer", [1, 8])
@pytest.mark.parametrize("split_axes", [None, (0, 1)])
def test_compute_parallelization_schedule(
    max_workers, split_only_outer, split_axes, test_shapes
):
    boxes, schedule = compute_schedule(
        shape=test_shapes["target"],
        padding=test_shapes["template"],
        matching_method="CC",
        max_workers=max_workers,
        max_memory=int(1e10),
        split_only_outer=split_only_outer,
        split_axes=split_axes,
    )
    if split_only_outer:
        assert schedule[1] == 1

    if split_axes is not None:
        shape1 = test_shapes["target"]
        remaining_axes = tuple(i for i in range(len(shape1)) if i not in split_axes)
        for box in boxes:
            for axis in remaining_axes:
                assert (box[axis].stop - box[axis].start) == shape1[axis]


def test_compute_parallelization_schedule_error(test_shapes):
    # Insufficient memory
    with pytest.raises(ValueError, match="No viable schedule"):
        compute_schedule(
            shape=test_shapes["target"],
            padding=test_shapes["template"],
            matching_method="CC",
            max_workers=10,
            max_memory=-1,
        )
