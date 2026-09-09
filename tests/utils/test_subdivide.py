import numpy as np

from tme.utils.subdivide import solve_subdivide


def _saturated_cost(shapes, n_sat=1024):
    """Test cost function with an N_sat floor.

    Below n_sat voxels, FFT runtime is overhead-bound and roughly constant per
    box. The default solver cost has no floor, so it splits down to unit
    voxels; tests need a floor that matches the production `_box_cost`
    behavior at test scale.
    """
    shapes = np.asarray(shapes, dtype=np.float64)
    volumes = np.prod(shapes, axis=-1)
    effective = np.maximum(n_sat, volumes)
    return effective * np.log1p(effective)


def _make_two_blob_mask(shape=(40, 40, 40)):
    """Two disjoint blobs in opposite corners of the volume."""
    mask = np.zeros(shape, dtype=bool)
    mask[2:10, 2:10, 2:10] = True
    mask[28:38, 28:38, 28:38] = True
    return mask


def _slice_to_shape(s):
    return tuple(x.stop - x.start for x in s)


def test_solve_subdivide_two_blobs_baseline():
    """Default (equal_shape=False) splits two-blob mask into two tight boxes
    under a saturated cost function."""
    mask = _make_two_blob_mask()
    boxes, cost = solve_subdivide(mask, return_slices=True, cost_fn=_saturated_cost)
    assert len(boxes) == 2
    assert cost is not None
    # Coverage: every active voxel sits inside some box.
    coverage = np.zeros_like(mask, dtype=np.int32)
    for box in boxes:
        coverage[box] += 1
    assert (mask & (coverage > 0)).sum() == mask.sum()
    # Tightness: union of box volumes should be exactly the two 8^3 + 10^3
    # boxes (no padding under equal_shape=False).
    total = sum(int(np.prod(_slice_to_shape(b))) for b in boxes)
    assert total == 8**3 + 10**3


def test_solve_subdivide_empty_mask():
    mask = np.zeros((10, 10, 10), dtype=bool)
    boxes, cost = solve_subdivide(mask)
    assert boxes == ()
    assert cost is None


def test_solve_subdivide_equal_shape_two_blobs_siblings_match():
    """With equal_shape=True, the two sibling boxes from a single-split tree
    must share a shape. Two disjoint blobs make a single split optimal."""
    mask = _make_two_blob_mask()
    boxes, _ = solve_subdivide(
        mask, return_slices=True, equal_shape=True, cost_fn=_saturated_cost
    )
    assert len(boxes) >= 2
    shapes = {_slice_to_shape(b) for b in boxes}
    # All boxes should ultimately have one shape (the global pad in Task 4
    # enforces this). For now, after Task 2 alone, sibling pairs share a
    # shape — assert weakly: at least one pair of identical shapes exists.
    assert len(shapes) < len(boxes) or len(boxes) == 1


def test_solve_subdivide_equal_shape_all_boxes_same_shape():
    """With equal_shape=True, every output box has the same shape and
    every active voxel of the mask is covered by some box."""
    mask = _make_two_blob_mask()
    boxes, cost = solve_subdivide(
        mask, return_slices=True, equal_shape=True, cost_fn=_saturated_cost
    )
    assert cost is not None and len(boxes) >= 1

    shapes = {_slice_to_shape(b) for b in boxes}
    assert len(shapes) == 1, f"Expected one shape, got {shapes}"

    # Coverage of original mask.
    coverage = np.zeros_like(mask, dtype=np.int32)
    for box in boxes:
        coverage[box] += 1
    assert (mask & (coverage > 0)).sum() == mask.sum()

    # Every box stays inside mask bounds.
    for box in boxes:
        for s, dim in zip(box, mask.shape):
            assert 0 <= s.start
            assert s.stop <= dim


def test_solve_subdivide_equal_shape_three_blobs_asymmetric():
    """Three disjoint blobs of different sizes: shape uniformity should
    still hold, and the global shape should be the per-axis max of the
    individual blob bboxes (clipped to mask shape)."""
    mask = np.zeros((60, 60, 60), dtype=bool)
    mask[2:8, 2:8, 2:8] = True  # 6^3
    mask[20:32, 20:32, 20:32] = True  # 12^3
    mask[45:55, 45:55, 45:55] = True  # 10^3

    boxes, _ = solve_subdivide(
        mask, return_slices=True, equal_shape=True, cost_fn=_saturated_cost
    )
    shapes = {_slice_to_shape(b) for b in boxes}
    assert len(shapes) == 1
    # Global shape per axis is the max of blob extents along that axis.
    assert next(iter(shapes)) == (12, 12, 12)

    # Coverage.
    coverage = np.zeros_like(mask, dtype=np.int32)
    for box in boxes:
        coverage[box] += 1
    assert (mask & (coverage > 0)).sum() == mask.sum()


def test_solve_subdivide_equal_shape_clamped_to_mask():
    """If a blob sits flush against a mask edge and would, after equal-shape
    padding, extend past it, the global pad must re-anchor to fit."""
    mask = np.zeros((20, 20, 20), dtype=bool)
    # Small blob in the far corner of axis 1 — padded equal-shape box would
    # extend past axis-1 edge if not re-anchored.
    mask[2:6, 16:20, 2:6] = True
    mask[2:6, 2:8, 14:20] = True  # different shape so equal_shape>tight

    boxes, _ = solve_subdivide(
        mask, return_slices=True, equal_shape=True, cost_fn=_saturated_cost
    )
    for box in boxes:
        for s, dim in zip(box, mask.shape):
            assert 0 <= s.start, f"start={s.start} dim={dim}"
            assert s.stop <= dim, f"stop={s.stop} dim={dim}"


def test_solve_subdivide_equal_shape_empty_mask():
    boxes, cost = solve_subdivide(np.zeros((10, 10, 10), dtype=bool), equal_shape=True)
    assert boxes == ()
    assert cost is None


def test_solve_subdivide_equal_shape_min_box_size():
    """min_box_size still rejects splits whose equal shape falls below it."""
    mask = _make_two_blob_mask()
    # With a min larger than either blob's extent along some axis, the
    # solver must keep the root as a single box (no valid split).
    boxes, _ = solve_subdivide(
        mask,
        return_slices=True,
        equal_shape=True,
        min_box_size=20,
        cost_fn=_saturated_cost,
    )
    assert len(boxes) == 1


def test_compute_schedule_subdivide_equal_shape():
    """End-to-end: compute_schedule with mode='subdivide' and equal_shape=True
    produces a schedule where every box has the same shape."""
    from tme.memory import compute_schedule

    mask = np.zeros((64, 64, 64), dtype=bool)
    mask[4:14, 4:14, 4:14] = True
    mask[40:56, 40:56, 40:56] = True

    boxes, _ = compute_schedule(
        shape=(64, 64, 64),
        max_memory=int(1e12),
        max_workers=2,
        matching_method="CC",
        mode="subdivide",
        mask=mask,
        equal_shape=True,
        verbose=False,
    )
    shapes = {tuple(s.stop - s.start for s in box) for box in boxes}
    # compute_schedule may decide subdivide isn't worth it and fall back to
    # uniform — both paths must respect equal_shape. Either way, one shape.
    assert len(shapes) == 1, f"Expected one shape across all boxes, got {shapes}"
