import numpy as np
import pytest

from tme.utils.evaluation import (
    match_peaks,
    compute_f1_sweep,
    compute_metrics,
    compute_best_f1,
    compute_mann_whitney,
    compute_pairwise_logistic,
    OBJECTIVE_FUNCTIONS,
)


@pytest.fixture
def simple_data():
    """5 predicted peaks, 3 ground truth. Peaks 0-2 are close to GT, 3-4 are far."""
    gt = np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]], dtype=float)
    pred = np.array(
        [[10.5, 10, 10], [20, 20.5, 20], [30, 30, 30.5], [50, 50, 50], [60, 60, 60]],
        dtype=float,
    )
    scores = np.array([0.9, 0.8, 0.7, 0.5, 0.3])
    return pred, scores, gt


def test_match_peaks_basic(simple_data):
    pred, scores, gt = simple_data
    result = match_peaks(pred, scores, gt, distance_threshold=2.0)

    assert result["tp_mask"].sum() == 3
    assert (~result["tp_mask"][:3]).sum() == 0
    assert result["tp_mask"][3] == False
    assert result["tp_mask"][4] == False
    assert result["gt_matched"].sum() == 3


def test_match_peaks_tight_threshold(simple_data):
    pred, scores, gt = simple_data
    result = match_peaks(pred, scores, gt, distance_threshold=0.1)

    # All predictions are ~0.5 away from GT, threshold 0.1 should match none
    assert result["tp_mask"].sum() == 0
    assert result["gt_matched"].sum() == 0


def test_match_peaks_greedy_order():
    """Two predictions close to the same GT point — higher score should win."""
    gt = np.array([[10, 10, 10]], dtype=float)
    pred = np.array([[10.1, 10, 10], [10.2, 10, 10]], dtype=float)
    scores = np.array([0.5, 0.9])

    result = match_peaks(pred, scores, gt, distance_threshold=1.0)

    # Peak 1 (score 0.9) should match, peak 0 (score 0.5) should not
    assert result["tp_mask"][0] == False
    assert result["tp_mask"][1] == True


def test_match_peaks_fallback_neighbor():
    """When nearest GT is claimed, should fall back to next-nearest within threshold."""
    gt = np.array([[10, 10, 10], [12, 10, 10]], dtype=float)
    pred = np.array([[11, 10, 10], [10, 10, 10]], dtype=float)
    scores = np.array([0.9, 0.8])

    result = match_peaks(pred, scores, gt, distance_threshold=3.0)

    # Pred 0 (score 0.9) claims GT 0 (nearest at dist 1).
    # Pred 1 (score 0.8) nearest is GT 0 (claimed), falls back to GT 1 (dist 2).
    assert result["tp_mask"].sum() == 2
    assert result["gt_matched"].sum() == 2


def test_match_peaks_empty():
    result = match_peaks(
        np.zeros((0, 3)), np.array([]), np.array([[1, 1, 1]]), distance_threshold=5.0
    )
    assert result["tp_mask"].shape == (0,)
    assert result["gt_matched"].sum() == 0


def test_match_peaks_with_angles():
    gt = np.array([[10, 10, 10]], dtype=float)
    pred = np.array([[10, 10, 10]], dtype=float)
    scores = np.array([1.0])

    # Identical rotations → 0 error
    rot = np.eye(3)
    result = match_peaks(
        pred,
        scores,
        gt,
        distance_threshold=1.0,
        angle_threshold=5.0,
        predicted_rotations=rot[None],
        gt_rotations=rot[None],
    )
    assert result["tp_mask"][0] == True
    assert result["angle_errors"][0] == pytest.approx(0.0, abs=1e-6)

    # 90 degree rotation → should fail with 5 degree threshold
    rot_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    result = match_peaks(
        pred,
        scores,
        gt,
        distance_threshold=1.0,
        angle_threshold=5.0,
        predicted_rotations=rot[None],
        gt_rotations=rot_90[None],
    )
    assert result["tp_mask"][0] == False


def test_compute_f1_sweep_perfect():
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    tp_mask = np.array([True, True, True, False, False])
    result = compute_f1_sweep(scores, tp_mask, n_ground_truth=3)

    assert result["best_f1"] == pytest.approx(1.0)
    assert result["recall_at_best"] == pytest.approx(1.0)
    assert result["precision_at_best"] == pytest.approx(1.0)
    assert result["best_threshold"] >= 0.7


def test_compute_f1_sweep_partial():
    # 2 TP at top, 1 FP in between, 1 TP at bottom
    scores = np.array([0.9, 0.8, 0.6, 0.5])
    tp_mask = np.array([True, True, False, True])
    result = compute_f1_sweep(scores, tp_mask, n_ground_truth=4)

    assert 0 < result["best_f1"] < 1.0


def test_compute_f1_sweep_empty():
    result = compute_f1_sweep(np.array([]), np.array([], dtype=bool), n_ground_truth=0)
    assert result["best_f1"] == 0.0


def test_compute_metrics(simple_data):
    pred, scores, gt = simple_data
    result = compute_metrics(pred, scores, gt, distance_threshold=2.0)

    assert result["n_tp"] == 3
    assert result["n_fp"] == 2
    assert result["n_ground_truth"] == 3
    assert result["n_gt_matched"] == 3
    assert result["n_gt_missed"] == 0
    assert result["f1"] > 0


# --- Objective functions ---


def test_compute_best_f1():
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    labels = np.array([1, 1, 0, 0], dtype=float)
    f1, threshold = compute_best_f1(scores, labels)
    assert f1 == pytest.approx(1.0)
    assert threshold >= 0.3


def test_compute_best_f1_no_positives():
    f1, thresh = compute_best_f1(np.array([0.5, 0.3]), np.array([0, 0], dtype=float))
    assert f1 == 0.0


def test_compute_mann_whitney():
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    labels = np.array([1, 1, 0, 0], dtype=float)
    u = compute_mann_whitney(scores, labels)
    assert u == pytest.approx(1.0)


def test_compute_mann_whitney_equal():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 1, 0, 0], dtype=float)
    u = compute_mann_whitney(scores, labels)
    assert 0 <= u <= 1


def test_compute_pairwise_logistic():
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    labels = np.array([1, 1, 0, 0], dtype=float)
    val = compute_pairwise_logistic(scores, labels)
    assert val < 0  # Negative loss


def test_objective_functions_keys():
    assert set(OBJECTIVE_FUNCTIONS.keys()) == {
        "f1",
        "mann_whitney",
        "pairwise_logistic",
    }
    for func in OBJECTIVE_FUNCTIONS.values():
        scores = np.array([0.9, 0.3])
        labels = np.array([1, 0], dtype=float)
        result = func(scores, labels)
        assert isinstance(result, float)
