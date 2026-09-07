"""
Evaluation utilities for comparing predicted peaks against ground truth.

Copyright (c) 2026 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.stats import mannwhitneyu


def match_peaks(
    predicted_translations,
    predicted_scores,
    gt_translations,
    distance_threshold,
    angle_threshold=None,
    predicted_rotations=None,
    gt_rotations=None,
):
    """Match predicted peaks to ground truth via greedy nearest-neighbor.

    Peaks are processed in descending score order. Each predicted peak is
    assigned to the closest unmatched ground truth point within
    *distance_threshold*. Optionally, an angular threshold is also enforced.

    Parameters
    ----------
    predicted_translations : ndarray, shape (N, d)
        Predicted peak positions.
    predicted_scores : ndarray, shape (N,)
        Score per predicted peak (higher = better).
    gt_translations : ndarray, shape (M, d)
        Ground truth positions.
    distance_threshold : float
        Maximum distance for a match.
    angle_threshold : float, optional
        Maximum angular error in degrees for a match.
    predicted_rotations : ndarray, optional, shape (N, d, d)
        Predicted rotation matrices.
    gt_rotations : ndarray, optional, shape (M, d, d)
        Ground truth rotation matrices.

    Returns
    -------
    dict
        tp_mask : bool ndarray (N,) — True for true positives.
        distances : float ndarray (N,) — distance to nearest GT.
        gt_indices : int ndarray (N,) — matched GT index (-1 if FP).
        angle_errors : float ndarray (N,) — angular error (NaN if unavailable).
        gt_matched : bool ndarray (M,) — True for matched GT points.
    """
    predicted_translations = np.asarray(predicted_translations)
    predicted_scores = np.asarray(predicted_scores)
    gt_translations = np.asarray(gt_translations)

    n_pred = len(predicted_translations)
    n_gt = len(gt_translations)

    tp_mask = np.zeros(n_pred, dtype=bool)
    distances = np.full(n_pred, np.inf)
    gt_indices = np.full(n_pred, -1, dtype=int)
    angle_errors = np.full(n_pred, np.nan)
    gt_matched = np.zeros(n_gt, dtype=bool)

    if n_pred == 0 or n_gt == 0:
        return {
            "tp_mask": tp_mask,
            "distances": distances,
            "gt_indices": gt_indices,
            "angle_errors": angle_errors,
            "gt_matched": gt_matched,
        }

    tree = KDTree(gt_translations)
    k = min(n_gt, 10)
    all_dists, all_idxs = tree.query(predicted_translations, k=k)
    if k == 1:
        all_dists = all_dists[:, None]
        all_idxs = all_idxs[:, None]

    check_angles = (
        angle_threshold is not None
        and predicted_rotations is not None
        and gt_rotations is not None
    )

    order = np.argsort(-predicted_scores)
    for i in order:
        distances[i] = all_dists[i, 0]

        for j in range(k):
            d = all_dists[i, j]
            gt_idx = all_idxs[i, j]

            if d > distance_threshold:
                break
            if gt_matched[gt_idx]:
                continue

            if check_angles:
                angle_err = _rotation_error(
                    predicted_rotations[i], gt_rotations[gt_idx]
                )
                angle_errors[i] = angle_err
                if angle_err > angle_threshold:
                    continue

            tp_mask[i] = True
            gt_indices[i] = gt_idx
            gt_matched[gt_idx] = True
            break

    return {
        "tp_mask": tp_mask,
        "distances": distances,
        "gt_indices": gt_indices,
        "angle_errors": angle_errors,
        "gt_matched": gt_matched,
    }


def _rotation_error(r1, r2):
    """Compute angular distance between two rotation matrices in degrees."""
    r_diff = r1 @ r2.T
    trace = np.clip(np.trace(r_diff), -1.0, 3.0)
    angle = np.arccos((trace - 1.0) / 2.0)
    return float(np.degrees(angle))


def compute_f1_sweep(scores, tp_mask, n_ground_truth):
    """Sweep score thresholds to find optimal F1.

    Parameters
    ----------
    scores : ndarray (N,)
        Predicted scores.
    tp_mask : bool ndarray (N,)
        True for true positives.
    n_ground_truth : int
        Total number of ground truth positives.

    Returns
    -------
    dict
        best_f1, best_threshold, precision_at_best, recall_at_best,
        precisions, recalls, f1s, thresholds (arrays for the full curve).
    """
    scores = np.asarray(scores)
    tp_mask = np.asarray(tp_mask)

    if n_ground_truth == 0 or len(scores) == 0:
        return {
            "best_f1": 0.0,
            "best_threshold": 0.0,
            "precision_at_best": 0.0,
            "recall_at_best": 0.0,
        }

    order = np.argsort(-scores)
    sorted_tp = tp_mask[order]

    tp_cumsum = np.cumsum(sorted_tp)
    n_predicted = np.arange(1, len(sorted_tp) + 1)

    precisions = tp_cumsum / n_predicted
    recalls = tp_cumsum / n_ground_truth
    denom = precisions + recalls
    f1s = np.where(denom > 0, 2 * precisions * recalls / denom, 0.0)

    best_idx = np.argmax(f1s)

    return {
        "best_f1": float(f1s[best_idx]),
        "best_threshold": float(scores[order[best_idx]]),
        "precision_at_best": float(precisions[best_idx]),
        "recall_at_best": float(recalls[best_idx]),
        "precisions": precisions,
        "recalls": recalls,
        "f1s": f1s,
        "thresholds": scores[order],
    }


def compute_metrics(
    predicted_translations,
    predicted_scores,
    gt_translations,
    distance_threshold,
    angle_threshold=None,
    predicted_rotations=None,
    gt_rotations=None,
):
    """Match peaks and compute summary metrics.

    Returns
    -------
    dict
        Matching results plus F1/precision/recall summary.
    """
    matching = match_peaks(
        predicted_translations=predicted_translations,
        predicted_scores=predicted_scores,
        gt_translations=gt_translations,
        distance_threshold=distance_threshold,
        angle_threshold=angle_threshold,
        predicted_rotations=predicted_rotations,
        gt_rotations=gt_rotations,
    )

    f1_result = compute_f1_sweep(
        scores=predicted_scores,
        tp_mask=matching["tp_mask"],
        n_ground_truth=len(gt_translations),
    )

    n_tp = int(matching["tp_mask"].sum())
    n_fp = int((~matching["tp_mask"]).sum())
    n_gt_matched = int(matching["gt_matched"].sum())
    n_gt_missed = len(gt_translations) - n_gt_matched

    return {
        "f1": f1_result["best_f1"],
        "threshold": f1_result["best_threshold"],
        "precision": f1_result["precision_at_best"],
        "recall": f1_result["recall_at_best"],
        "n_predicted": len(predicted_translations),
        "n_ground_truth": len(gt_translations),
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_gt_matched": n_gt_matched,
        "n_gt_missed": n_gt_missed,
        "matching": matching,
        "f1_curve": f1_result,
    }


def compute_best_f1(scores, labels):
    """Compute best F1 score by sweeping thresholds."""
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tp = np.cumsum(sorted_labels == 1)
    n_positive = np.arange(1, len(sorted_labels) + 1)
    total_positive = np.sum(labels == 1)

    if total_positive == 0:
        return 0.0, 0.0

    precision = tp / n_positive
    recall = tp / total_positive
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    best_idx = np.argmax(f1)
    return float(f1[best_idx]), float(scores[order[best_idx]])


def compute_mann_whitney(scores, labels):
    """Compute normalized Mann-Whitney U statistic in [0, 1]."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0

    u_stat, _ = mannwhitneyu(pos_scores, neg_scores, alternative="greater")
    return u_stat / (len(pos_scores) * len(neg_scores))


def compute_pairwise_logistic(scores, labels):
    """Compute negative mean pairwise logistic loss.

    For each (positive, negative) pair, computes log(1 + exp(-(s_pos - s_neg))).
    Minimizing this loss is equivalent to maximizing AUC-ROC.
    """
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0

    diffs = pos_scores[:, None] - neg_scores[None, :]
    loss = np.log1p(np.exp(-diffs)).mean()
    return -loss


OBJECTIVE_FUNCTIONS = {
    "f1": lambda scores, labels: compute_best_f1(scores, labels)[0],
    "mann_whitney": compute_mann_whitney,
    "pairwise_logistic": compute_pairwise_logistic,
}
