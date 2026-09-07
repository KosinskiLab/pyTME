"""
Verify that batched scoring produces identical results to looped unbatched scoring
for corr_scoring, flc_scoring, and mcc_scoring across target-batched,
template-batched, and both-batched cases.
"""

import numpy as np
import pytest
from copy import deepcopy

from tme.matching_data import MatchingData
from tme.matching_scores import (
    cc_setup,
    ncc_setup,
    flc_setup,
    flcSphericalMask_setup,
    mcc_setup,
    corr_scoring,
    ncc_scoring,
    flc_scoring,
    mcc_scoring,
)


def _make_matching_data(
    target_shape,
    template_shape,
    n_rotations=3,
    target_batched=False,
    template_batched=False,
    with_target_mask=False,
    seed=42,
):
    rng = np.random.RandomState(seed)
    target = rng.rand(*target_shape).astype(np.float32)
    template = rng.rand(*template_shape).astype(np.float32)

    md = MatchingData(target=target, template=template)
    md.template_mask = np.ones_like(template)
    if with_target_mask:
        md.target_mask = np.ones_like(target)

    from tme.rotations import get_rotation_matrices

    spatial_ndim = len(target_shape) - int(target_batched)
    md.rotations = get_rotation_matrices(angular_sampling=90, dim=spatial_ndim)[
        :n_rotations
    ]

    if target_batched or template_batched:
        md.set_matching_dimension(
            target_batched=target_batched,
            template_batched=template_batched,
        )

    return md


class ScoreCollector:
    """Minimal callback that collects per-rotation scores."""

    shareable = False

    def __init__(self, **kwargs):
        self.scores = []

    def __call__(self, score, rotation_matrix=None, **kwargs):
        self.scores.append(score.copy())

    def correct_background(self, *args):
        pass

    @classmethod
    def merge(cls, callbacks, **kwargs):
        return callbacks[0]

    def result(self, *args, **kwargs):
        return self


def _run_scoring(setup_func, scoring_func, matching_data):
    """Run a scoring function on matching_data, return per-rotation scores."""
    md = deepcopy(matching_data)
    md.to_backend()

    _, fwd, inv, _ = md.fourier_padding()

    setup_data = setup_func(
        matching_data=md,
        fast_shape=fwd,
        fast_ft_shape=inv,
        shm_handler=None,
    )

    from tme.backends import backend as _be

    collector = ScoreCollector()
    scoring_func(
        **setup_data,
        fast_shape=fwd,
        fast_ft_shape=inv,
        rotations=md.rotations,
        callback=collector,
        interpolation_order=1,
        score_mask=_be.to_sharedarr(np.ones(1, dtype=np.float32), None),
        template_filter=_be.to_sharedarr(np.ones(1, dtype=np.float32), None),
    )
    return collector.scores


def _run_per_target_slice(setup_func, scoring_func, matching_data):
    """Run unbatched scoring once per target batch element, stack results."""
    n_target = matching_data._target.shape[0]
    spatial_template = matching_data._template
    has_target_mask = matching_data.target_mask is not None

    all_scores = []
    for ti in range(n_target):
        target_slice = matching_data._target[ti]
        md_slice = MatchingData(target=target_slice, template=spatial_template)
        md_slice.template_mask = np.ones_like(spatial_template)
        if has_target_mask:
            md_slice.target_mask = matching_data.target_mask[ti]
        md_slice.rotations = matching_data.rotations
        all_scores.append(_run_scoring(setup_func, scoring_func, md_slice))

    n_rot = len(all_scores[0])
    return [
        np.stack([all_scores[ti][ri] for ti in range(n_target)], axis=0)
        for ri in range(n_rot)
    ]


def _run_per_template_slice(setup_func, scoring_func, matching_data):
    """Run unbatched scoring once per template batch element, stack results."""
    n_template = matching_data._template.shape[0]
    spatial_target = matching_data._target

    all_scores = []
    for ni in range(n_template):
        template_slice = matching_data._template[ni]
        md_slice = MatchingData(target=spatial_target, template=template_slice)
        md_slice.template_mask = np.ones_like(template_slice)
        if matching_data.target_mask is not None:
            tm = matching_data.target_mask
            if tm.ndim > spatial_target.ndim:
                tm = tm[0]
            md_slice.target_mask = tm
        md_slice.rotations = matching_data.rotations
        all_scores.append(_run_scoring(setup_func, scoring_func, md_slice))

    n_rot = len(all_scores[0])
    return [
        np.stack([all_scores[ni][ri] for ni in range(n_template)], axis=0)
        for ri in range(n_rot)
    ]


def _run_per_both_slice(setup_func, scoring_func, matching_data):
    """Run unbatched scoring for each (target, template) pair, stack results."""
    n_target = matching_data._target.shape[0]
    n_template = matching_data._template.shape[0]
    has_target_mask = matching_data.target_mask is not None

    all_scores = []
    for ti in range(n_target):
        row = []
        for ni in range(n_template):
            md_slice = MatchingData(
                target=matching_data._target[ti],
                template=matching_data._template[ni],
            )
            md_slice.template_mask = np.ones_like(matching_data._template[ni])
            if has_target_mask:
                md_slice.target_mask = matching_data.target_mask[ti]
            md_slice.rotations = matching_data.rotations
            row.append(_run_scoring(setup_func, scoring_func, md_slice))
        all_scores.append(row)

    n_rot = len(all_scores[0][0])
    return [
        np.stack(
            [
                np.stack([all_scores[ti][ni][ri] for ni in range(n_template)], axis=0)
                for ti in range(n_target)
            ],
            axis=0,
        )
        for ri in range(n_rot)
    ]


def _assert_scores_match(reference, batched, squeeze=None, msg=""):
    """Compare reference (looped) scores against batched scores."""
    assert len(reference) == len(batched)
    for i, (ref, bat) in enumerate(zip(reference, batched)):
        if squeeze is not None:
            bat = bat[squeeze]
        np.testing.assert_allclose(
            ref,
            bat,
            atol=1e-4,
            rtol=1e-4,
            err_msg=f"{msg} mismatch at rotation {i}",
        )


SCORE_CONFIGS = [
    pytest.param(cc_setup, corr_scoring, False, id="corr"),
    pytest.param(flc_setup, flc_scoring, False, id="flc"),
    pytest.param(ncc_setup, ncc_scoring, False, id="ncc"),
    pytest.param(mcc_setup, mcc_scoring, True, id="mcc"),
]

BATCH_MODES = [
    pytest.param("target", id="target-batched"),
    pytest.param("template", id="template-batched"),
    pytest.param("both", id="both-batched"),
]

_BATCH_CONFIG = {
    "target": dict(
        target_shape=(3, 30, 25, 20),
        template_shape=(10, 8, 7),
        target_batched=True,
        ref_runner=_run_per_target_slice,
        squeeze=(slice(None), 0),
    ),
    "template": dict(
        target_shape=(30, 25, 20),
        template_shape=(4, 10, 8, 7),
        template_batched=True,
        ref_runner=_run_per_template_slice,
        squeeze=0,
    ),
    "both": dict(
        target_shape=(3, 30, 25, 20),
        template_shape=(4, 10, 8, 7),
        target_batched=True,
        template_batched=True,
        ref_runner=_run_per_both_slice,
        squeeze=None,
    ),
}


@pytest.mark.parametrize("setup_func, scoring_func, needs_target_mask", SCORE_CONFIGS)
@pytest.mark.parametrize("batch_mode", BATCH_MODES)
def test_batched_scoring(setup_func, scoring_func, needs_target_mask, batch_mode):
    cfg = _BATCH_CONFIG[batch_mode]
    md = _make_matching_data(
        target_shape=cfg["target_shape"],
        template_shape=cfg["template_shape"],
        n_rotations=2,
        target_batched=cfg.get("target_batched", False),
        template_batched=cfg.get("template_batched", False),
        with_target_mask=needs_target_mask,
    )
    ref = cfg["ref_runner"](setup_func, scoring_func, md)
    bat = _run_scoring(setup_func, scoring_func, md)
    _assert_scores_match(ref, bat, squeeze=cfg["squeeze"])


def test_corr_with_filter():
    md = _make_matching_data(
        target_shape=(30, 25, 20),
        template_shape=(10, 8, 7),
        n_rotations=2,
    )
    scores = _run_scoring(flcSphericalMask_setup, corr_scoring, md)
    assert all(np.isfinite(s).all() for s in scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
