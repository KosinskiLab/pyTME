import numpy as np
import pytest

from scipy.ndimage import laplace

from tme.analyzer import MaxScoreOverRotations
from tme.matching_exhaustive import (
    scan,
    scan_subsets,
    MATCHING_EXHAUSTIVE_REGISTER,
    register_matching_exhaustive,
)
from tme.matching_data import MatchingData
from tme.matching_utils import get_rotation_matrices
from tme.matching_memory import MATCHING_MEMORY_REGISTRY


class TestMatchExhaustive:
    def setup_method(self):
        target = np.zeros((50, 50, 50))
        target[20:30, 30:41, 12:17] = 1

        self.target = target
        template = np.zeros((40, 40, 35))
        template[15:25, 20:31, 17:22] = 1
        self.template = template
        self.template_mask = np.ones_like(template)
        self.target_mask = np.ones_like(target)

        self.rotations = get_rotation_matrices(60)[0,]
        self.peak_position = np.array([25, 30, 12])

    def teardown_method(self):
        self.target = None
        self.template = None
        self.coordinates = None
        self.coordinates_weights = None
        self.rotations = None

    @pytest.mark.parametrize("evaluate_peak", (False, True))
    @pytest.mark.parametrize("score", list(MATCHING_EXHAUSTIVE_REGISTER.keys()))
    @pytest.mark.parametrize("n_jobs", (1, 2))
    def test_scan(self, score: str, n_jobs: int, evaluate_peak: bool):
        matching_data = MatchingData(target=self.target, template=self.template)
        matching_data.target_mask = self.target_mask
        matching_data.template_mask = self.template_mask
        matching_data.rotations = self.rotations

        callback_class = None
        if evaluate_peak:
            callback_class = MaxScoreOverRotations

        setup, process = MATCHING_EXHAUSTIVE_REGISTER[score]
        ret = scan(
            matching_data=matching_data,
            matching_setup=setup,
            matching_score=process,
            n_jobs=n_jobs,
            callback_class=callback_class,
        )
        if not evaluate_peak:
            return None

        scores = ret[0]
        peak = np.unravel_index(np.argmax(scores), scores.shape)
        assert np.allclose(peak, self.peak_position)

        theoretical_score = 1
        if score == "CC":
            theoretical_score = self.template.sum()
        elif score == "LCC":
            theoretical_score = (laplace(self.template) * laplace(self.template)).sum()

        assert np.allclose(scores[peak], theoretical_score)

    @pytest.mark.parametrize("evaluate_peak", (False, True))
    @pytest.mark.parametrize("score", list(MATCHING_EXHAUSTIVE_REGISTER.keys()))
    @pytest.mark.parametrize("job_schedule", ((1, 1), (2, 1), (2, 2)))
    @pytest.mark.parametrize("pad_fourier", (True, False))
    @pytest.mark.parametrize("pad_edge", (True, False))
    def test_scan_subset(
        self,
        score: str,
        job_schedule: int,
        evaluate_peak: bool,
        pad_fourier: bool,
        pad_edge: bool,
    ):
        matching_data = MatchingData(target=self.target, template=self.template)
        matching_data.target_mask = self.target_mask
        matching_data.template_mask = self.template_mask
        matching_data.rotations = self.rotations

        setup, process = MATCHING_EXHAUSTIVE_REGISTER[score]

        target_splits = {i: 1 if i != 0 else 2 for i in range(self.target.ndim)}

        callback_class = None
        if evaluate_peak:
            callback_class = MaxScoreOverRotations

        ret = scan_subsets(
            matching_data=matching_data,
            matching_setup=setup,
            matching_score=process,
            target_splits=target_splits,
            job_schedule=job_schedule,
            callback_class=callback_class,
            pad_target_edges=pad_edge,
            pad_fourier=pad_fourier,
        )

        if not evaluate_peak:
            return None

        scores = ret[0]

        if not pad_edge:
            # To be valid, the match needs to be fully within the target subset
            return None

        peak = np.unravel_index(np.argmax(scores), scores.shape)
        assert np.allclose(peak, self.peak_position)

        theoretical_score = 1
        if score == "CC":
            theoretical_score = self.template.sum()
        elif score == "LCC":
            theoretical_score = (laplace(self.template) * laplace(self.template)).sum()

        assert np.allclose(scores[peak], theoretical_score, rtol=0.3)

    def test_register_matching_exhaustive(self):
        setup, matching = MATCHING_EXHAUSTIVE_REGISTER[
            list(MATCHING_EXHAUSTIVE_REGISTER.keys())[0]
        ]
        memory_class = MATCHING_MEMORY_REGISTRY[
            list(MATCHING_EXHAUSTIVE_REGISTER.keys())[0]
        ]
        register_matching_exhaustive(
            matching="TEST",
            matching_setup=setup,
            matching_scoring=matching,
            memory_class=memory_class,
        )

    def test_register_matching_exhaustive_error(self):
        key = list(MATCHING_EXHAUSTIVE_REGISTER.keys())[0]
        setup, matching = MATCHING_EXHAUSTIVE_REGISTER[key]
        memory_class = MATCHING_MEMORY_REGISTRY[key]
        with pytest.raises(ValueError):
            register_matching_exhaustive(
                matching=key,
                matching_setup=setup,
                matching_scoring=matching,
                memory_class=memory_class,
            )
