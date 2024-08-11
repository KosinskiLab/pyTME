import numpy as np
import pytest

from scipy.ndimage import laplace

from tme.matching_data import MatchingData
from tme.memory import MATCHING_MEMORY_REGISTRY
from tme.matching_utils import get_rotation_matrices
from tme.analyzer import MaxScoreOverRotations, PeakCallerSort
from tme.matching_exhaustive import (
    scan,
    scan_subsets,
    MATCHING_EXHAUSTIVE_REGISTER,
    register_matching_exhaustive,
)


class TestMatchExhaustive:
    def setup_method(self):
        # To be valid for splitting, the template needs to be fully inside the object
        target = np.zeros((80, 80, 80))
        target[25:31, 22:28, 12:16] = 1

        self.target = target
        self.template = np.zeros((41, 41, 35))
        self.template[20:26, 25:31, 17:21] = 1
        self.template_mask = np.ones_like(self.template)
        self.target_mask = np.ones_like(target)

        self.rotations = get_rotation_matrices(60)[0,]
        self.peak_position = np.array([25, 17, 12])

    def teardown_method(self):
        self.target = None
        self.template = None
        self.coordinates = None
        self.coordinates_weights = None
        self.rotations = None

    @pytest.mark.parametrize("score", list(MATCHING_EXHAUSTIVE_REGISTER.keys()))
    @pytest.mark.parametrize("n_jobs", (1, 2))
    def test_scan(self, score: str, n_jobs: int):
        matching_data = MatchingData(
            target=self.target,
            template=self.template,
            target_mask=self.target_mask,
            template_mask=self.template_mask,
            rotations=self.rotations,
        )
        setup, process = MATCHING_EXHAUSTIVE_REGISTER[score]
        ret = scan(
            matching_data=matching_data,
            matching_setup=setup,
            matching_score=process,
            n_jobs=n_jobs,
            callback_class=MaxScoreOverRotations,
        )
        scores = ret[0]
        peak = np.unravel_index(np.argmax(scores), scores.shape)

        theoretical_score = 1
        if score == "CC":
            theoretical_score = self.template.sum()
        elif score == "LCC":
            theoretical_score = (laplace(self.template) * laplace(self.template)).sum()

        assert np.allclose(peak, self.peak_position)
        assert np.allclose(scores[peak], theoretical_score, rtol=0.05)

    @pytest.mark.parametrize("evaluate_peak", (False, True))
    @pytest.mark.parametrize("score", tuple(MATCHING_EXHAUSTIVE_REGISTER.keys()))
    @pytest.mark.parametrize("job_schedule", ((2, 1), (1, 1)))
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
        matching_data = MatchingData(
            target=self.target,
            template=self.template,
            target_mask=self.target_mask,
            template_mask=self.template_mask,
            rotations=self.rotations,
        )

        setup, process = MATCHING_EXHAUSTIVE_REGISTER[score]

        target_splits = {}
        if job_schedule[0] == 2:
            target_splits = {0: 2 if i != 0 else 2 for i in range(self.target.ndim)}

        callback_class = PeakCallerSort
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

        if evaluate_peak:
            scores = ret[0]
            peak = np.unravel_index(np.argmax(scores), scores.shape)
            achieved_score = scores[tuple(peak)]

        else:
            peak, achieved_score = ret[0][0], ret[2][0]

        if not pad_edge:
            # To be valid, the match needs to be fully within the target subset
            return None

        theoretical_score = 1
        if score == "CC":
            theoretical_score = self.template.sum()
        elif score == "LCC":
            theoretical_score = (laplace(self.template) * laplace(self.template)).sum()

        if not np.allclose(peak, self.peak_position):
            print(peak)
            assert False

        assert np.allclose(achieved_score, theoretical_score, rtol=0.3)

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
