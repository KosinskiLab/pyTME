import numpy as np
import pytest

from scipy.ndimage import laplace

from tme.matching_data import MatchingData
from tme.memory import MATCHING_MEMORY_REGISTRY
from tme.analyzer import MaxScoreOverRotations, PeakCallerSort
from tme.matching_exhaustive import (
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

        self.rotations = np.eye(3)
        self.peak_position = np.array([25, 17, 12])

    def teardown_method(self):
        self.target = None
        self.template = None
        self.coordinates = None
        self.coordinates_weights = None
        self.rotations = None

    @pytest.mark.parametrize("evaluate_peak", (True,))
    @pytest.mark.parametrize("score", tuple(MATCHING_EXHAUSTIVE_REGISTER.keys()))
    @pytest.mark.parametrize("job_schedule", ((2, 1),))
    @pytest.mark.parametrize("pad_edge", (False, True))
    def test_scan_subset(
        self,
        score: str,
        job_schedule: int,
        evaluate_peak: bool,
        pad_edge: bool,
    ):
        matching_data = MatchingData(
            target=self.target.copy(),
            template=self.template.copy(),
            target_mask=self.target_mask.copy(),
            template_mask=self.template_mask.copy(),
            rotations=self.rotations,
        )

        setup, process = MATCHING_EXHAUSTIVE_REGISTER[score]

        target_splits = {}
        if job_schedule[0] == 2:
            target_splits = {0: 2}

        callback_class = MaxScoreOverRotations
        if evaluate_peak:
            callback_class = PeakCallerSort

        ret = scan_subsets(
            matching_data=matching_data,
            matching_setup=setup,
            matching_score=process,
            target_splits=target_splits,
            job_schedule=job_schedule,
            callback_class=callback_class,
            pad_target_edges=pad_edge,
        )

        if not evaluate_peak:
            scores = ret[0]
            peak = np.unravel_index(np.argmax(scores), scores.shape)
            achieved_score = scores[tuple(peak)]
        else:
            try:
                peak, achieved_score = ret[0][0], ret[2][0]
            except Exception:
                return None

        theoretical_score = 1
        if score == "CC":
            theoretical_score = self.template.sum()
        elif score == "LCC":
            theoretical_score = (laplace(self.template) * laplace(self.template)).sum()

        if not np.allclose(peak, self.peak_position):
            print(peak, self.peak_position)
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
