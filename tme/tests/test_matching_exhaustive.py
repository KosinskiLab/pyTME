import numpy as np
import pytest

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
        target[20:30, 30:40, 12:17] = 1

        self.target = target
        template = np.zeros((50, 50, 50))
        template[15:25, 20:30, 2:7] = 1
        self.template = template
        self.rotations = get_rotation_matrices(60)[0:2,]

    def teardown_method(self):
        self.target = None
        self.template = None
        self.coordinates = None
        self.coordinates_weights = None
        self.rotations = None

    @pytest.mark.parametrize("score", list(MATCHING_EXHAUSTIVE_REGISTER.keys()))
    def test_scan_single_core(self, score):
        matching_data = MatchingData(target=self.target, template=self.template)
        matching_data.target_mask = self.target
        matching_data.template_mask = self.template
        matching_data.rotations = self.rotations

        setup, process = MATCHING_EXHAUSTIVE_REGISTER[score]
        scan(matching_data=matching_data, matching_setup=setup, matching_score=process)

    @pytest.mark.parametrize("score", list(MATCHING_EXHAUSTIVE_REGISTER.keys()))
    def test_scan_single_multi_core(self, score):
        matching_data = MatchingData(target=self.target, template=self.template)
        matching_data.target_mask = self.target
        matching_data.template_mask = self.template
        matching_data.rotations = self.rotations

        setup, process = MATCHING_EXHAUSTIVE_REGISTER[score]
        scan(
            matching_data=matching_data,
            matching_setup=setup,
            matching_score=process,
            n_jobs=2,
        )

    @pytest.mark.parametrize("score", list(MATCHING_EXHAUSTIVE_REGISTER.keys()))
    def test_scan_subsets_single_core(self, score):
        matching_data = MatchingData(target=self.target, template=self.template)
        matching_data.target_mask = self.target
        matching_data.template_mask = self.template
        matching_data.rotations = self.rotations

        setup, process = MATCHING_EXHAUSTIVE_REGISTER[score]

        target_splits = {i: 1 for i in range(self.target.ndim)}
        template_splits = {i: 1 for i in range(self.target.ndim)}
        target_splits[0], template_splits[1] = 2, 2
        scan_subsets(
            matching_data=matching_data,
            matching_setup=setup,
            matching_score=process,
            target_splits=target_splits,
            template_splits=template_splits,
            job_schedule=(2, 1),
        )

    @pytest.mark.parametrize("score", list(MATCHING_EXHAUSTIVE_REGISTER.keys()))
    def test_scan_subsets_single_multi_core(self, score):
        matching_data = MatchingData(target=self.target, template=self.template)
        matching_data.target_mask = self.target
        matching_data.template_mask = self.template
        matching_data.rotations = self.rotations

        setup, process = MATCHING_EXHAUSTIVE_REGISTER[score]

        target_splits = {i: 1 for i in range(self.target.ndim)}
        template_splits = {i: 1 for i in range(self.target.ndim)}
        target_splits[0], template_splits[1] = 2, 2

        scan_subsets(
            matching_data=matching_data,
            matching_setup=setup,
            matching_score=process,
            target_splits=target_splits,
            template_splits=template_splits,
            job_schedule=(2, 1),
        )

    @pytest.mark.parametrize("score", list(MATCHING_EXHAUSTIVE_REGISTER.keys()))
    def test_scan_subsets_single_multi_core_both(self, score):
        matching_data = MatchingData(target=self.target, template=self.template)
        matching_data.target_mask = self.target
        matching_data.template_mask = self.template
        matching_data.rotations = self.rotations

        setup, process = MATCHING_EXHAUSTIVE_REGISTER[score]

        target_splits = {i: 1 for i in range(self.target.ndim)}
        template_splits = {i: 1 for i in range(self.target.ndim)}
        target_splits[0], template_splits[1] = 2, 2

        scan_subsets(
            matching_data=matching_data,
            matching_setup=setup,
            matching_score=process,
            target_splits=target_splits,
            template_splits=template_splits,
            job_schedule=(2, 2),
        )

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
