from tempfile import mkstemp

import pytest
import numpy as np

from tme.backends import backend as be
from tme.matching_data import MatchingData


class TestDensity:
    def setup_method(self):
        target = np.zeros((50, 50, 50))
        target[20:30, 30:40, 12:17] = 1

        self.target = target
        template = np.zeros((50, 50, 50))
        template[15:25, 20:30, 2:7] = 1
        self.template = template
        self.rotations = np.random.rand(100, target.ndim, target.ndim).astype(
            np.float32
        )

    def teardown_method(self):
        self.target = None
        self.template = None
        self.coordinates = None
        self.coordinates_weights = None
        self.rotations = None

    def test_initialization(self):
        _ = MatchingData(target=self.target, template=self.template)

    @pytest.mark.parametrize("shape", [(10,), (10, 15), (10, 20, 30)])
    def test__shape_to_slice(self, shape):
        slices = MatchingData._shape_to_slice(shape=shape)
        assert len(slices) == len(shape)
        for i, k in enumerate(shape):
            assert slices[i].start == 0
            assert slices[i].stop == k

    @pytest.mark.parametrize("shape", [(10,), (10, 15), (10, 20, 30)])
    def test_slice_to_mesh(self, shape):
        if shape is not None:
            slices = MatchingData._shape_to_slice(shape=shape)

        indices = MatchingData._slice_to_mesh(slice_variable=slices, shape=shape)
        assert len(indices) == len(shape)
        for i, k in enumerate(shape):
            assert indices[i].min() == 0
            assert indices[i].max() == k - 1

        indices = MatchingData._slice_to_mesh(slice_variable=None, shape=shape)
        assert len(indices) == len(shape)
        for i, k in enumerate(shape):
            assert indices[i].min() == 0
            assert indices[i].max() == k - 1

    def test__load_array(self):
        arr = MatchingData._load_array(self.target)
        assert np.allclose(arr, self.target)

    def test__load_array_memmap(self):
        _, filename = mkstemp()
        shape, dtype = self.target.shape, self.target.dtype
        arr_memmap = np.memmap(filename, mode="w+", dtype=dtype, shape=shape)
        arr_memmap[:] = self.target[:]
        arr_memmap.flush()

        arr = MatchingData._load_array(arr_memmap)
        assert np.allclose(arr, self.target)

    def test_subset_array(self):
        matching_data = MatchingData(target=self.target, template=self.template)
        slices = MatchingData._shape_to_slice(
            shape=np.divide(self.target.shape, 2).astype(int)
        )
        ret = matching_data.subset_array(
            arr=self.target, arr_slice=slices, padding=(2, 2, 2)
        )
        assert np.allclose(
            ret.shape, np.add(np.divide(self.target.shape, 2).astype(int), 2)
        )

    def test_subset_by_slice_none(self):
        matching_data = MatchingData(target=self.target, template=self.template)
        matching_data.rotations = self.rotations
        matching_data.target_mask = self.target
        matching_data.template_mask = self.template

        ret = matching_data.subset_by_slice()

        assert type(ret) == type(matching_data)
        assert np.allclose(ret.target, matching_data.target)
        assert np.allclose(ret.template, matching_data.template)
        assert np.allclose(ret.target_mask, matching_data.target_mask)
        assert np.allclose(ret.template_mask, matching_data.template_mask)

    def test_subset_by_slice(self):
        matching_data = MatchingData(target=self.target, template=self.template)
        matching_data.rotations = self.rotations
        matching_data.target_mask = self.target
        matching_data.template_mask = self.template

        target_slice = MatchingData._shape_to_slice(
            shape=np.divide(self.target.shape, 2).astype(int)
        )
        template_slice = MatchingData._shape_to_slice(
            shape=np.divide(self.template.shape, 2).astype(int)
        )
        ret = matching_data.subset_by_slice(
            target_slice=target_slice, template_slice=template_slice
        )
        assert type(ret) == type(matching_data)

        assert np.allclose(
            ret.target.shape, np.divide(self.target.shape, 2).astype(int)
        )
        assert np.allclose(
            ret.template.shape, np.divide(self.target.shape, 2).astype(int)[::-1]
        )

    def test_rotations(self):
        matching_data = MatchingData(target=self.target, template=self.template)
        matching_data.rotations = self.rotations
        matching_data.target_mask = self.target

        assert np.allclose(matching_data.rotations, self.rotations)

        matching_data.rotations = np.random.rand(self.target.ndim, self.target.ndim)
        assert np.allclose(
            matching_data.rotations.shape, (1, self.target.ndim, self.target.ndim)
        )

    def test_target(self):
        matching_data = MatchingData(target=self.target, template=self.template)

        assert np.allclose(matching_data.target, self.target)

    def test_template(self):
        matching_data = MatchingData(target=self.target, template=self.template)

        assert np.allclose(matching_data.template, be.reverse(self.template))

    def test_target_mask(self):
        matching_data = MatchingData(target=self.target, template=self.template)
        matching_data.target_mask = self.target

        assert np.allclose(matching_data.target_mask, self.target)

    def test_template_mask(self):
        matching_data = MatchingData(target=self.target, template=self.template)
        matching_data.template_mask = self.template

        assert np.allclose(matching_data.template_mask, be.reverse(self.template))

    @pytest.mark.parametrize("jobs", range(1, 50, 5))
    def test__split_rotations_on_jobs(self, jobs):
        matching_data = MatchingData(target=self.target, template=self.template)
        matching_data.rotations = self.rotations

        ret = matching_data._split_rotations_on_jobs(n_jobs=jobs)
        assert len(ret) == jobs
