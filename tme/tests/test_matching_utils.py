import sys
from tempfile import mkstemp
from itertools import combinations, chain, product

import pytest
import numpy as np
from scipy.signal import correlate
from scipy.spatial.transform import Rotation

from tme import Density
from tme.backends import backend as be
from tme.matching_utils import (
    compute_parallelization_schedule,
    elliptical_mask,
    box_mask,
    tube_mask,
    create_mask,
    scramble_phases,
    split_shape,
    compute_full_convolution_index,
    apply_convolution_mode,
    get_rotation_matrices,
    write_pickle,
    load_pickle,
    euler_from_rotationmatrix,
    euler_to_rotationmatrix,
    get_rotations_around_vector,
    rotation_aligning_vectors,
    _normalize_template_overflow_safe,
)
from tme.memory import MATCHING_MEMORY_REGISTRY
from tme.matching_exhaustive import _handle_traceback


class TestMatchingUtils:
    def setup_method(self):
        self.density = Density.from_file(filename="./tme/tests/data/Raw/em_map.map")
        self.structure_density = Density.from_structure(
            filename_or_structure="./tme/tests/data/Structures/5khe.cif",
            origin=self.density.origin,
            shape=self.density.shape,
            sampling_rate=self.density.sampling_rate,
        )

    @pytest.mark.parametrize("matching_method", list(MATCHING_MEMORY_REGISTRY.keys()))
    @pytest.mark.parametrize("max_cores", range(1, 10, 3))
    @pytest.mark.parametrize("max_ram", [1e5, 1e7, 1e9])
    def test_compute_parallelization_schedule(
        self, matching_method, max_cores, max_ram
    ):
        max_cores, max_ram = int(max_cores), int(max_ram)
        compute_parallelization_schedule(
            self.density.shape,
            self.structure_density.shape,
            matching_method=matching_method,
            max_cores=max_cores,
            max_ram=max_ram,
            max_splits=256,
        )

    def test_create_mask(self):
        create_mask(
            mask_type="ellipse",
            shape=self.density.shape,
            radius=5,
            center=np.divide(self.density.shape, 2),
        )

    def test_create_mask_error(self):
        with pytest.raises(ValueError):
            create_mask(mask_type=None)

    def test_elliptical_mask(self):
        elliptical_mask(
            shape=self.density.shape,
            radius=5,
            center=np.divide(self.density.shape, 2),
        )

    def test_box_mask(self):
        box_mask(
            shape=self.density.shape,
            height=[5, 10, 20],
            center=np.divide(self.density.shape, 2),
        )

    def test_tube_mask(self):
        tube_mask(
            shape=self.density.shape,
            outer_radius=10,
            inner_radius=5,
            height=5,
            base_center=np.divide(self.density.shape, 2),
            symmetry_axis=1,
        )

    def test_tube_mask_error(self):
        with pytest.raises(ValueError):
            tube_mask(
                shape=self.density.shape,
                outer_radius=5,
                inner_radius=10,
                height=5,
                base_center=np.divide(self.density.shape, 2),
                symmetry_axis=1,
            )

        with pytest.raises(ValueError):
            tube_mask(
                shape=self.density.shape,
                outer_radius=5,
                inner_radius=10,
                height=10 * np.max(self.density.shape),
                base_center=np.divide(self.density.shape, 2),
                symmetry_axis=1,
            )

        with pytest.raises(ValueError):
            tube_mask(
                shape=self.density.shape,
                outer_radius=5,
                inner_radius=10,
                height=10 * np.max(self.density.shape),
                base_center=np.divide(self.density.shape, 2),
                symmetry_axis=len(self.density.shape) + 1,
            )

    def test_scramble_phases(self):
        scramble_phases(arr=self.density.data, noise_proportion=0.5)

    @pytest.mark.parametrize("dim", range(1, 3, 5))
    @pytest.mark.parametrize("angular_sampling", [10, 15, 20])
    def test_get_rotation_matrices(self, dim, angular_sampling):
        rotation_matrices = get_rotation_matrices(
            angular_sampling=angular_sampling, dim=dim
        )
        assert np.allclose(rotation_matrices[0] @ rotation_matrices[0].T, np.eye(dim))

    def test_split_correlation(self):
        arr1 = elliptical_mask(shape=(50, 51), center=(20, 30), radius=5)

        arr2 = elliptical_mask(shape=(41, 36), center=(25, 20), radius=5)
        s = range(arr1.ndim)
        outer_split = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

        s = range(arr2.ndim)
        inner_split = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

        outer_splits = [dict(zip(i, [2] * len(i))) for i in list(outer_split)]
        inner_splits = [dict(zip(i, [2] * len(i))) for i in list(inner_split)]

        for outer_split, inner_split in product(outer_splits, inner_splits):
            splits1 = split_shape(
                shape=arr1.shape, splits=outer_split, equal_shape=False
            )
            splits2 = split_shape(
                shape=arr2.shape, splits=inner_split, equal_shape=False
            )

            full = correlate(arr1, arr2, method="direct", mode="full")
            temp = np.zeros_like(full)

            for arr1_split, arr2_split in product(splits1, splits2):
                correlation = correlate(
                    arr1[arr1_split], arr2[arr2_split], method="direct", mode="full"
                )
                score_slice = compute_full_convolution_index(
                    arr1.shape, arr2.shape, arr1_split, arr2_split
                )
                temp[score_slice] += correlation

            assert np.allclose(temp, full)

    @pytest.mark.parametrize("convolution_mode", ["full", "valid", "same"])
    def test_apply_convolution_mode(self, convolution_mode):
        correlation = correlate(
            self.density.data, self.structure_density.data, method="direct", mode="full"
        )
        ret = apply_convolution_mode(
            arr=correlation,
            convolution_mode=convolution_mode,
            s1=self.density.shape,
            s2=self.structure_density.shape,
        )
        if convolution_mode == "full":
            expected_size = correlation.shape
        elif convolution_mode == "same":
            expected_size = self.density.shape
        else:
            expected_size = np.subtract(
                self.density.shape, self.structure_density.shape
            )
            expected_size += np.mod(self.structure_density.shape, 2)
        assert np.allclose(ret.shape, expected_size)

    def test_apply_convolution_mode_error(self):
        correlation = correlate(
            self.density.data, self.structure_density.data, method="direct", mode="full"
        )
        with pytest.raises(ValueError):
            _ = apply_convolution_mode(
                arr=correlation,
                convolution_mode=None,
                s1=self.density.shape,
                s2=self.structure_density.shape,
            )

    def test_handle_traceback(self):
        try:
            raise ValueError("Test error")
        except Exception:
            type_, value_, traceback_ = sys.exc_info()
            with pytest.raises(Exception, match="Test error"):
                _handle_traceback(type_, value_, traceback_)

    def test_pickle_io(self):
        _, filename = mkstemp()

        data = ["Hello", 123, np.array([1, 2, 3])]
        write_pickle(data=data, filename=filename)
        loaded_data = load_pickle(filename)
        assert all([np.array_equal(a, b) for a, b in zip(data, loaded_data)])

        data = 42
        write_pickle(data=data, filename=filename)
        loaded_data = load_pickle(filename)
        assert loaded_data == data

        _, filename = mkstemp()
        data = np.memmap(filename, dtype="float32", mode="w+", shape=(3,))
        data[:] = [1.1, 2.2, 3.3]
        data.flush()
        data = np.memmap(filename, dtype="float32", mode="r+", shape=(3,))
        _, filename = mkstemp()
        write_pickle(data=data, filename=filename)
        loaded_data = load_pickle(filename)
        assert np.array_equal(loaded_data, data)

    @pytest.mark.parametrize(
        "cone_angle, cone_sampling, axis_angle, axis_sampling, vector, n_symmetry, convention",
        [
            (30, 5, 360, None, (1, 0, 0), 1, None),
            (45, 10, 180, 15, (0, 1, 0), 2, "zyx"),
            (60, 15, 90, 30, (0, 0, 1), 4, "xyz"),
        ],
    )
    def test_get_rotations_around_vector(
        self,
        cone_angle,
        cone_sampling,
        axis_angle,
        axis_sampling,
        vector,
        n_symmetry,
        convention,
    ):
        result = get_rotations_around_vector(
            cone_angle,
            cone_sampling,
            axis_angle,
            axis_sampling,
            vector,
            n_symmetry,
            convention,
        )

        assert isinstance(result, np.ndarray)
        if convention is None:
            assert result.shape[1:] == (3, 3)
        else:
            assert result.shape[1] == 3

    @pytest.mark.parametrize(
        "initial_vector, target_vector, convention",
        [
            ([1, 0, 0], [0, 1, 0], None),
            ([0, 1, 0], [0, 0, 1], "zyx"),
            ([1, 1, 1], [1, 0, 0], "xyz"),
        ],
    )
    def test_rotation_aligning_vectors(self, initial_vector, target_vector, convention):
        result = rotation_aligning_vectors(initial_vector, target_vector, convention)

        assert isinstance(result, np.ndarray)
        if convention is None:
            assert result.shape == (3, 3)
            assert np.allclose(np.dot(result, result.T), np.eye(3), atol=1e-6)
        else:
            assert len(result) == 3
            result = Rotation.from_euler(convention, result, degrees=True).as_matrix()
            assert np.allclose(np.dot(result, result.T), np.eye(3), atol=1e-6)

        rotated = np.dot(Rotation.from_matrix(result).as_matrix(), initial_vector)
        assert np.allclose(
            rotated / np.linalg.norm(rotated),
            target_vector / np.linalg.norm(target_vector),
            atol=1e-6,
        )

    def test_normalize_template_overflow_safe(self):
        template = be.random.random((10, 10)).astype(be.float32)
        mask = be.ones_like(template)
        n_observations = 100.0

        result = _normalize_template_overflow_safe(template, mask, n_observations)
        assert result.shape == template.shape
        assert result.dtype == template.dtype
        assert np.allclose(result.mean(), 0, atol=0.1)
        assert np.allclose(result.std(), 1, atol=0.1)

    def test_euler_conversion(self):
        rotation_matrix_initial = np.array(
            [
                [0.35355339, 0.61237244, -0.70710678],
                [-0.8660254, 0.5, -0.0],
                [0.35355339, 0.61237244, 0.70710678],
            ]
        )
        euler_angles = euler_from_rotationmatrix(rotation_matrix_initial)
        rotation_matrix_converted = euler_to_rotationmatrix(euler_angles)
        assert np.allclose(
            rotation_matrix_initial, rotation_matrix_converted, atol=1e-6
        )
