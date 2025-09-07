from importlib_resources import files
from itertools import combinations, chain, product

import pytest
import numpy as np

from tme import Density
from scipy.spatial.transform import Rotation
from scipy.signal import correlate

from tme.mask import elliptical_mask
from tme.rotations import (
    euler_from_rotationmatrix,
    euler_to_rotationmatrix,
    get_cone_rotations,
    align_vectors,
    get_rotation_matrices,
)
from tme.matching_utils import (
    split_shape,
    compute_full_convolution_index,
)

BASEPATH = files("tests.data")


class TestRotations:
    def setup_method(self):
        self.density = Density.from_file(str(BASEPATH.joinpath("Raw/em_map.map")))
        self.structure_density = Density.from_structure(
            filename_or_structure=str(BASEPATH.joinpath("Structures/5khe.cif")),
            origin=self.density.origin,
            shape=self.density.shape,
            sampling_rate=self.density.sampling_rate,
        )

    @pytest.mark.parametrize(
        "initial_vector, target_vector",
        [
            ([1, 0, 0], [0, 1, 0]),
            ([0, 1, 0], [0, 0, 1]),
            ([1, 1, 1], [1, 0, 0]),
        ],
    )
    def test_align_vectors(self, initial_vector, target_vector):
        result = align_vectors(initial_vector, target_vector)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        assert np.allclose(np.dot(result, result.T), np.eye(3), atol=1e-6)

        rotated = np.dot(Rotation.from_matrix(result).as_matrix(), initial_vector)
        assert np.allclose(
            rotated / np.linalg.norm(rotated),
            target_vector / np.linalg.norm(target_vector),
            atol=1e-6,
        )

    @pytest.mark.parametrize(
        "cone_angle, cone_sampling, axis_angle, axis_sampling, vector, n_symmetry",
        [
            (30, 5, 360, None, (1, 0, 0), 1),
            (45, 10, 180, 15, (0, 1, 0), 2),
            (60, 15, 90, 30, (0, 0, 1), 4),
        ],
    )
    def test_get_cone_rotations(
        self,
        cone_angle,
        cone_sampling,
        axis_angle,
        axis_sampling,
        vector,
        n_symmetry,
    ):
        result = get_cone_rotations(
            cone_angle=cone_angle,
            cone_sampling=cone_sampling,
            axis_angle=axis_angle,
            axis_sampling=axis_sampling,
            reference=vector,
            n_symmetry=n_symmetry,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape[1:] == (3, 3)

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
