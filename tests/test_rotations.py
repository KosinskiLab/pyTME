import pytest
import numpy as np

from tme.rotations import (
    get_cone_rotations,
    align_vectors,
    euler_to_rotationmatrix,
    euler_from_rotationmatrix,
    get_rotation_matrices,
    align_to_axis,
    get_symmetry_matrices,
    reduce_rotations_by_symmetry,
)


class TestConeRotations:
    def test_basic_cone(self):
        rots = get_cone_rotations(cone_angle=30.0, cone_sampling=15.0)
        assert rots.shape[1:] == (3, 3)
        assert np.allclose(np.linalg.det(rots), 1.0)

    def test_symmetry_reduces_rotations(self):
        rots_no_sym = get_cone_rotations(30.0, 15.0, n_symmetry=1)
        rots_sym = get_cone_rotations(30.0, 15.0, n_symmetry=2)
        assert len(rots_no_sym) > len(rots_sym)

    def test_custom_reference_axis(self):
        rots = get_cone_rotations(20.0, 10.0, reference=(1, 0, 0))
        assert rots.shape[1:] == (3, 3)


class TestAlignVectors:
    def test_align_to_z_axis(self):
        rot = align_vectors((1, 0, 0), (0, 0, 1))
        result = rot @ np.array([1, 0, 0])
        assert np.allclose(result, [0, 0, 1], atol=1e-6)

    def test_identity_for_same_vectors(self):
        rot = align_vectors((0, 0, 1), (0, 0, 1))
        assert np.allclose(rot, np.eye(3), atol=1e-6)


class TestEulerConversions:
    def test_roundtrip_conversion(self):
        angles = (45.0, 30.0, 60.0)
        rot = euler_to_rotationmatrix(angles, seq="ZYZ")
        recovered = euler_from_rotationmatrix(rot, seq="ZYZ")
        assert np.allclose(angles, recovered, atol=1e-5)

    def test_identity_rotation(self):
        rot = euler_to_rotationmatrix((0, 0, 0))
        assert np.allclose(rot, np.eye(3))

    def test_xyz_convention(self):
        angles = (90, 0, 0)
        rot = euler_to_rotationmatrix(angles, seq="XYZ")
        assert rot.shape == (3, 3)


class TestGetRotationMatrices:
    def test_returns_valid_rotations(self):
        rots = get_rotation_matrices(angular_sampling=30.0)
        assert np.allclose(np.linalg.det(rots), 1.0)
        assert rots[0].shape == (3, 3)

    def test_finer_sampling(self):
        coarse = get_rotation_matrices(45.0)
        fine = get_rotation_matrices(15.0)
        assert len(fine) > len(coarse)

    def test_identity_included(self):
        rots = get_rotation_matrices(30.0)
        assert np.allclose(rots[0], np.eye(3))

    def test_dimensions(self, dim=2):
        rots = get_rotation_matrices(30.0, dim=dim, use_optimized_set=False)
        assert rots.shape[1:] == (dim, dim)


class TestAlignToAxis:
    def test_align_line_to_z(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        rot = align_to_axis(coords, axis=2)
        transformed = coords @ rot.T
        principal = (
            np.pca.fit(transformed).components_[0]
            if hasattr(np, "pca")
            else transformed[-1] - transformed[0]
        )
        assert abs(principal[2]) > abs(principal[0])

    def test_with_weights(self):
        coords = np.random.randn(10, 3)
        weights = np.ones(10)
        rot = align_to_axis(coords, weights=weights)
        assert rot.shape == (3, 3)
        assert np.allclose(np.linalg.det(rot), 1.0)

    def test_flip_direction(self):
        coords = np.array([[0, 0, 0], [1, 0, 0]])
        rot_normal = align_to_axis(coords, axis=2, flip=False)
        rot_flipped = align_to_axis(coords, axis=2, flip=True)
        assert not np.allclose(rot_normal, rot_flipped)

    def test_secondary_eigenvector(self):
        coords = np.random.randn(20, 3)
        rot = align_to_axis(coords, eigenvector_index=1)
        assert rot.shape == (3, 3)

    def test_invalid_eigenvector_index(self):
        coords = np.random.randn(10, 3)
        with pytest.raises(ValueError):
            align_to_axis(coords, eigenvector_index=3)


class TestSymmetryMatrices:
    def test_c2_symmetry(self):
        mats = get_symmetry_matrices("C2")
        assert len(mats) == 2
        assert np.allclose(mats[0], np.eye(3))

    def test_c4_symmetry(self):
        mats = get_symmetry_matrices("C4")
        assert len(mats) == 4
        # Each should be 90 degree rotation
        for mat in mats[1:]:
            assert np.allclose(np.linalg.det(mat), 1.0)

    def test_d2_symmetry(self):
        mats = get_symmetry_matrices("D2")
        assert len(mats) == 4

    def test_custom_axis(self):
        mats = get_symmetry_matrices("C3", axis=(1, 1, 0))
        assert len(mats) == 3
        assert all(np.allclose(np.linalg.det(m), 1.0) for m in mats)

    def test_unsupported_symmetry(self):
        with pytest.raises(ValueError):
            get_symmetry_matrices("X5")

    def test_symmetry_groups_are_closed(self):
        for sym in ("C2", "C4", "D2", "D3", "D4"):
            mats = get_symmetry_matrices(sym)
            for a in mats:
                for b in mats:
                    prod = a @ b
                    nearest = min(float(np.linalg.norm(prod - m)) for m in mats)
                    assert nearest < 1e-4, f"{sym} is not closed under composition"


class TestReduceRotationsBySymmetry:
    @staticmethod
    def _worst_coverage(full, reduced, sym_ops):
        def ang(a, b):
            return np.degrees(np.arccos(np.clip((np.trace(a @ b.T) - 1) / 2, -1, 1)))

        worst = 0.0
        for r in full:
            nearest = min(min(ang(r, k @ s) for s in sym_ops) for k in reduced)
            worst = max(worst, nearest)
        return worst

    def test_c1_is_noop(self):
        full = get_rotation_matrices(30.0)
        reduced = reduce_rotations_by_symmetry(full, "C1")
        assert reduced.shape[0] == full.shape[0]

    def test_reduces_by_group_order_cyclic(self):
        full = get_rotation_matrices(15.0)
        reduced = reduce_rotations_by_symmetry(full, "C4")
        ratio = full.shape[0] / reduced.shape[0]
        assert 3.0 < ratio < 5.0

    def test_reduces_by_group_order_dihedral(self):
        full = get_rotation_matrices(15.0)
        reduced = reduce_rotations_by_symmetry(full, "D2")
        ratio = full.shape[0] / reduced.shape[0]
        assert 3.0 < ratio < 5.5

    def test_sampling_independent(self):
        # Coarse sampling must still reduce by ~group order. A closure-dependent
        # implementation would collapse nothing here and fail this test.
        full = get_rotation_matrices(30.0)
        reduced = reduce_rotations_by_symmetry(full, "C4")
        assert 3.0 < full.shape[0] / reduced.shape[0] < 5.0

    def test_coverage_within_two_samplings(self):
        samp = 30.0
        full = get_rotation_matrices(samp)
        sym = get_symmetry_matrices("D2")
        reduced = reduce_rotations_by_symmetry(full, "D2")
        assert self._worst_coverage(full, reduced, sym) <= 2.0 * samp

    def _cone_coverage(self, reduced, reference, cone_angle, n_dirs=4000):
        # Reproduces the constrained analyzer's cone test: a seed direction is
        # covered when some surviving rotation's facing vector R.T @ reference
        # lies within cone_angle of it. The seeds tile the sphere like a mesh.
        facing = reduced.transpose(0, 2, 1) @ np.asarray(reference, dtype=float)

        i = np.arange(n_dirs)
        z = 1 - 2 * (i + 0.5) / n_dirs
        r = np.sqrt(1 - z * z)
        theta = np.pi * (1 + 5**0.5) * i
        seeds = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)

        cutoff = np.cos(np.radians(cone_angle))
        return (seeds @ facing.T >= cutoff).sum(axis=1)

    def test_cyclic_leaves_no_empty_cone(self):
        # Regression: reducing on the wrong side emptied a localized patch of seed
        # directions, so constrained matching rejected every rotation there.
        reference, cone_angle = (0, 0, 1), 20.0
        full = get_rotation_matrices(15.0)
        for sym in ("C2", "C3", "C4"):
            reduced = reduce_rotations_by_symmetry(full, sym)
            coverage = self._cone_coverage(reduced, reference, cone_angle)
            assert coverage.min() > 0, f"{sym} leaves seed directions with no rotation"


class TestRotationProperties:
    def test_orthogonality(self):
        rots = get_rotation_matrices(30.0)
        for rot in rots[:5]:
            assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-6)

    def test_determinant_one(self):
        rots = get_cone_rotations(45.0, 20.0)
        assert np.allclose(np.linalg.det(rots), 1.0, atol=1e-6)

    def test_vector_length_preservation(self):
        rot = align_vectors((1, 2, 3), (0, 0, 1))
        vec = np.array([1.0, 2.0, 3.0])
        rotated = rot @ vec
        assert np.allclose(np.linalg.norm(vec), np.linalg.norm(rotated))
