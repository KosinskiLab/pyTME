import pytest
import numpy as np

from tme.filters import Wedge, WedgeReconstructed


class TestWedge:
    """Test Wedge filter core functionality."""

    @pytest.fixture
    def basic_wedge(self):
        return Wedge(angles=np.arange(-60, 61, 3), tilt_axis=0, opening_axis=2)

    def test_wedge_initialization(self, basic_wedge):
        assert len(basic_wedge.angles) == 41
        assert basic_wedge.tilt_axis == 0
        assert basic_wedge.opening_axis == 2

    def test_wedge_evaluation(self, basic_wedge):
        shape = (32, 32, 32)
        result = basic_wedge(shape=shape)

        assert "data" in result
        assert result["data"].shape[0] == len(basic_wedge.angles)

    def test_wedge_weighting_angle(self):
        wedge = Wedge(
            angles=np.array([-60, 0, 60]),
            weight_type="angle",
            tilt_axis=0,
            opening_axis=2,
        )
        result = wedge(shape=(32, 32, 32))
        assert result["data"].shape[0] == 3


class TestWedgeReconstructed:
    """Test WedgeReconstructed for 3D reconstructions."""

    def test_continuous_wedge(self):
        wedge = WedgeReconstructed(
            angles=(-60, 60), create_continuous_wedge=True, tilt_axis=0, opening_axis=2
        )
        result = wedge(shape=(64, 64, 64))

        assert result["data"].shape == (64, 64, 64)
        assert np.all((result["data"] == 0) | (result["data"] == 1))

    def test_step_wedge(self):
        angles = np.arange(-60, 61, 10)
        wedge = WedgeReconstructed(
            angles=angles, create_continuous_wedge=False, tilt_axis=0, opening_axis=2
        )
        result = wedge(shape=(64, 64, 64))

        assert result["data"].shape == (64, 64, 64)

    def test_wedge_with_weights(self):
        angles = np.array([-60, -30, 0, 30, 60])
        weights = np.array([0.5, 0.8, 1.0, 0.8, 0.5])

        wedge = WedgeReconstructed(
            angles=angles,
            weights=weights,
            weight_wedge=True,
            tilt_axis=0,
            opening_axis=2,
        )
        result = wedge(shape=(64, 64, 64))
        assert not np.all((result["data"] == 0) | (result["data"] == 1))
