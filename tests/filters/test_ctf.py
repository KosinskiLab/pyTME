import pytest
import numpy as np
from tme.filters import CTF, CTFReconstructed


class TestCTF:
    """Test CTF filter core functionality."""

    @pytest.fixture
    def basic_ctf(self):
        return CTF(defocus=(30000,), angles=(0,), sampling_rate=4.0)

    def test_ctf_initialization(self, basic_ctf):
        assert basic_ctf.defocus == (30000,)
        assert basic_ctf.angles == (0,)
        assert basic_ctf.sampling_rate == 4.0

    def test_ctf_evaluation(self, basic_ctf):
        shape = (64, 64)
        result = basic_ctf(shape=shape)

        assert "data" in result
        assert result["data"].shape[0] == len(basic_ctf.angles)
        assert result["data"].shape[1:] == shape

    def test_ctf_multiple_angles(self):
        ctf = CTF(defocus=(30000, 32000), angles=(-30, 30), sampling_rate=4.0)
        result = ctf(shape=(64, 64))

        assert result["data"].shape[0] == 2

    def test_ctf_astigmatism(self):
        # defocus_delta = (defocus_x - defocus_y) / 2 = (30000 - 28000) / 2 = 1000
        ctf = CTF(
            defocus=(29000,),
            defocus_delta=(1000,),
            astigmatism_angle=np.radians(45),
            angles=(0,),
            sampling_rate=4.0,
        )
        result = ctf(shape=(64, 64))

        assert result["data"].shape == (1, 64, 64)


class TestCTFReconstructed:
    """Test CTFReconstructed filter for 3D reconstructions."""

    def test_ctf_reconstructed_basic(self):
        ctf = CTFReconstructed(defocus=30000, sampling_rate=4.0)
        result = ctf(shape=(64, 64, 64))

        assert "data" in result
        assert result["data"].shape == (64, 64, 64)

    def test_ctf_reconstructed_flip_phase(self):
        ctf = CTFReconstructed(
            defocus=30000, sampling_rate=4.0, correction_mode="phase-flip"
        )
        result = ctf(shape=(64, 64, 64))
        assert np.all(result["data"] >= 0)
