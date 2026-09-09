import pytest
import numpy as np
from typing import Tuple

from tme.backends import backend as be
from tme.filters._utils import fftfreqn
from tme.filters import BandPassReconstructed
from tme.filters.bandpass import gaussian_bandpass, discrete_bandpass


class TestBandPassFilter:
    @pytest.fixture
    def band_pass_filter(self):
        return BandPassReconstructed()

    @pytest.mark.parametrize(
        "shape, lowpass, highpass, sampling_rate",
        [
            ((10, 10), 0.2, 0.8, 1),
            ((20, 20, 20), 0.1, 0.9, 2),
            ((30, 30), None, 0.5, 1),
            ((40, 40), 0.3, None, 0.5),
        ],
    )
    def test_discrete_bandpass(
        self, shape: Tuple[int], lowpass: float, highpass: float, sampling_rate: float
    ):
        grid = fftfreqn(
            shape=shape,
            sampling_rate=0.5,
            shape_is_real_fourier=False,
            compute_euclidean_norm=True,
        )
        result = discrete_bandpass(grid, lowpass, highpass, sampling_rate)
        assert isinstance(result, type(be.ones((1,))))
        assert result.shape == shape
        assert np.all((result >= 0) & (result <= 1))

    @pytest.mark.parametrize(
        "shape, lowpass, highpass, sampling_rate",
        [
            ((10, 10), 0.2, 0.8, 1),
            ((20, 20, 20), 0.1, 0.9, 2),
            ((30, 30), None, 0.5, 1),
            ((40, 40), 0.3, None, 0.5),
        ],
    )
    def test_gaussian_bandpass(
        self, shape: Tuple[int], lowpass: float, highpass: float, sampling_rate: float
    ):
        grid = fftfreqn(
            shape=shape,
            sampling_rate=0.5,
            shape_is_real_fourier=False,
            compute_euclidean_norm=True,
        )
        result = gaussian_bandpass(grid, lowpass, highpass, sampling_rate)
        assert isinstance(result, type(be.ones((1,))))
        assert result.shape == shape
        assert np.all((result >= 0) & (result <= 1))

    @pytest.mark.parametrize("use_gaussian", [True, False])
    @pytest.mark.parametrize("return_real_fourier", [True, False])
    @pytest.mark.parametrize("shape_is_real_fourier", [True, False])
    def test_call_method(
        self,
        band_pass_filter: BandPassReconstructed,
        use_gaussian: bool,
        return_real_fourier: bool,
        shape_is_real_fourier: bool,
    ):
        band_pass_filter.use_gaussian = use_gaussian
        band_pass_filter.shape_is_real_fourier = shape_is_real_fourier

        result = band_pass_filter(shape=(10, 10), lowpass=0.2, highpass=0.8)

        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], type(be.ones((1,))))

    def test_default_values(self, band_pass_filter: BandPassReconstructed):
        assert band_pass_filter.lowpass is None
        assert band_pass_filter.highpass is None
        assert band_pass_filter.sampling_rate == 1
        assert band_pass_filter.use_gaussian is True

    @pytest.mark.parametrize("shape", ((10, 10), (20, 20, 20), (30, 30)))
    def test_return_real_fourier(self, shape: Tuple[int]):
        bpf = BandPassReconstructed()
        result = bpf(shape=shape, lowpass=0.2, highpass=0.8)
        assert result["data"].shape == shape


class TestInterpolateSpectrum:

    @pytest.mark.parametrize("shape", ((10, 10), (21, 20, 31)))
    @pytest.mark.parametrize("shape_is_real_fourier", (False, True))
    def test_interpolate_spectrum(self, shape: Tuple[int], shape_is_real_fourier: bool):
        from tme.filters.curve import _interpolate_spectrum

        n_bins = 100
        bin_centers = (np.arange(n_bins) + 0.5) / n_bins
        spectrum = np.random.random(n_bins)
        result = _interpolate_spectrum(
            bin_centers, spectrum, shape, shape_is_real_fourier
        )
        assert result.shape == tuple(shape)
        assert isinstance(result, np.ndarray)
