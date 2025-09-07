import pytest
import numpy as np
from typing import Tuple

from tme.backends import backend as be
from tme.filters._utils import compute_fourier_shape
from tme.filters import BandPassReconstructed, LinearWhiteningFilter
from tme.filters.bandpass import gaussian_bandpass, discrete_bandpass
from tme.filters._utils import fftfreqn


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


class TestLinearWhiteningFilter:
    @pytest.mark.parametrize(
        "shape, n_bins",
        [
            ((10, 10), None),
            ((20, 20, 20), 15),
            ((30, 30, 30), 20),
            ((40, 40, 40, 40), 25),
        ],
    )
    def test_compute_spectrum(self, shape: Tuple[int], n_bins: int):
        data_rfft = be.fft.rfftn(be.random.random(shape))
        bins, radial_averages = LinearWhiteningFilter._compute_spectrum(
            data_rfft, n_bins
        )
        data_shape = tuple(int(x) for i, x in enumerate(data_rfft.shape))

        assert isinstance(bins, np.ndarray)
        assert isinstance(radial_averages, np.ndarray)
        assert bins.shape == data_shape
        assert radial_averages.ndim == 1
        assert np.all(radial_averages >= 0) and np.all(radial_averages <= 1)

    @pytest.mark.parametrize("shape", ((10, 10), (21, 20, 31)))
    @pytest.mark.parametrize("shape_is_real_fourier", (False, True))
    @pytest.mark.parametrize("order", (1, 3))
    def test_interpolate_spectrum(
        self, shape: Tuple[int], shape_is_real_fourier: bool, order: int
    ):
        spectrum = be.random.random(100)
        result = LinearWhiteningFilter()._interpolate_spectrum(
            spectrum, shape, shape_is_real_fourier, order
        )
        assert result.shape == tuple(shape)
        assert isinstance(result, np.ndarray)

    @pytest.mark.parametrize(
        "shape, n_bins, batch_dimension, order",
        [
            ((10, 10), None, (), 1),
            ((20, 20, 20), 15, 0, 1),
        ],
    )
    def test_call_method(
        self,
        shape: Tuple[int],
        n_bins: int,
        batch_dimension: int,
        order: int,
    ):
        data = be.random.random(shape)
        result = LinearWhiteningFilter()(
            shape=tuple(x for i, x in enumerate(shape) if i != batch_dimension),
            data_rfft=np.fft.rfftn(data),
            n_bins=n_bins,
            axes=batch_dimension,
            order=order,
        )

        assert isinstance(result, dict)
        assert result.get("data", False) is not False
        assert isinstance(result["data"], type(be.ones((1,))))
        assert result["data"].shape == shape

    def test_call_method_with_data_rfft(self):
        shape = (30, 30, 30)
        data_rfft = be.fft.rfftn(be.random.random(shape))
        result = LinearWhiteningFilter()(
            shape=shape, data_rfft=data_rfft, return_real_fourier=True
        )

        assert isinstance(result, dict)
        assert result.get("data", False) is not False
        assert isinstance(result["data"], type(be.ones((1,))))
        assert result["data"].shape == data_rfft.shape

    @pytest.mark.parametrize("shape", [(10, 10), (20, 20, 20), (30, 30, 30)])
    def test_filter_mask_range(self, shape: Tuple[int]):
        data = be.random.random(shape)
        result = LinearWhiteningFilter()(shape=shape, data_rfft=np.fft.rfftn(data))

        filter_mask = result["data"]
        assert np.all(filter_mask >= 0) and np.all(filter_mask <= 1)
