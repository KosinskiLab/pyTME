import pytest
import numpy as np
from typing import Tuple

from tme.backends import backend as be
from tme.filters._utils import compute_fourier_shape
from tme.filters import BandPassFilter, LinearWhiteningFilter


class TestBandPassFilter:
    @pytest.fixture
    def band_pass_filter(self):
        return BandPassFilter()

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
        result = BandPassFilter.discrete_bandpass(
            shape, lowpass, highpass, sampling_rate
        )
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
        result = BandPassFilter.gaussian_bandpass(
            shape, lowpass, highpass, sampling_rate
        )
        assert isinstance(result, type(be.ones((1,))))
        assert result.shape == shape
        assert np.all((result >= 0) & (result <= 1))

    @pytest.mark.parametrize("use_gaussian", [True, False])
    @pytest.mark.parametrize("return_real_fourier", [True, False])
    @pytest.mark.parametrize("shape_is_real_fourier", [True, False])
    def test_call_method(
        self,
        band_pass_filter: BandPassFilter,
        use_gaussian: bool,
        return_real_fourier: bool,
        shape_is_real_fourier: bool,
    ):
        band_pass_filter.use_gaussian = use_gaussian
        band_pass_filter.return_real_fourier = return_real_fourier
        band_pass_filter.shape_is_real_fourier = shape_is_real_fourier

        result = band_pass_filter(shape=(10, 10), lowpass=0.2, highpass=0.8)

        assert isinstance(result, dict)
        assert "data" in result
        assert "is_multiplicative_filter" in result
        assert isinstance(result["data"], type(be.ones((1,))))
        assert result["is_multiplicative_filter"] is True

    def test_default_values(self, band_pass_filter: BandPassFilter):
        assert band_pass_filter.lowpass is None
        assert band_pass_filter.highpass is None
        assert band_pass_filter.sampling_rate == 1
        assert band_pass_filter.use_gaussian is True
        assert band_pass_filter.return_real_fourier is False
        assert band_pass_filter.shape_is_real_fourier is False

    @pytest.mark.parametrize("shape", ((10, 10), (20, 20, 20), (30, 30)))
    def test_return_real_fourier(self, shape: Tuple[int]):
        bpf = BandPassFilter(return_real_fourier=True)
        result = bpf(shape=shape, lowpass=0.2, highpass=0.8)
        expected_shape = tuple(compute_fourier_shape(shape, False))
        assert result["data"].shape == expected_shape


class TestLinearWhiteningFilter:
    @pytest.mark.parametrize(
        "shape, n_bins, batch_dimension",
        [
            ((10, 10), None, None),
            ((20, 20, 20), 15, 0),
            ((30, 30, 30), 20, 1),
            ((40, 40, 40, 40), 25, 2),
        ],
    )
    def test_compute_spectrum(
        self, shape: Tuple[int], n_bins: int, batch_dimension: int
    ):
        data_rfft = be.fft.rfftn(be.random.random(shape))
        bins, radial_averages = LinearWhiteningFilter._compute_spectrum(
            data_rfft, n_bins, batch_dimension
        )
        data_shape = tuple(
            int(x) for i, x in enumerate(data_rfft.shape) if i != batch_dimension
        )

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
            ((10, 10), None, None, 1),
            ((20, 20, 20), 15, 0, 2),
            ((30, 30, 30), 20, 1, None),
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
            shape=shape,
            data=data,
            n_bins=n_bins,
            batch_dimension=batch_dimension,
            order=order,
        )

        assert isinstance(result, dict)
        assert result.get("data", False) is not False
        assert result.get("is_multiplicative_filter", False)
        assert isinstance(result["data"], type(be.ones((1,))))
        data_shape = tuple(
            int(x) for i, x in enumerate(data.shape) if i != batch_dimension
        )
        assert result["data"].shape == tuple(compute_fourier_shape(data_shape, False))

    def test_call_method_with_data_rfft(self):
        shape = (30, 30, 30)
        data_rfft = be.fft.rfftn(be.random.random(shape))
        result = LinearWhiteningFilter()(
            shape=shape, data_rfft=data_rfft, return_real_fourier=True
        )

        assert isinstance(result, dict)
        assert result.get("data", False) is not False
        assert result.get("is_multiplicative_filter", False)
        assert isinstance(result["data"], type(be.ones((1,))))
        assert result["data"].shape == data_rfft.shape

    @pytest.mark.parametrize("shape", [(10, 10), (20, 20, 20), (30, 30, 30)])
    def test_filter_mask_range(self, shape: Tuple[int]):
        data = be.random.random(shape)
        result = LinearWhiteningFilter()(shape=shape, data=data)

        filter_mask = result["data"]
        assert np.all(filter_mask >= 0) and np.all(filter_mask <= 1)
