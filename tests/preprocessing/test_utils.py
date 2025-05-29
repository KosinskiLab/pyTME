import pytest
import numpy as np

from tme.filters._utils import (
    fftfreqn,
    centered_grid,
    shift_fourier,
    compute_fourier_shape,
    crop_real_fourier,
    compute_tilt_shape,
    frequency_grid_at_angle,
)


class TestPreprocessUtils:
    @pytest.mark.parametrize("reduce_dim", (False, True))
    @pytest.mark.parametrize("shape", ((10,), (10, 15), (10, 15, 30)))
    def test_compute_tilt_shape(self, shape, reduce_dim):
        tilt_shape = compute_tilt_shape(
            shape=shape, opening_axis=0, reduce_dim=reduce_dim
        )
        if reduce_dim:
            assert len(tilt_shape) == len(shape) - 1
        else:
            assert len(tilt_shape) == len(shape)
            assert tilt_shape[0] == 1

    @pytest.mark.parametrize("shape", ((10,), (10, 15), (10, 15, 30)))
    def test_centered_grid(self, shape):
        grid = centered_grid(shape=shape)
        assert grid.shape[0] == len(shape)
        center = tuple(int(x) // 2 for x in shape)
        for i in range(grid.shape[0]):
            assert grid[i][center] == 0
            assert np.max(grid[i]) <= center[i]

    @pytest.mark.parametrize("shape", ((10, 15, 30),))
    @pytest.mark.parametrize("sampling_rate", (0.5, 1, 2))
    @pytest.mark.parametrize("angle", (-5, 0, 5))
    @pytest.mark.parametrize("wedge", ((0, 1), (1, 0)))
    def test_frequency_grid_at_angle(self, shape, sampling_rate, angle, wedge):
        opening, tilt = wedge
        fgrid = frequency_grid_at_angle(
            shape=shape,
            angle=angle,
            sampling_rate=sampling_rate,
            opening_axis=opening,
            tilt_axis=tilt,
        )
        tilt_shape = compute_tilt_shape(shape, opening_axis=opening, reduce_dim=True)
        assert fgrid.shape == tuple(tilt_shape)
        assert fgrid.max() <= np.sqrt(1 / sampling_rate * len(shape))

    @pytest.mark.parametrize("n", [10, 100, 1000])
    @pytest.mark.parametrize("sampling_rate", range(1, 4))
    def test_fftfreqn(self, n, sampling_rate):
        assert np.allclose(
            fftfreqn(
                shape=(n,), sampling_rate=sampling_rate, compute_euclidean_norm=True
            ),
            np.abs(np.fft.ifftshift(np.fft.fftfreq(n=n, d=sampling_rate))),
        )

    @pytest.mark.parametrize("shape", ((10,), (10, 15), (10, 15, 30)))
    def test_crop_real_fourier(self, shape):
        data = np.random.rand(*shape)
        data_crop = crop_real_fourier(data)
        assert data_crop.shape == tuple(compute_fourier_shape(data.shape, False))

    @pytest.mark.parametrize("real", (False, True))
    @pytest.mark.parametrize("shape", ((10,), (10, 15), (10, 15, 30)))
    def test_compute_fourier_shape(self, shape, real: bool):
        data = np.random.rand(*shape)
        func = np.fft.rfftn if real else np.fft.fftn
        assert func(data).shape == tuple(compute_fourier_shape(data.shape, not real))

    def test_shift_fourier(self):
        data = np.random.rand(10)
        assert np.allclose(shift_fourier(data, False), np.fft.ifftshift(data))
