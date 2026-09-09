import pytest
import numpy as np

from tme.filters._utils import (
    fftfreqn,
    shift_fourier,
    compute_fourier_shape,
    crop_real_fourier,
    compute_tilt_shape,
    frequency_grid_at_angle,
    create_reconstruction_filter,
    pad_to_length,
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

    def test_frequency_grid_at_angle_edge_cases(self):
        """Test edge cases: missing axes and uniform shapes."""
        shape = (10, 15, 20)
        # Missing axes should fall back to fftfreqn
        grid = frequency_grid_at_angle(shape, 5, 1.0, opening_axis=None, tilt_axis=0)
        assert grid.shape == shape

        # Uniform shape should also use fftfreqn fallback
        grid = frequency_grid_at_angle(
            (10, 10, 10), 5, 1.0, opening_axis=2, tilt_axis=0
        )
        assert grid.shape == (10, 10)

    @pytest.mark.parametrize("shape", ((15, 15, 15), (31, 31, 31), (64, 64, 64)))
    @pytest.mark.parametrize("sampling_rate", (0.5, 1, 2))
    @pytest.mark.parametrize("angle", (-5, 0, 5))
    def test_freqgrid_comparison(self, shape, sampling_rate, angle):
        grid = frequency_grid_at_angle(
            shape=shape,
            angle=angle,
            sampling_rate=sampling_rate,
            opening_axis=2,
            tilt_axis=0,
        )
        grid2 = fftfreqn(
            shape=shape[1:], sampling_rate=sampling_rate, compute_euclidean_norm=True
        )
        # These should be equal for cubical input shapes
        assert np.allclose(grid, grid2)

    @pytest.mark.parametrize("n", [10, 100, 1000])
    @pytest.mark.parametrize("sampling_rate", range(1, 4))
    def test_fftfreqn(self, n, sampling_rate):
        assert np.allclose(
            fftfreqn(
                shape=(n,),
                sampling_rate=sampling_rate,
                compute_euclidean_norm=True,
                fftshift=True,
            ),
            np.abs(np.fft.ifftshift(np.fft.fftfreq(n=n, d=sampling_rate))),
        )
        assert np.allclose(
            fftfreqn(
                shape=(n,),
                sampling_rate=sampling_rate,
                compute_euclidean_norm=True,
                fftshift=False,
            ),
            np.abs(np.fft.fftfreq(n=n, d=sampling_rate)),
        )

    def test_fftfreqn_real_fourier(self):
        shape = (10, 15, 8)
        grid = fftfreqn(shape, 2.0, shape_is_real_fourier=True, fftshift=False)
        assert grid.shape == (len(shape), *shape)

    def test_fftfreqn_sparse(self):
        grids = fftfreqn((10, 15), 1.0, return_sparse_grid=True)
        assert isinstance(grids, list) and len(grids) == 2

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

    @pytest.mark.parametrize("real_fourier", (False, True))
    def test_shift_fourier(self, real_fourier):
        data = np.random.rand(10, 15)
        shifted = shift_fourier(data, shape_is_real_fourier=real_fourier)
        assert shifted.shape == data.shape

    @pytest.mark.parametrize(
        "filter_type", ["ram-lak", "ramp-cont", "shepp-logan", "cosine", "hamming"]
    )
    def test_create_reconstruction_filter(self, filter_type):
        filt = create_reconstruction_filter((32, 32), filter_type, fftshift=True)
        assert filt.shape == (32, 32) and np.all(np.isfinite(filt))

    def test_create_reconstruction_filter_ramp_with_angles(self):
        angles = np.array([-60, -30, 0, 30, 60])
        filt = create_reconstruction_filter(
            (32,), "ramp", tilt_angles=angles, fftshift=True
        )
        assert filt.shape == (32,) and np.max(filt) <= 1.0

    def test_create_reconstruction_filter_errors(self):
        with pytest.raises(ValueError, match="tilt angles"):
            create_reconstruction_filter((32,), "ramp")

        with pytest.raises(ValueError, match="Unsupported"):
            create_reconstruction_filter((32,), "invalid")

    @pytest.mark.parametrize("arr,length", [(1, 10), (2, 4)])
    def test_pad_to_length(self, arr, length):
        result = pad_to_length(arr, length)
        expected_len = length // np.atleast_1d(arr).size
        assert len(result) == expected_len
