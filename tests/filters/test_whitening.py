import numpy as np

from tme.backends import backend as be
from tme.filters._utils import fftfreqn
from tme.filters.wedge import WedgeReconstructed
from tme.filters.curve import Curve
from tme.filters.whitening import (
    estimate_radial_noise_spectrum,
    _mean_log_periodogram,
    _weighted_log_radial,
)


def _colored_noise(shape, exponent, rng):
    """Noise whose Fourier amplitude scales as (1 + q) ** exponent."""
    f = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    q = fftfreqn(
        shape,
        sampling_rate=0.5,
        compute_euclidean_norm=True,
        shape_is_real_fourier=False,
        fftshift=False,
    )
    amp = (1.0 + q) ** exponent
    return np.fft.ifftn(f * amp).real.astype(np.float32)


def _wedged(vol, angles=(-50.0, 50.0)):
    """Zero the Fourier voxels outside the sampled region of a tilt geometry."""
    wedge = WedgeReconstructed(
        angles=angles, opening_axis=2, tilt_axis=0, create_continuous_wedge=True
    )
    mask = be.to_numpy_array(wedge(shape=vol.shape)["data"]) > 0
    fvol = np.fft.fftn(vol) * mask
    return np.fft.ifftn(fvol).real.astype(np.float32)


class TestPatchHelpers:
    def test_mean_log_periodogram_shape_and_dof(self):
        rng = np.random.default_rng(0)
        vol = rng.standard_normal((32, 32, 32)).astype(np.float32)
        mlp, dof = _mean_log_periodogram(
            vol, patch_size=16, overlap=0.5, reject_frac=0.0, max_patches=1024
        )
        assert mlp.shape == (16, 16, 16 // 2 + 1)
        assert np.isfinite(mlp).all()
        assert dof > 0

    def test_mean_log_periodogram_rejects_high_variance_patches(self):
        rng = np.random.default_rng(1)
        vol = rng.standard_normal((32, 32, 32)).astype(np.float32)
        _, dof_all = _mean_log_periodogram(vol, 16, 0.5, 0.0, max_patches=1024)
        _, dof_cut = _mean_log_periodogram(vol, 16, 0.5, 0.2, max_patches=1024)
        assert dof_cut < dof_all


class TestWeightedLogRadial:
    def test_recovers_flat_power(self):
        rng = np.random.default_rng(0)
        L = 24
        psd = rng.exponential(scale=1.0, size=(L, L, L // 2 + 1)).astype(np.float64)
        mean_log = np.log(psd)
        weight = np.ones_like(psd)
        _, prof = _weighted_log_radial(mean_log, weight)
        assert abs(np.nanmedian(prof[2:-2]) - 1.0) < 0.15

    def test_zero_weight_gives_nan(self):
        L = 16
        mean_log = np.zeros((L, L, L // 2 + 1))
        weight = np.zeros_like(mean_log)
        q, prof = _weighted_log_radial(mean_log, weight)
        assert np.isnan(prof).all()
        assert abs(q[-1] - 0.5) < 1e-6


class TestEstimator:
    def test_recovers_spectrum_despite_missing_wedge(self):
        rng = np.random.default_rng(0)
        vol = _colored_noise((64, 64, 64), exponent=-1.5, rng=rng)
        vol_w = _wedged(vol)
        angles = np.arange(-50, 51, 2.0)
        w = estimate_radial_noise_spectrum(
            vol_w, angles, opening_axis=2, tilt_axis=0, patch_size=32
        )
        assert np.isfinite(w).all()
        assert abs(w.max() - 1.0) < 1e-6
        assert w[0] == 0.0
        # A falling amplitude spectrum yields whitening that rises with frequency.
        assert w[-3] > w[3]

    def test_signal_blob_does_not_dominate(self):
        # Needs enough patches for variance rejection to isolate the feature,
        # so use a volume that yields hundreds of patches rather than tens.
        rng = np.random.default_rng(2)
        vol = _colored_noise((128, 128, 128), exponent=-1.0, rng=rng)
        angles = np.arange(-60, 61, 2.0)
        w0 = estimate_radial_noise_spectrum(vol, angles, patch_size=32)

        vol2 = vol.copy()
        c = 64
        vol2[c - 8 : c + 8, c - 8 : c + 8, c - 8 : c + 8] += (
            rng.standard_normal((16, 16, 16)) * 20
        ).astype(np.float32)

        w_norej = estimate_radial_noise_spectrum(
            vol2, angles, patch_size=32, reject_frac=0.0
        )
        w_rej = estimate_radial_noise_spectrum(
            vol2, angles, patch_size=32, reject_frac=0.1
        )
        d_norej = np.nanmax(np.abs(w_norej - w0))
        d_rej = np.nanmax(np.abs(w_rej - w0))
        # Rejection reduces contamination, and the contaminated estimate stays
        # close to the clean one.
        assert d_rej < d_norej
        assert d_rej < 0.05


class TestEstimatorRegularData:
    def test_angles_none_gives_uniform_whitening(self):
        rng = np.random.default_rng(3)
        vol = _colored_noise((64, 64, 64), exponent=-1.5, rng=rng)
        w = estimate_radial_noise_spectrum(vol, angles=None, patch_size=32)

        assert np.isfinite(w).all()
        assert abs(w.max() - 1.0) < 1e-6
        assert w[0] == 0.0
        assert w[-3] > w[3]

    def test_estimates_on_2d_data(self):
        rng = np.random.default_rng(4)
        img = _colored_noise((128, 128), exponent=-1.5, rng=rng)
        w = estimate_radial_noise_spectrum(img, angles=None, patch_size=32)
        assert np.isfinite(w).all()
        assert w[-3] > w[3]


def _whitening_profile():
    rng = np.random.default_rng(0)
    vol = _wedged(_colored_noise((64, 64, 64), exponent=-1.0, rng=rng))
    w = estimate_radial_noise_spectrum(vol, np.arange(-50, 51, 2.0), patch_size=32)
    return w


class TestCurve:
    def test_repeated_evaluation_is_deterministic(self):
        filt = Curve(spectrum=_whitening_profile())
        a = be.to_numpy_array(filt(shape=(40, 40, 40))["data"])
        b = be.to_numpy_array(filt(shape=(40, 40, 40))["data"])
        np.testing.assert_array_equal(a, b)

    def test_applies_to_decoupled_shape(self):
        filt = Curve(spectrum=_whitening_profile())
        data = be.to_numpy_array(filt(shape=(50, 50, 50))["data"])
        assert data.shape == (50, 50, 50)
        assert np.isfinite(data).all()
        assert data.min() >= 0.0
        assert abs(data.max() - 1.0) < 1e-3

    def test_flat_profile_maps_to_flat_filter(self):
        filt = Curve(spectrum=np.ones(20, dtype=np.float32))
        data = be.to_numpy_array(filt(shape=(16, 16, 16))["data"])
        assert data.shape == (16, 16, 16)
        assert data.min() >= 0.0
        assert abs(data.max() - 1.0) < 1e-6
