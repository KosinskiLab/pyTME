"""
Estimate radial noise spectra for whitening.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import numpy as np

from ._utils import radial_bins, fftfreqn
from ..backends import backend as be
from ..matching_utils import sliding_window_slices

__all__ = ["estimate_radial_noise_spectrum"]


def _hann_window(shape: tuple[int, ...]) -> np.ndarray:
    """Separable nD Hann window of the given shape."""
    window = np.ones(shape, dtype=np.float32)
    for axis, length in enumerate(shape):
        w = np.hanning(length + 2)[1:-1].astype(np.float32)
        broadcast = [1] * len(shape)
        broadcast[axis] = length
        window = window * w.reshape(broadcast)
    return window


def _mean_log_periodogram(
    data: np.ndarray,
    patch_size: int,
    overlap: float,
    reject_frac: float,
    max_patches: int,
) -> tuple[np.ndarray, int]:
    """Average the log periodogram over overlapping windowed patches."""
    length = int(patch_size)
    step = max(1, int(length * (1.0 - overlap) + 0.5))
    window = _hann_window((length,) * data.ndim)
    window_power = float((window**2).sum())

    slices = list(sliding_window_slices(data.shape, length, step))
    if max_patches and len(slices) > max_patches:
        rng = np.random.default_rng(42)
        index = rng.choice(len(slices), int(max_patches), replace=False)

        # Ascending order keeps memory mapped reads walking the file forward.
        slices = [slices[i] for i in np.sort(index)]

    logs, variances = [], []
    for sl in slices:
        # Stream memmap
        patch = np.asarray(data[sl], dtype=np.float32)
        patch = patch - patch.mean()
        rfft = np.fft.rfftn(patch * window)
        psd = (rfft.real**2 + rfft.imag**2) / window_power
        logs.append(np.log(np.fmax(psd, 1e-12)))
        variances.append(float(patch.var()))

    logs = np.asarray(logs)
    variances = np.asarray(variances)
    if logs.shape[0] == 0:
        raise RuntimeError("no usable patches for whitening filter estimation")

    keep = np.ones(logs.shape[0], bool)
    if reject_frac > 0 and logs.shape[0] >= 10:
        keep = variances <= np.quantile(variances, 1.0 - reject_frac)
    if keep.sum() == 0:
        raise RuntimeError("no usable patches after variance rejection")

    return logs[keep].mean(axis=0), int(keep.sum())


def _weighted_log_radial(
    mean_log: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted radial average of the log periodogram, debiased by the Euler-Mascheroni
    constant and exponentiated back to a linear power estimate.

    Returns ``(q, profile)`` with ``q`` the frequency of each shell in cycles per
    sample, 0.5 at Nyquist, and ``profile`` the per shell power, ``nan`` in shells
    that received no weighted samples.
    """
    from scipy import ndimage

    shape = mean_log.shape
    nb = shape[-1]
    if len(shape) > 1:
        nb = max(max(shape[:-1]) // 2 + 1, nb)

    freqs = fftfreqn(
        shape=shape,
        sampling_rate=1.0,
        compute_euclidean_norm=True,
        shape_is_real_fourier=True,
        fftshift=False,
    )
    shells_per_cycle = 2 * (nb - 1)
    bin_centers = np.arange(nb) / shells_per_cycle
    shells = np.floor(freqs * shells_per_cycle + 0.5).astype(int)

    # Naming the shells in index zeros out entries past the inscribed Nyquist sphere.
    idx = np.arange(nb)
    wsum = ndimage.sum(weight, labels=shells, index=idx)
    wlog = ndimage.sum(weight * mean_log, labels=shells, index=idx)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = wlog / wsum
    profile = np.exp(mean + np.euler_gamma)
    profile[wsum == 0] = np.nan

    return bin_centers.astype(np.float32), profile


def _smooth_profile(q: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Smooth a 1D radial profile in log space and evaluate it at all ``q``."""
    from scipy.interpolate import make_smoothing_spline

    good = np.isfinite(profile) & (profile > 0)
    if good.sum() == 0:
        return profile

    x = q[good].astype(np.float64)
    y = np.log(profile[good]).astype(np.float64)
    grid = q.astype(np.float64)
    if good.sum() < 4:
        filled = np.interp(grid, x, y)
    else:
        filled = make_smoothing_spline(x, y)(grid)
    return np.exp(filled)


def _sampling_weight(
    patch_shape: tuple[int, ...],
    angles: tuple[float, ...],
    opening_axis: int,
    tilt_axis: int,
) -> np.ndarray:
    """Continuous per voxel Fourier sampling weight from the tilt geometry.

    Reuses :py:class:`tme.filters.wedge.WedgeReconstructed` to build a graded
    tilt sampling wedge with no dose or CTF weighting. The weight is normalized
    to a maximum of one, and voxels below a small floor are set to zero. Returned
    in reduced rfft shape.
    """
    from .wedge import WedgeReconstructed

    angles = np.asarray(angles, dtype=np.float32)
    wedge = WedgeReconstructed(
        angles=angles,
        opening_axis=opening_axis,
        tilt_axis=tilt_axis,
        weight_wedge=False,
        create_continuous_wedge=len(angles) == 2,
    )
    ret = wedge(shape=patch_shape, return_real_fourier=True)
    weight = be.to_numpy_array(ret["data"]).astype(np.float32)
    weight = np.clip(weight / max(float(weight.max()), 1e-12), 0.0, 1.0)
    weight[weight < 1e-2] = 0.0
    return weight


def estimate_radial_noise_spectrum(
    data: np.ndarray,
    angles: tuple[float, ...] = None,
    opening_axis: int = 2,
    tilt_axis: int = 0,
    patch_size: int = 64,
    overlap: float = 0.5,
    reject_frac: float = 0.10,
    max_patches: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a radial whitening profile.

    The noise power spectrum is estimated over overlapping windowed patches. When
    ``angles`` are given (a tomogram), the radial average is restricted to the
    Fourier region the tilt geometry sampled, so the estimate is not biased by
    the missing wedge. Without ``angles`` all sampled frequencies contribute
    equally, giving a plain whitening spectrum for regular data.

    Parameters
    ----------
    data : np.ndarray
        Real space array to estimate from. May be memory mapped; patches are
        streamed so the full array is never realized.
    angles : tuple of float, optional
        Tilt angles in degrees. When None no wedge weighting is applied.
    opening_axis, tilt_axis : int
        Wedge geometry, following the :py:class:`tme.filters.wedge.Wedge`
        convention. Only used when ``angles`` are given.
    patch_size : int
        Edge length of the cubic estimation box, clamped to the array.
    overlap : float
        Fractional overlap between patches.
    reject_frac : float
        Fraction of the highest variance patches to reject.
    max_patches : int
        Upper bound on the number of patches. Patches are drawn at random from
        the full set when it is exceeded.

    Returns
    -------
    np.ndarray
        The whitening profile ``1 / sqrt(power)`` with the DC term set to zero and
        normalized to a maximum of one, over equal shells from DC to Nyquist. This
        is the abscissa :py:class:`tme.filters.Curve` rebuilds from the profile
        length, so it can be handed to it directly.
    """
    patch_size = min(int(patch_size), *data.shape)
    patch_size -= patch_size % 2
    if patch_size < 8:
        raise ValueError(f"patch_size {patch_size} too small to estimate a spectrum")

    mean_log, _ = _mean_log_periodogram(
        data, patch_size, overlap, reject_frac, max_patches
    )

    if angles is None:
        weight = np.ones_like(mean_log)
    else:
        patch_shape = (int(patch_size),) * data.ndim
        weight = _sampling_weight(patch_shape, angles, opening_axis, tilt_axis)

    q, profile = _weighted_log_radial(mean_log, weight)

    model = _smooth_profile(q, profile)
    w = np.where(model > 0, 1.0 / np.sqrt(model), 0.0)
    w[0] = 0.0
    w = w / max(float(w.max()), 1e-30)
    return w.astype(np.float32)
