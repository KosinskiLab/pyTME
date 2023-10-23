""" General utility functions.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import os
import yaml
import pickle
from itertools import product
from typing import Tuple, Dict

import numpy as np
from numpy.typing import NDArray
from scipy.special import iv as scipy_special_iv
from scipy.ndimage import correlate1d, gaussian_filter
from scipy.optimize import minimize
from scipy.signal import convolve
from scipy.interpolate import splrep, BSpline
from scipy.stats import entropy


def is_gzipped(filename: str) -> bool:
    """Check if a file is a gzip file by reading its magic number."""
    with open(filename, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def window_to_volume(window: NDArray) -> NDArray:
    """
    Convert a 1D window to a 3D volume.

    Parameters
    ----------
    window : numpy.ndarray
        1D window.

    Returns
    -------
    numpy.ndarray
        3D volume generated from the 1D window.
    """
    window /= np.trapz(window)
    return (
        window[:, np.newaxis, np.newaxis]
        * window[np.newaxis, :, np.newaxis]
        * window[np.newaxis, np.newaxis, :]
    )


def window_kaiserb(width: int, beta: float = 3.2, order: int = 0) -> NDArray:
    """
    Create a Kaiser-Bessel window.

    Parameters
    ----------
    width : int
        Width of the window.
    beta : float, optional
        Beta parameter of the Kaiser-Bessel window. Default is 3.2.
    order : int, optional
        Order of the Bessel function. Default is 0.

    Returns
    -------
    NDArray
        Kaiser-Bessel window.

    References
    ----------
    .. [1]  Sorzano, Carlos et al (Mar. 2015). Fast and accurate conversion
            of atomic models into electron density maps. AIMS Biophysics
            2, 8–20.
    """
    window = np.arange(0, width)
    alpha = (width - 1) / 2.0
    arr = beta * np.sqrt(1 - ((window - alpha) / alpha) ** 2.0)

    return bessel(order, arr) / bessel(order, beta)


def window_blob(width: int, beta: float = 3.2, order: int = 2) -> NDArray:
    """
    Generate a blob window based on Bessel functions.

    Parameters
    ----------
    width : int
        Width of the window.
    beta : float, optional
        Beta parameter. Default is 3.2.
    order : int, optional
        Order of the Bessel function. Default is 2.

    Returns
    -------
    NDArray
        Blob window.

    References
    ----------
    .. [1]  Sorzano, Carlos et al (Mar. 2015). Fast and accurate conversion
            of atomic models into electron density maps. AIMS Biophysics
            2, 8–20.
    """
    window = np.arange(0, width)
    alpha = (width - 1) / 2.0
    arr = beta * np.sqrt(1 - ((window - alpha) / alpha) ** 2.0)

    arr = np.divide(np.power(arr, order) * bessel(order, arr), bessel(order, beta))
    arr[arr != arr] = 0
    return arr


def window_sinckb(omega: float, d: float, dw: float):
    """
    Compute the sinc window combined with a Kaiser window.

    Parameters
    ----------
    omega : float
        Reduction factor.
    d : float
        Ripple.
    dw : float
        Delta w.

    Returns
    -------
    ndarray
        Impulse response of the low-pass filter.

    References
    ----------
    .. [1]  Sorzano, Carlos et al (Mar. 2015). Fast and accurate conversion
            of atomic models into electron density maps. AIMS Biophysics
            2, 8–20.
    """
    kaiser = kaiser_mask(d, dw)
    sinc_m = sinc_mask(np.zeros(kaiser.shape), omega)

    mask = sinc_m * kaiser

    return mask / np.sum(mask)


def apply_window_filter(
    arr: NDArray,
    filter_window: NDArray,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
):
    """
    Apply a window filter on an input array.

    Parameters
    ----------
    arr : NDArray,
        Input array.
    filter_window : NDArray,
        Window filter to apply.
    mode : str, optional
        Mode for the filtering, default is "reflect".
    cval : float, optional
        Value to fill when mode is "constant", default is 0.0.
    origin : int, optional
        Origin of the filter window, default is 0.

    Returns
    -------
    NDArray,
        Array after filtering.

    """
    filter_window = filter_window[::-1]
    for axs in range(arr.ndim):
        correlate1d(
            input=arr,
            weights=filter_window,
            axis=axs,
            output=arr,
            mode=mode,
            cval=cval,
            origin=origin,
        )
    return arr


def sinc_mask(mask: NDArray, omega: float) -> NDArray:
    """
    Create a sinc mask.

    Parameters
    ----------
    mask : NDArray
        Input mask.
    omega : float
        Reduction factor.

    Returns
    -------
    NDArray
        Sinc mask.
    """
    # Move filter origin to the center of the mask
    mask_origin = int((mask.size - 1) / 2)
    dist = np.arange(-mask_origin, mask_origin + 1)

    return np.multiply(omega / np.pi, np.sinc((omega / np.pi) * dist))


def kaiser_mask(d: float, dw: float) -> NDArray:
    """
    Create a Kaiser mask.

    Parameters
    ----------
    d : float
        Ripple.
    dw : float
        Delta-w.

    Returns
    -------
    NDArray
        Kaiser mask.
    """
    # convert dw from a frequency normalized to 1 to a frequency normalized to pi
    dw *= np.pi
    A = -20 * np.log10(d)
    M = max(1, np.ceil((A - 8) / (2.285 * dw)))

    beta = 0
    if A > 50:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21:
        beta = 0.5842 * np.power(A - 21, 0.4) + 0.07886 * (A - 21)

    mask_values = np.abs(np.arange(-M, M + 1))
    mask = np.sqrt(1 - np.power(mask_values / M, 2))

    return np.divide(bessel(0, beta * mask), bessel(0, beta))


def bessel(order: int, arr: NDArray) -> NDArray:
    """
    Compute the modified Bessel function of the first kind.

    Parameters
    ----------
    order : int
        Order of the Bessel function.
    arr : NDArray
        Input array.

    Returns
    -------
    NDArray
        Bessel function values.

    """
    return scipy_special_iv(order, arr)


def electron_factor(
    dist: NDArray, method: str, atom: str, fourier: bool = False
) -> NDArray:
    """
    Compute the electron factor.

    Parameters
    ----------
    dist : NDArray
        Distance.
    method : str
        Method name.
    atom : str
        Atom type.
    fourier : bool, optional
        Whether to compute the electron factor in Fourier space.

    Returns
    -------
    NDArray
        Computed electron factor.
    """
    data = get_scattering_factors(method)
    n_range = len(data.get(atom, [])) // 2
    default = np.zeros(n_range * 3)

    res = 0.0
    a_values = data.get(atom, default)[:n_range]
    b_values = data.get(atom, default)[n_range : 2 * n_range]

    if method == "dt1969":
        b_values = data.get(atom, default)[1 : (n_range + 1)]

    for i in range(n_range):
        a = a_values[i]
        b = b_values[i]

        if fourier:
            temp = a * np.exp(-b * np.power(dist, 2))
        else:
            b = b / (4 * np.power(np.pi, 2))
            temp = a * np.sqrt(np.pi / b) * np.exp(-np.power(dist, 2) / (4 * b))

        if not np.isnan(temp).any():
            res += temp

    return res / (2 * np.pi)


def optimize_hlfp(profile, M, T, atom, method, filter_method):
    """
    Optimize high-low pass filter (HLFP).

    Parameters
    ----------
    profile : NDArray
        Input profile.
    M : int
        Scaling factor.
    T : float
        Time step.
    atom : str
        Atom type.
    method : str
        Method name.
    filter_method : str
        Filter method name.

    Returns
    -------
    float
        Fitness value.

    References
    ----------
    .. [1]  Sorzano, Carlos et al (Mar. 2015). Fast and accurate conversion
            of atomic models into electron density maps. AIMS Biophysics
            2, 8–20.
    """
    # omega, d, dw
    initial_params = [1.0, 0.01, 1.0 / 8.0]
    if filter_method == "brute":
        best_fitness = float("inf")
        OMEGA, D, DW = np.meshgrid(
            np.arange(0.7, 1.3, 0.015),
            np.arange(0.01, 0.2, 0.015),
            np.arange(0.05, 0.2, 0.015),
        )
        for omega, d, dw in zip(OMEGA.ravel(), D.ravel(), DW.ravel()):
            current_fitness = _hlpf_fitness([omega, d, dw], T, M, profile, atom, method)
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                initial_params = [omega, d, dw]
        final_params = np.array(initial_params)
    else:
        res = minimize(
            _hlpf_fitness,
            initial_params,
            args=tuple([T, M, profile, atom, method]),
            method="SLSQP",
            bounds=([0.2, 2], [1e-3, 2], [1e-3, 1]),
        )
        final_params = res.x
        if np.any(final_params != final_params):
            print(f"Solver returned NAs for atom {atom} at {M}" % (atom, M))
            final_params = final_params

    final_params[0] *= np.pi / M
    mask = window_sinckb(*final_params)

    if profile.shape[0] > mask.shape[0]:
        profile_origin = int((profile.size - 1) / 2)
        mask = window(mask, profile_origin, profile_origin)

    return mask


def _hlpf_fitness(
    params: Tuple[float], T: float, M: float, profile: NDArray, atom: str, method: str
) -> float:
    """
    Fitness function for high-low pass filter optimization.

    Parameters
    ----------
    params : tuple of float
        Parameters [omega, d, dw] for optimization.
    T : float
        Time step.
    M : int
        Scaling factor.
    profile : NDArray
        Input profile.
    atom : str
        Atom type.
    method : str
        Method name.

    Returns
    -------
    float
        Fitness value.

    References
    ----------
    .. [1]  Sorzano, Carlos et al (Mar. 2015). Fast and accurate conversion
            of atomic models into electron density maps. AIMS Biophysics
            2, 8–20.
    .. [2]  https://github.com/I2PC/xmipp/blob/707f921dfd29cacf5a161535034d28153b58215a/src/xmipp/libraries/data/pdb.cpp#L1344
    """
    omega, d, dw = params

    if not (0.7 <= omega <= 1.3) and (0 <= d <= 0.2) and (1e-3 <= dw <= 0.2):
        return 1e38 * np.random.randint(1, 100)

    mask = window_sinckb(omega=omega * np.pi / M, d=d, dw=dw)

    if profile.shape[0] > mask.shape[0]:
        profile_origin = int((profile.size - 1) / 2)
        mask = window(mask, profile_origin, profile_origin)
    else:
        filter_origin = int((mask.size - 1) / 2)
        profile = window(profile, filter_origin, filter_origin)

    f_mask = convolve(profile, mask)

    orig = int((f_mask.size - 1) / 2)
    dist = np.arange(-orig, orig + 1) * T
    t, c, k = splrep(x=dist, y=f_mask, k=3)
    i_max = np.ceil(np.divide(f_mask.shape, M))
    coarse_mask = np.arange(-i_max, i_max + 1) * M
    spline = BSpline(t, c, k)
    coarse_values = spline(coarse_mask)

    # padding to retain longer fourier response
    aux = window(
        coarse_values, x0=10 * coarse_values.shape[0], xf=10 * coarse_values.shape[0]
    )
    f_filter = np.fft.fftn(aux)
    f_filter_mag = np.abs(f_filter)
    freq = np.fft.fftfreq(f_filter.size)
    freq /= M * T
    amplitude_f = mask.sum() / coarse_values.sum()

    size_f = f_filter_mag.shape[0] * amplitude_f
    fourier_form_f = electron_factor(dist=freq, atom=atom, method=method, fourier=True)

    valid_freq_mask = freq >= 0
    f1_values = np.log10(f_filter_mag[valid_freq_mask] * size_f)
    f2_values = np.log10(np.divide(T, fourier_form_f[valid_freq_mask]))
    squared_differences = np.square(f1_values - f2_values)
    error = np.sum(squared_differences)
    error /= np.sum(valid_freq_mask)

    return error


def window(arr, x0, xf, constant_values=0):
    """
    Window an array by slicing between x0 and xf and padding if required.

    Parameters
    ----------
    arr : ndarray
        Input array to be windowed.
    x0 : int
        Start of the window.
    xf : int
        End of the window.
    constant_values : int or float, optional
        The constant values to use for padding, by default 0.

    Returns
    -------
    ndarray
        Windowed array.
    """
    origin = int((arr.size - 1) / 2)

    xs = origin - x0
    xe = origin - xf

    if xs >= 0 and xe <= arr.shape[0]:
        if xs <= arr.shape[0] and xe > 0:
            arr = arr[xs:xe]
            xs = 0
            xe = 0
        elif xs <= arr.shape[0]:
            arr = arr[xs:]
            xs = 0
    elif xe >= 0 and xe <= arr.shape[0]:
        arr = arr[:xe]
        xe = 0

    xs *= -1
    xe *= -1

    return np.pad(
        arr, (int(xs), int(xe)), mode="constant", constant_values=constant_values
    )


def atom_profile(
    M, atom, T=0.08333333, method="peng1995", lfilter=True, filter_method="minimize"
):
    """
    Generate an atom profile using a variety of methods.

    Parameters
    ----------
    M : float
        Down sampling factor.
    atom : Any
        Type or representation of the atom.
    T : float, optional
        Sampling rate in angstroms/pixel, by default 0.08333333.
    method : str, optional
        Method to be used for generating the profile, by default "peng1995".
    lfilter : bool, optional
        Whether to apply filter on the profile, by default True.
    filter_method : str, optional
        The method for the filter, by default "minimize".

    Returns
    -------
    BSpline
        A spline representation of the atom profile.

    References
    ----------
    .. [1]  Sorzano, Carlos et al (Mar. 2015). Fast and accurate conversion
            of atomic models into electron density maps. AIMS Biophysics
            2, 8–20.
    .. [2]  https://github.com/I2PC/xmipp/blob/707f921dfd29cacf5a161535034d28153b58215a/src/xmipp/libraries/data/pdb.cpp#L1344
    """
    M = M / T
    imax = np.ceil(4 / T * np.sqrt(76.7309 / (2 * np.power(np.pi, 2))))
    dist = np.arange(-imax, imax + 1) * T

    profile = electron_factor(dist, method, atom)

    if lfilter:
        window = optimize_hlfp(
            profile=profile,
            M=M,
            T=T,
            atom=atom,
            method=method,
            filter_method=filter_method,
        )
        profile = convolve(profile, window)

        indices = np.where(profile > 1e-3)
        min_indices = np.maximum(np.amin(indices, axis=1), 0)
        max_indices = np.minimum(np.amax(indices, axis=1) + 1, profile.shape)
        slices = tuple(slice(*coord) for coord in zip(min_indices, max_indices))
        profile = profile[slices]

    profile_origin = int((profile.size - 1) / 2)
    dist = np.arange(-profile_origin, profile_origin + 1) * T
    t, c, k = splrep(x=dist, y=profile, k=3)

    return BSpline(t, c, k)


def get_scattering_factors(method: str) -> Dict:
    """
    Retrieve scattering factors from a stored file based on the given method.

    Parameters
    ----------
    method : str
        Method name used to get the scattering factors.

    Returns
    -------
    Dict
        Dictionary containing scattering factors for the given method.

    Raises
    ------
    ValueError
        If the method is not found in the stored data.

    """
    path = os.path.join(os.path.dirname(__file__), "data", "scattering_factors.pickle")
    with open(path, "rb") as infile:
        data = pickle.load(infile)

    if method not in data:
        raise ValueError(f"{method} is not valid. Use {', '.join(data.keys())}.")
    return data[method]


def load_quaternions_by_angle(angle: float) -> (NDArray, NDArray, float):
    """
    Get orientations and weights proportional to the given angle.

    Parameters
    ----------
    angle : float
        Given angle.

    Returns
    -------
    tuple
        quaternions : NDArray
            Quaternion representations of orientations.
        weights : NDArray
            Weights associated with each orientation.
        angle : float
            The closest angle to the provided angle from the metadata.
    """
    # Metadata contains (N orientations, rotational sampling, coverage as values)
    with open(
        os.path.join(os.path.dirname(__file__), "data", "metadata.yaml"), "r"
    ) as infile:
        metadata = yaml.full_load(infile)

    set_diffs = {
        setname: abs(angle - set_angle)
        for setname, (_, set_angle, _) in metadata.items()
    }
    fname = min(set_diffs, key=set_diffs.get)

    infile = os.path.join(os.path.dirname(__file__), "data", fname)
    quat_weights = np.load(infile)

    quat = quat_weights[:, :4]
    weights = quat_weights[:, -1]
    angle = metadata[fname][0]

    return quat, weights, angle


def quaternion_to_rotation_matrix(quaternions: NDArray) -> NDArray:
    """
    Convert quaternions to rotation matrices.

    Parameters
    ----------
    quaternions : NDArray
        Array containing quaternions.

    Returns
    -------
    NDArray
        Rotation matrices corresponding to the given quaternions.
    """
    q0 = quaternions[:, 0]
    q1 = quaternions[:, 1]
    q2 = quaternions[:, 2]
    q3 = quaternions[:, 3]

    s = np.linalg.norm(quaternions, axis=1) * 2
    rotmat = np.zeros((quaternions.shape[0], 3, 3), dtype=np.float64)

    rotmat[:, 0, 0] = 1.0 - s * ((q2 * q2) + (q3 * q3))
    rotmat[:, 0, 1] = s * ((q1 * q2) - (q0 * q3))
    rotmat[:, 0, 2] = s * ((q1 * q3) + (q0 * q2))

    rotmat[:, 1, 0] = s * ((q2 * q1) + (q0 * q3))
    rotmat[:, 1, 1] = 1.0 - s * ((q3 * q3) + (q1 * q1))
    rotmat[:, 1, 2] = s * ((q2 * q3) - (q0 * q1))

    rotmat[:, 2, 0] = s * ((q3 * q1) - (q0 * q2))
    rotmat[:, 2, 1] = s * ((q3 * q2) + (q0 * q1))
    rotmat[:, 2, 2] = 1.0 - s * ((q1 * q1) + (q2 * q2))

    np.around(rotmat, decimals=8, out=rotmat)

    return rotmat


def reverse(arr: NDArray) -> NDArray:
    """
    Reverse the order of elements in an array along all its axes.

    Parameters
    ----------
    arr : NDArray
        Input array.

    Returns
    -------
    NDArray
        Reversed array.
    """
    return arr[(slice(None, None, -1),) * arr.ndim]


class Ntree:
    """
    N-dimensional dyadic tree.

    Each array dimension is split into two similarly sized halves. The amount of
    subvolumes per split equals 2**n with n being the dimension of the input array.

    Attributes
    ----------
    nleaves : int
        Number of leaves in the Ntree.

    """

    def __init__(self, arr: NDArray):
        """
        Initialize the Ntree with the given array.

        Parameters
        ----------
        arr : np.ndarray
            Input array to build the N-dimensional dyadic tree.

        """
        arr = np.asarray(arr)
        self._subvolumes = []
        self._sd = []
        self._arr = arr.copy()
        self._arr += np.abs(np.min(self._arr))

        np.seterr(divide="ignore", invalid="ignore")
        self._create_node(self._arr, np.zeros(arr.ndim, dtype=int), 0)
        np.seterr(divide="warn", invalid="warn")

    @property
    def nleaves(self):
        return len(self._subvolumes)

    def _create_node(self, arr: NDArray, offset: NDArray, ig: float):
        """
        Recursively split the array into nodes based on specific criteria.

        Parameters
        ----------
        arr : NDArray
            The array to split into nodes.
        offset : NDArray
            The offset for the current split in the array.
        ig : float
            Information gain value.

        """
        coordinates = self._split_arr(arr)
        sd_arr = np.std(arr)

        for chunk in coordinates:
            sd_chunk = np.std(arr[chunk])
            split_needed = False

            if np.count_nonzero(arr[chunk]) == 0 or sd_chunk == 0:
                split_needed = False
            elif not np.all(np.greater(arr[chunk].shape, 3)):
                split_needed = False
            else:
                new_split = self._split_arr(arr[chunk])
                igo = self._information_gain(arr[chunk], new_split)
                if sd_chunk < sd_arr or igo > ig:
                    split_needed = True

            if split_needed:
                new_offset = np.add(offset, [n.start for n in chunk])
                self._create_node(arr[chunk], new_offset, igo)
            else:
                final_coordinates = tuple(
                    slice(n.start + offset[i], n.stop + offset[i])
                    for i, n in enumerate(chunk)
                )
                self._subvolumes.append(final_coordinates)
                self._sd.append(np.sum(arr[tuple(chunk)] != 0))

    @staticmethod
    def _information_gain(arr: NDArray, chunks: Tuple[NDArray]):
        """
        Calculate the information gain of splitting the array.

        Parameters
        ----------
        arr : NDArray
            The array from which to calculate information gain.
        chunks : Tuple
            List of sub-arrays (chunks) created by splitting.

        Returns
        -------
        float
            The information gain of the split.

        """
        if not isinstance(chunks, list) and not isinstance(chunks, tuple):
            chunks = [chunks]

        arr_entropy = entropy(arr.ravel())
        weighted_split_entropy = [
            (arr[tuple(i)].size / arr.size) * entropy(arr[tuple(i)].ravel())
            for i in chunks
        ]
        return arr_entropy - np.sum(weighted_split_entropy)

    @staticmethod
    def _split_arr(arr: NDArray) -> Tuple[NDArray]:
        """
        Split the given array into multiple similarly sized chunks.

        Parameters
        ----------
        arr : NDArray
            The array to split.

        Returns
        -------
        tuple
            Tuple containing the slices to split the array.

        """
        old_shape = np.asarray(arr.shape).astype(int)
        new_shape = np.divide(arr.shape, 2).astype(int)
        split = tuple(
            product(
                *[
                    (slice(0, n_shape), slice(n_shape, o_shape))
                    for n_shape, o_shape in np.nditer([new_shape, old_shape])
                ]
            )
        )
        return split

    def _sd_to_range(self, scale_range: Tuple[float, float] = (0.1, 20)) -> NDArray:
        """
        Scale the standard deviation values to a specific range.

        Parameters
        ----------
        scale_range : tuple of float, optional
            The range to scale the standard deviation to.

        Returns
        -------
        NDArray
            Array of scaled standard deviation values.

        """
        scaled_sd = np.interp(
            self._sd, (np.min(self._sd), np.max(self._sd)), scale_range
        )
        return np.round(scaled_sd, decimals=0)

    def filter_chunks(
        self, arr: NDArray = None, sigma_range: Tuple[float, float] = (0.2, 10)
    ) -> NDArray:
        """
        Apply Gaussian filter to each chunk and return the filtered array.

        Parameters
        ----------
        arr : NDArray, optional
            The array to filter. If None, the original array is used.
        sigma_range : tuple of float, optional
            Range of sigma values for the Gaussian filter.

        Returns
        -------
        NDArray
            The filtered array.

        """
        if arr is None:
            arr = self._arr
        result = np.zeros_like(arr)
        chunk_sigmas = self._sd_to_range(sigma_range)

        for chunk, sigma in zip(self._subvolumes, chunk_sigmas):
            result[chunk] = gaussian_filter(arr[chunk], sigma)

        return result
