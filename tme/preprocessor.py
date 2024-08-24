""" Implements Preprocessor class for filtering operations.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import os
import pickle
import inspect
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage
from scipy.special import iv as bessel
from scipy.interpolate import interp1d, splrep, BSpline
from scipy.optimize import differential_evolution, minimize

from .types import NDArray


class Preprocessor:
    """
    Implements filtering operations on density arrays.
    """

    def apply_method(self, method: str, parameters: Dict):
        """
        Invoke ``Preprocessor.method`` using ``parameters``.

        Parameters
        ----------
        method : str
            The name of the method to be used.
        parameters : dict
            The parameters for the specified method.

        Returns
        -------
        The output of ``method``.

        Raises
        ------
        NotImplementedError
            If ``method`` is not a member of :py:class:`Preprocessor`.
        """
        if not hasattr(self, method):
            raise NotImplementedError(
                f"'{method}' is not supported as a filter method on this class."
            )
        return getattr(self, method)(**parameters)

    def method_to_id(self, method: str, parameters: Dict) -> str:
        """
        Generate a unique ID for a specific method operation.

        Parameters
        ----------
        method : str
            The name of the method.
        parameters : dict
            A dictionary containing the parameters used by the method.

        Returns
        -------
        str
            A string representation of the method operation, which can be used
            as a unique identifier.

        Raises
        ------
        NotImplementedError
            If ``method`` is not a member of :py:class:`Preprocessor`.
        """
        if not hasattr(self, method):
            raise NotImplementedError(
                f"'{method}' is not supported as a filter method on this class."
            )
        signature = inspect.signature(getattr(self, method))
        default = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        default.update(parameters)

        return "-".join([str(default[key]) for key in sorted(default.keys())])

    def gaussian_filter(
        self,
        template: NDArray,
        sigma: Tuple[float],
        cutoff_value: float = 4.0,
    ) -> NDArray:
        """
        Convolve an atomic structure with a Gaussian kernel.

        Parameters
        ----------
        template : NDArray
            Input data.
        sigma : float or tuple of floats
            The standard deviation of the Gaussian kernel along one or all axes.
        cutoff_value : float, optional
            Truncates the Gaussian kernel at cutoff_values times sigma.

        Returns
        -------
        NDArray
            Gaussian filtered template.
        """
        sigma = 0 if sigma is None else sigma
        return ndimage.gaussian_filter(template, sigma, cval=cutoff_value)

    def difference_of_gaussian_filter(
        self, template: NDArray, low_sigma: NDArray, high_sigma: NDArray
    ) -> NDArray:
        """
        Apply the Difference of Gaussian (DoG) bandpass filter on
        the provided template.

        Parameters
        ----------
        template : NDArray
            The input template on which to apply the technique.
        low_sigma : NDArray
            The smaller standard deviation for the Gaussian kernel.
            Should be scalar or sequence of scalars of length template.ndim.
        high_sigma : NDArray
            The larger standard deviation for the Gaussian kernel.
            Should be scalar or sequence of scalars of length template.ndim.

        Returns
        -------
        NDArray
            The result of applying the Difference of Gaussian technique on the template.
        """
        if np.any(low_sigma > high_sigma):
            print("low_sigma should be smaller than high_sigma.")
        im1 = self.gaussian_filter(template, low_sigma)
        im2 = self.gaussian_filter(template, high_sigma)
        return im1 - im2

    def local_gaussian_alignment_filter(
        self,
        target: NDArray,
        template: NDArray,
        lbd: float,
        sigma_range: Tuple[float, float] = (0.1, 20),
    ) -> NDArray:
        """
        Simulate electron density by optimizing a sum of Gaussians.

        For that, the following minimization problem is considered:

        .. math::
            dl_{\\text{{target}}} = \\frac{\\lambda}{\\sigma_{x}^{2}} + \\epsilon^{2}

        .. math::
            \\epsilon^{2} = \\| \\text{target} - \\text{template} \\|^{2}

        Parameters
        ----------
        target : NDArray
            The target electron density map.
        template : NDArray
            The input atomic structure map.
        lbd : float
            The lambda hyperparameter.
        sigma_range : tuple of float, optional
            The range of sigma values for the optimizer. Default is (0.1, 20).

        Returns
        -------
        NDArray
            Simulated electron densities.

        References
        ----------
        .. [1]  Gomez, G (Jan. 2000). Local Smoothness in terms of Variance:
                The Adaptive Gaussian Filter. In Procedings of the British
                Machine Vision Conference 2000.
        """

        class _optimizer(Preprocessor):
            def __init__(self, target, template, lbd):
                self._target = target
                self._template = template
                self._dl = np.full(template.shape, 10**9)
                self._filter = np.zeros_like(template)
                self._lbd = lbd

            def __call__(self, x, *args):
                x = x[0]
                filter = super().gaussian_filter(sigma=x, template=template)
                dl = self._lbd / (x**2) + np.power(self._target - filter, 2)
                ind = dl < self._dl
                self._dl[ind] = dl[ind]
                self._filter[ind] = filter[ind]
                return np.sum(self._dl)

        # This method needs pre normalization
        template = template.copy()
        target = target.copy()
        sd_target = np.std(target)
        sd_template = np.std(template)
        m_target = np.mean(target)
        m_template = np.mean(target)
        if sd_target != 0:
            target = (target - m_target) / sd_target

        if sd_template != 0:
            template = (template - m_template) / sd_template

        temp = _optimizer(target=target, template=template, lbd=lbd)

        _ = differential_evolution(temp, bounds=[sigma_range], seed=2)

        # Make sure there is no negative density
        temp._filter += np.abs(np.min(temp._filter))

        return temp._filter

    def local_gaussian_filter(
        self,
        template: NDArray,
        lbd: float,
        sigma_range: Tuple[float, float],
        gaussian_sigma: float,
    ) -> NDArray:
        """
        Wrapper for `Preprocessor.local_gaussian_alignment_filter` if no
        target is available.

        Parameters
        ----------
        template : NDArray
            The input atomic structure map.
        apix : float
            Ångstrom per voxel passed to `Preprocessor.gaussian_filter`.
        lbd : float
            The lambda hyperparameter, common values: 2, 5, 20.
        sigma_range : tuple of float
            The range of sigma values for the optimizer.
        gaussian_sigma : float
            The sigma value passed to `Preprocessor.gaussian_filter` to
            obtain a target.

        Returns
        -------
        NDArray
            Simulated electron densities.
        """
        filtered_data = self.gaussian_filter(sigma=gaussian_sigma, template=template)
        return self.local_gaussian_alignment_filter(
            target=filtered_data,
            template=template,
            lbd=lbd,
            sigma_range=sigma_range,
        )

    def edge_gaussian_filter(
        self,
        template: NDArray,
        edge_algorithm: str,
        sigma: float,
        reverse: bool = False,
    ) -> NDArray:
        """
        Perform Gaussian filterring according to edges in the input template.

        Parameters
        ----------
        template : NDArray
            The input atomic structure map.
        sigma : NDArray
            The sigma value for the Gaussian filter.
        edge_algorithm : str
            The algorithm used to identify edges.  Options are:

            +-------------------+------------------------------------------------+
            | 'sobel'           | Applies sobel filter for edge detection.       |
            +-------------------+------------------------------------------------+
            | 'prewitt'         | Applies prewitt filter for edge detection.     |
            +-------------------+------------------------------------------------+
            | 'laplace'         | Computes edges as second derivative.           |
            +-------------------+------------------------------------------------+
            | 'gaussian'        | See scipy.ndimage.gaussian_gradient_magnitude  |
            +-------------------+------------------------------------------------+
            | 'gaussian_laplace | See scipy.ndimage.gaussian_laplace             |
            +-------------------+------------------------------------------------+
        reverse : bool, optional
            If true, the filterring is strong along edges. Default is False.

        Returns
        -------
        NDArray
            Simulated electron densities.
        """
        if edge_algorithm == "sobel":
            edges = ndimage.generic_gradient_magnitude(template, ndimage.sobel)
        elif edge_algorithm == "prewitt":
            edges = ndimage.generic_gradient_magnitude(template, ndimage.prewitt)
        elif edge_algorithm == "laplace":
            edges = ndimage.laplace(template)
        elif edge_algorithm == "gaussian":
            edges = ndimage.gaussian_gradient_magnitude(template, sigma / 2)
        elif edge_algorithm == "gaussian_laplace":
            edges = ndimage.gaussian_laplace(template, sigma / 2)
        else:
            raise ValueError(
                "Supported edge_algorithm values are"
                "'sobel', 'prewitt', 'laplace', 'gaussian_laplace', 'gaussian'"
            )
        edges[edges != 0] = 1
        edges /= edges.max()

        edges = ndimage.gaussian_filter(edges, sigma)
        filt = ndimage.gaussian_filter(template, sigma)

        if not reverse:
            res = template * edges + filt * (1 - edges)
        else:
            res = template * (1 - edges) + filt * (edges)

        return res

    def mean_filter(self, template: NDArray, width: NDArray) -> NDArray:
        """
        Perform mean filtering.

        Parameters
        ----------
        template : NDArray
            The input atomic structure map.
        width : NDArray
            Width of the mean filter along each axis. Can either have length
            one or template.ndim.

        Returns
        -------
        NDArray
            Simulated electron densities.
        """
        template = template.copy()
        interpolation_box = template.shape

        width = np.array(width)
        filter_width = np.repeat(width, template.ndim // width.size)
        filter_mask = np.ones(filter_width)
        filter_mask = filter_mask / np.sum(filter_mask)
        template = ndimage.convolve(template, filter_mask, mode="reflect")

        # Sometimes scipy messes up the box sizes ...
        template = self.interpolate_box(box=interpolation_box, arr=template)

        return template

    def kaiserb_filter(self, template: NDArray, width: int) -> NDArray:
        """
        Apply Kaiser filter defined as:

        .. math::
            f_{kaiser} = \\frac{I_{0}(\\beta\\sqrt{1-
            \\frac{4n^{2}}{(M-1)^{2}}})}{I_{0}(\\beta)}
            -\\frac{M-1}{2} \\leq n \\leq \\frac{M-1}{2}
            \\text{With } \\beta=3.2

        Parameters
        ----------
        template : NDArray
            The input atomic structure map.
        width : int
            Width of the filter window.
        normalize : bool, optional
            If true, the output is z-transformed. Default is False.

        Returns
        -------
        NDArray
            Simulated electron densities.

        References
        ----------
        .. [1]  Sorzano, Carlos et al (Mar. 2015). Fast and accurate conversion
                of atomic models into electron density maps. AIMS Biophysics
                2, 8–20.
        """
        template, interpolation_box = template.copy(), template.shape

        kaiser_window = window_kaiserb(width=width)
        template = apply_window_filter(arr=template, filter_window=kaiser_window)

        if not np.all(template.shape == interpolation_box):
            template = self.interpolate_box(box=interpolation_box, arr=template)

        return template

    def blob_filter(self, template: NDArray, width: int) -> NDArray:
        """
        Apply blob filter defined as:

        .. math::
            f_{blob} = \\frac{\\sqrt{1-(\\frac{4n^{2}}{(M-1)^{2}})^{m}} I_{m}
            (\\beta\\sqrt{1-(\\frac{4n^{2}}{(M-1)^{2}})})}
            {I_{m}(\\beta)}
            -\\frac{M-1}{2} \\leq n \\leq \\frac{M-1}{2}
            \\text{With } \\beta=3.2 \\text{ and order=2}

        Parameters
        ----------
        template : NDArray
            The input atomic structure map.
        width : int
            Width of the filter window.

        Returns
        -------
        NDArray
            Simulated electron densities.

        References
        ----------
        .. [1]  Sorzano, Carlos et al (Mar. 2015). Fast and accurate conversion
                of atomic models into electron density maps. AIMS Biophysics
                2, 8–20.
        """
        template, interpolation_box = template.copy(), template.shape

        blob_window = window_blob(width=width)
        template = apply_window_filter(arr=template, filter_window=blob_window)

        if not np.all(template.shape == interpolation_box):
            template = self.interpolate_box(box=interpolation_box, arr=template)

        return template

    def hamming_filter(self, template: NDArray, width: int) -> NDArray:
        """
        Apply Hamming filter defined as:

        .. math::
            f_{hamming} = 0.54 - 0.46\\cos(\\frac{2\\pi n}{M-1})
            0 \\leq n \\leq M-1

        Parameters
        ----------
        template : NDArray
            The input atomic structure map.
        width : int
            Width of the filter window.

        Returns
        -------
        NDArray
            Simulated electron densities.
        """
        template, interpolation_box = template.copy(), template.shape

        hamming_window = np.hamming(int(width))
        hamming_window /= hamming_window.sum()

        template = apply_window_filter(arr=template, filter_window=hamming_window)

        if not np.all(template.shape == interpolation_box):
            template = self.interpolate_box(box=interpolation_box, arr=template)

        return template

    def rank_filter(self, template: NDArray, rank: int) -> NDArray:
        """
        Perform rank filtering.

        Parameters
        ----------
        template : NDArray
            The input atomic structure map.
        rank : int
            Footprint value. 0 -> minimum filter, -1 -> maximum filter.

        Returns
        -------
        NDArray
            Simulated electron densities.
        """
        template = template.copy()
        interpolation_box = template.shape

        size = rank // 2
        if size <= 1:
            size = 3

        template = ndimage.rank_filter(template, rank=rank, size=size)
        template = self.interpolate_box(box=interpolation_box, arr=template)

        return template

    def median_filter(self, template: NDArray, size: int = None) -> NDArray:
        """
        Perform median filtering.

        Parameters
        ----------
        template : NDArray
            The template to be filtered.
        size : int, optional
            Size of the filter.

        Returns
        -------
        NDArray
            Filtered template.
        """
        interpolation_box = template.shape

        template = ndimage.median_filter(template, size=size)
        template = self.interpolate_box(box=interpolation_box, arr=template)

        return template

    def mipmap_filter(self, template: NDArray, level: int) -> NDArray:
        """
        Perform mip map antialiasing filtering.

        Parameters
        ----------
        template : NDArray
            The input atomic structure map.
        level : int
            Pyramid layer. Resolution decreases cubically with level.

        Returns
        -------
        NDArray
            Simulated electron densities.
        """
        array = template.copy()
        interpolation_box = array.shape

        for k in range(template.ndim):
            array = ndimage.decimate(array, q=level, axis=k)

        template = ndimage.zoom(array, np.divide(template.shape, array.shape))
        template = self.interpolate_box(box=interpolation_box, arr=template)

        return template

    def interpolate_box(
        self, arr: NDArray, box: Tuple[int], kind: str = "nearest"
    ) -> NDArray:
        """
        Resample ``arr`` within ``box`` using ``kind`` interpolation.

        Parameters
        ----------
        arr : NDArray
            The input numpy array.
        box : tuple of int
            Tuple of integers corresponding to the shape of the output array.
        kind : str, optional
            Interpolation method used (see scipy.interpolate.interp1d).
            Default is 'nearest'.

        Raises
        ------
        ValueError
            If the shape of box does not match arr.ndim

        Returns
        -------
        NDArray
            Interpolated numpy array.
        """
        if len(box) != arr.ndim:
            raise ValueError(f"Expected box of {arr.ndim}, got {len(box)}")

        for axis, size in enumerate(box):
            f = interp1d(
                np.linspace(0, 1, arr.shape[axis]),
                arr,
                kind=kind,
                axis=axis,
                fill_value="extrapolate",
            )
            arr = f(np.linspace(0, 1, size))

        return arr

    def bandpass_filter(
        self,
        template: NDArray,
        lowpass: float,
        highpass: float,
        sampling_rate: NDArray = None,
        gaussian_sigma: float = 0.0,
    ) -> NDArray:
        """
        Apply a band-pass filter on the provided template, using a
        Butterworth approximation.

        Parameters
        ----------
        template : NDArray
            The input numpy array on which the band-pass filter should be applied.
        lowpass : float
            The lower boundary of the frequency range to be preserved. Lower values will
            retain broader, more global features.
        highpass : float
            The upper boundary of the frequency range to be preserved.  Higher values
            will emphasize finer details and potentially noise.
        sampling_rate : NDarray, optional
            The sampling rate along each dimension.
        gaussian_sigma : float, optional
            Sigma value for the gaussian smoothing to be applied to the filter.

        Returns
        -------
        NDArray
            Bandpass filtered numpy array.
        """
        bpf = self.bandpass_mask(
            shape=template.shape,
            lowpass=lowpass,
            highpass=highpass,
            sampling_rate=sampling_rate,
            gaussian_sigma=gaussian_sigma,
            omit_negative_frequencies=False,
        )

        fft_data = np.fft.fftn(template)
        np.multiply(fft_data, bpf, out=fft_data)
        ret = np.real(np.fft.ifftn(fft_data))
        return ret

    def bandpass_mask(
        self,
        shape: Tuple[int],
        lowpass: float,
        highpass: float,
        sampling_rate: NDArray = None,
        gaussian_sigma: float = 0.0,
        omit_negative_frequencies: bool = True,
    ) -> NDArray:
        """
        Compute an approximate Butterworth bundpass filter. The returned filter
        has it's DC component at the origin.

        Parameters
        ----------
        shape : tuple of ints
            Shape of the returned bandpass filter.
        lowpass : float
            The lower boundary of the frequency range to be preserved. Lower values will
            retain broader, more global features.
        maximum_frequency : float
            The upper boundary of the frequency range to be preserved.  Higher values
            will emphasize finer details and potentially noise.
        sampling_rate : NDarray, optional
            The sampling rate along each dimension.
        gaussian_sigma : float, optional
            Sigma value for the gaussian smoothing to be applied to the filter.
        omit_negative_frequencies : bool, optional
            Whether the wedge mask should omit negative frequencies, i.e. be
            applicable to non hermitian-symmetric fourier transforms.

        Returns
        -------
        NDArray
            Bandpass filtered.
        """
        from .preprocessing import BandPassFilter

        return BandPassFilter(
            sampling_rate=sampling_rate,
            lowpass=lowpass,
            highpass=highpass,
            return_real_fourier=omit_negative_frequencies,
            use_gaussian=gaussian_sigma == 0.0,
        )(shape=shape)["data"]

    def step_wedge_mask(
        self,
        shape: Tuple[int],
        tilt_angles: Tuple[float] = None,
        opening_axis: int = 0,
        tilt_axis: int = 2,
        weights: float = None,
        infinite_plane: bool = False,
        omit_negative_frequencies: bool = True,
    ) -> NDArray:
        """
        Create a wedge mask with the same shape as template by rotating a
        plane according to tilt angles. The DC component of the filter is at the origin.

        Parameters
        ----------
        tilt_angles : tuple of float
            Sequence of tilt angles.
        shape : Tuple of ints
            Shape of the output wedge array.
        tilt_axis : int, optional
            Axis that the plane is tilted over.
            - 0 for Z-axis
            - 1 for Y-axis
            - 2 for X-axis
        opening_axis : int, optional
            Axis running through the void defined by the wedge.
            - 0 for Z-axis
            - 1 for Y-axis
            - 2 for X-axis
        sigma : float, optional
            Standard deviation for Gaussian kernel used for smoothing the wedge.
        weights : float, tuple of float
            Weight of each element in the wedge. Defaults to one.
        omit_negative_frequencies : bool, optional
            Whether the wedge mask should omit negative frequencies, i.e. be
            applicable to symmetric Fourier transforms (see :obj:`numpy.fft.fftn`)

        Returns
        -------
        NDArray
            A numpy array containing the wedge mask.

        See Also
        --------
        :py:meth:`Preprocessor.continuous_wedge_mask`
        """
        from .preprocessing.tilt_series import WedgeReconstructed

        return WedgeReconstructed(
            angles=tilt_angles,
            tilt_axis=tilt_axis,
            opening_axis=opening_axis,
            frequency_cutoff=None if infinite_plane else 0.5,
            create_continuous_wedge=False,
            weights=weights,
            weight_wedge=weights is not None,
        )(shape=shape, return_real_fourier=omit_negative_frequencies,)["data"]

    def continuous_wedge_mask(
        self,
        start_tilt: float,
        stop_tilt: float,
        shape: Tuple[int],
        opening_axis: int = 0,
        tilt_axis: int = 2,
        infinite_plane: bool = True,
        omit_negative_frequencies: bool = True,
    ) -> NDArray:
        """
        Generate a wedge in a given shape based on specified tilt angles and axis.
        The DC component of the filter is at the origin.

        Parameters
        ----------
        start_tilt : float
            Starting tilt angle in degrees, e.g. a stage tilt of 70 degrees
            would yield a start_tilt value of 70.
        stop_tilt : float
            Ending tilt angle in degrees, , e.g. a stage tilt of -70 degrees
            would yield a stop_tilt value of 70.
        tilt_axis : int
            Axis that the plane is tilted over.
            - 0 for Z-axis
            - 1 for Y-axis
            - 2 for X-axis
        opening_axis : int
            Axis running through the void defined by the wedge.
            - 0 for Z-axis
            - 1 for Y-axis
            - 2 for X-axis
        shape : Tuple of ints
            Shape of the output wedge array.
        omit_negative_frequencies : bool, optional
            Whether the wedge mask should omit negative frequencies, i.e. be
            applicable to symmetric Fourier transforms (see :obj:`numpy.fft.fftn`)
        infinite_plane : bool, optional
            Whether the plane should be considered to be larger than the shape. In this
            case the output wedge mask fill have no spheric component.

        Returns
        -------
        NDArray
            Array of the specified shape with the wedge created based on
            the tilt angles.

        See Also
        --------
        :py:meth:`Preprocessor.step_wedge_mask`
        """
        from .preprocessing.tilt_series import WedgeReconstructed

        return WedgeReconstructed(
            angles=(start_tilt, stop_tilt),
            tilt_axis=tilt_axis,
            opening_axis=opening_axis,
            frequency_cutoff=None if infinite_plane else 0.5,
            create_continuous_wedge=True,
        )(shape=shape, return_real_fourier=omit_negative_frequencies)["data"]


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
        ndimage.correlate1d(
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

    f_mask = ndimage.convolve(profile, mask)

    orig = int((f_mask.size - 1) / 2)
    dist = np.arange(-orig, orig + 1) * T
    t, c, k = splrep(x=dist, y=f_mask, k=3)
    i_max = np.ceil(np.divide(f_mask.shape, M)).astype(int)[0]
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
        profile = ndimage.convolve(profile, window)

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
