""" Implements Preprocessor class for filtering operations.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import inspect
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from scipy.ndimage import (
    fourier_gaussian,
    gaussian_filter,
    rank_filter,
    zoom,
    generic_gradient_magnitude,
    sobel,
    prewitt,
    laplace,
    gaussian_laplace,
    gaussian_gradient_magnitude,
)
from scipy.signal import convolve, decimate
from scipy.optimize import differential_evolution
from pywt import wavelist, wavedecn, waverecn
from scipy.interpolate import interp1d

from .density import Density
from .helpers import (
    window_kaiserb,
    window_blob,
    apply_window_filter,
    Ntree,
)
from .matching_utils import euler_to_rotationmatrix


class Preprocessor:
    """
    Implements filtering operations on density arrays.
    """

    def apply_method(self, method: str, parameters: Dict):
        """
        Apply a method on the atomic structure.

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
        method_to_call = getattr(self, method)
        return method_to_call(**parameters)

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

    @staticmethod
    def _gaussian_fourier(template: NDArray, sigma: NDArray) -> NDArray:
        """
        Apply a Gaussian filter in Fourier space on the provided template.

        Parameters
        ----------
        template : NDArray
            The input template on which to apply the filter.
        sigma : NDArray
            The standard deviation for Gaussian kernel. The greater the value,
            the more spread out is the filter.

        Returns
        -------
        NDArray
            The template after applying the Fourier Gaussian filter.
        """
        fourrier_map = fourier_gaussian(np.fft.fftn(template), sigma)
        template = np.real(np.fft.ifftn(fourrier_map))

        return template

    @staticmethod
    def _gaussian_real(
        template: NDArray, sigma: NDArray, cutoff_value: float = 4.0
    ) -> NDArray:
        """
        Apply a Gaussian filter on the provided template in real space.

        Parameters
        ----------
        template : NDArray
            The input template on which to apply the filter.
        sigma : NDArray
            The standard deviation for Gaussian kernel. The greater the value,
            the more spread out is the filter.
        cutoff_value : float, optional
            The value below which the data should be ignored. Default is 4.0.

        Returns
        -------
        NDArray
            The template after applying the Gaussian filter in real space.
        """
        template = gaussian_filter(template, sigma, cval=cutoff_value)
        return template

    def gaussian_filter(
        self,
        template: NDArray,
        sigma: NDArray,
        fourier: bool = False,
    ) -> NDArray:
        """
        Convolve an atomic structure with a Gaussian kernel.

        Parameters
        ----------
        template : NDArray
            The input atomic structure map.
        resolution : float, optional
            The resolution. The product of `resolution` and `sigma_coeff` is used
            to compute the `sigma` for the discretized Gaussian. Default is None.
        sigma : NDArray
            The standard deviation for Gaussian kernel. Should either be a scalar
            or a sequence of scalars.
        fourier : bool, optional
            If true, applies a Fourier Gaussian filter; otherwise, applies a
            real-space Gaussian filter. Default is False.

        Returns
        -------
        NDArray
            The simulated electron densities after applying the Gaussian filter.
        """
        sigma = 0 if sigma is None else sigma

        if sigma <= 0:
            return template

        func = self._gaussian_real if not fourier else self._gaussian_fourier
        template = func(template, sigma)

        return template

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
        im1 = self._gaussian_real(template, low_sigma)
        im2 = self._gaussian_real(template, high_sigma)
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
            edges = generic_gradient_magnitude(template, sobel)
        elif edge_algorithm == "prewitt":
            edges = generic_gradient_magnitude(template, prewitt)
        elif edge_algorithm == "laplace":
            edges = laplace(template)
        elif edge_algorithm == "gaussian":
            edges = gaussian_gradient_magnitude(template, sigma / 2)
        elif edge_algorithm == "gaussian_laplace":
            edges = gaussian_laplace(template, sigma / 2)
        else:
            raise ValueError(
                "Supported edge_algorithm values are"
                "'sobel', 'prewitt', 'laplace', 'gaussian_laplace', 'gaussian'"
            )
        edges[edges != 0] = 1
        edges /= edges.max()

        edges = gaussian_filter(edges, sigma)
        filter = gaussian_filter(template, sigma)

        if not reverse:
            res = template * edges + filter * (1 - edges)
        else:
            res = template * (1 - edges) + filter * (edges)

        return res

    def ntree_filter(
        self,
        template: NDArray,
        sigma_range: Tuple[float, float],
        target: NDArray = None,
    ) -> NDArray:
        """
        Use dyadic tree to identify volume partitions in *template*
        and filter them with respect to their occupancy.

        Parameters
        ----------
        template : NDArray
            The input atomic structure map.
        sigma_range : tuple of float
            Range of sigma values used to filter volume partitions.
        target : NDArray, optional
            If provided, dyadic tree is computed on target rather than template.

        Returns
        -------
        NDArray
            Simulated electron densities.
        """
        if target is None:
            target = template

        tree = Ntree(target)

        filter = tree.filter_chunks(arr=template, sigma_range=sigma_range)

        return filter

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
        template = convolve(template, filter_mask, mode="same")

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
        if template is None:
            raise ValueError("Argument template missing")
        template = template.copy()
        interpolation_box = template.shape

        size = rank // 2
        if size <= 1:
            size = 3

        template = rank_filter(template, rank=rank, size=size)
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

        print(array.shape)

        for k in range(template.ndim):
            array = decimate(array, q=level, axis=k)

        print(array.shape)
        template = zoom(array, np.divide(template.shape, array.shape))
        template = self.interpolate_box(box=interpolation_box, arr=template)

        return template

    def wavelet_filter(
        self,
        template: NDArray,
        level: int,
        wavelet: str = "bior2.2",
    ) -> NDArray:
        """
        Perform dyadic wavelet decomposition.

        Parameters
        ----------
        template : NDArray
            The input atomic structure map.
        level : int
            Scale of the wavelet transform.
        wavelet : str, optional
            Mother wavelet used for decomposition. Default is 'bior2.2'.

        Returns
        -------
        NDArray
            Simulated electron densities.
        """
        if wavelet not in wavelist(kind="discrete"):
            raise NotImplementedError(
                "Print argument wavelet has to be one of the following: %s",
                ", ".join(wavelist(kind="discrete")),
            )

        template, interpolation_box = template.copy(), template.shape
        decomp = wavedecn(template, level=level, wavelet=wavelet)

        for i in range(1, level + 1):
            decomp[i] = {k: np.zeros_like(v) for k, v in decomp[i].items()}

        template = waverecn(coeffs=decomp, wavelet=wavelet)
        template = self.interpolate_box(box=interpolation_box, arr=template)

        return template

    @staticmethod
    def molmap(
        coordinates: NDArray,
        weights: Tuple[float],
        resolution: float,
        sigma_factor: float = 1 / (np.pi * np.sqrt(2)),
        cutoff_value: float = 5.0,
        origin: Tuple[float] = None,
        shape: Tuple[int] = None,
        sampling_rate: float = None,
    ) -> NDArray:
        """
        Compute the electron densities analogous to Chimera's molmap function.

        Parameters
        ----------
        coordinates : NDArray
            A N x 3 array containing atomic coordinates in z, y, x format.
        weights : [float]
            The weights to use for the entries in coordinates.
        resolution : float
            The product of resolution and sigma_factor gives the sigma used to
            compute the discretized Gaussian.
        sigma_factor : float
            The factor used with resolution to compute sigma. Default is 1 / (π√2).
        cutoff_value : float
            The cutoff value for the Gaussian kernel. Default is 5.0.
        origin : (float,)
            The origin of the coordinate system used in coordinates. If not specified,
            the minimum coordinate along each axis is used.
        shape : (int,)
            The shape of the output array. If not specified, the function computes the
            smallest output array that contains all atoms.
        sampling_rate : float
            The Ångstrom per voxel of the output array. If not specified, the function
            sets this value to resolution/3.

        References
        ----------
        ..[1] https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/midas/molmap.html

        Returns
        -------
        NDArray
            A numpy array containing the simulated electron densities.
        """
        if sampling_rate is None:
            sampling_rate = resolution * (1.0 / 3)

        coordinates = coordinates.copy()
        if origin is None:
            origin = coordinates.min(axis=0)
        if shape is None:
            positions = (coordinates - origin) / sampling_rate
            shape = positions.max(axis=0).astype(int)[::-1] + 2

        positions = (coordinates - origin) / sampling_rate
        positions = positions[:, ::-1]

        out = np.zeros(shape, dtype=np.float32)
        sigma = sigma_factor * resolution
        sigma_grid = sigma / sampling_rate
        sigma_grid2 = sigma_grid * sigma_grid
        for index, point in enumerate(np.rollaxis(positions, 0)):
            starts = np.maximum(np.ceil(point - cutoff_value * sigma_grid), 0).astype(
                int
            )
            stops = np.minimum(
                np.floor(point + cutoff_value * sigma_grid), shape
            ).astype(int)

            grid_index = np.meshgrid(
                *[range(start, stop) for start, stop in zip(starts, stops)]
            )
            distances = np.einsum(
                "aijk->ijk",
                np.array([(grid_index[i] - point[i]) ** 2 for i in range(len(point))]),
                dtype=np.float64,
            )
            np.add.at(
                out,
                tuple(grid_index),
                weights[index] * np.exp(-0.5 * distances / sigma_grid2),
            )

        out *= np.power(2 * np.pi, -1.5) * np.power(sigma, -3)
        return out

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

    @staticmethod
    def fftfreqn(shape: NDArray, sampling_rate: NDArray) -> NDArray:
        """
        Calculate the N-dimensional equivalent to the inverse fftshifted
        absolute of numpy's fftfreq function, supporting anisotropic sampling.

        Parameters
        ----------
        shape : NDArray
            The shape of the N-dimensional array.
        sampling_rate : NDArray
            The sampling rate in the N-dimensional array.

        Returns
        -------
        NDArray
            A numpy array representing the norm of indices after normalization.

        Examples
        --------
        >>> import numpy as np
        >>> from dge import Preprocessor
        >>> freq = Preprocessor().fftfreqn((10,), 1)
        >>> freq_numpy = np.fft.fftfreq(10, 1)
        >>> np.allclose(freq, np.abs(np.fft.ifftshift(freq_numpy)))
        """
        indices = np.indices(shape).T
        norm = np.multiply(shape, sampling_rate)
        indices -= np.divide(shape, 2).astype(int)
        indices = np.divide(indices, norm)
        return np.linalg.norm(indices, axis=-1).T

    def _approximate_butterworth(
        self,
        radial_frequencies: NDArray,
        lowcut: float,
        highcut: float,
        gaussian_sigma: float,
    ) -> NDArray:
        """
        Approximate a Butterworth band-pass filter for given radial frequencies.
        The DC component of the filter is at the origin.

        Parameters
        ----------
        radial_frequencies : NDArray
            The radial frequencies for which the Butterworth band-pass
            filter is to be calculated.
        lowcut : float
            The lower cutoff frequency for the band-pass filter.
        highcut : float
            The upper cutoff frequency for the band-pass filter.
        gaussian_sigma : float
            The sigma value for the Gaussian smoothing applied to the filter.

        Returns
        -------
        NDArray
            A numpy array representing the approximate Butterworth
            band-pass filter applied to the radial frequencies.
        """
        bpf = ((radial_frequencies <= highcut) & (radial_frequencies >= lowcut)) * 1.0
        bpf = self.gaussian_filter(template=bpf, sigma=gaussian_sigma, fourier=False)
        bpf[bpf < np.exp(-2)] = 0
        bpf = np.fft.ifftshift(bpf)

        return bpf

    def bandpass_filter(
        self,
        template: NDArray,
        minimum_frequency: float,
        maximum_frequency: float,
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
        minimum_frequency : float
            The lower boundary of the frequency range to be preserved. Lower values will
            retain broader, more global features.
        maximum_frequency : float
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
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
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
        minimum_frequency: float,
        maximum_frequency: float,
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
        minimum_frequency : float
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
        if sampling_rate is None:
            sampling_rate = np.ones(len(shape))
        sampling_rate = np.asarray(sampling_rate, dtype=np.float32)
        sampling_rate /= sampling_rate.max()

        if minimum_frequency > maximum_frequency:
            minimum_frequency, maximum_frequency = maximum_frequency, minimum_frequency

        radial_freq = self.fftfreqn(shape, sampling_rate)
        bpf = self._approximate_butterworth(
            radial_frequencies=radial_freq,
            lowcut=minimum_frequency,
            highcut=maximum_frequency,
            gaussian_sigma=gaussian_sigma,
        )

        if omit_negative_frequencies:
            stop = 1 + (shape[-1] // 2)
            bpf = bpf[..., :stop]

        return bpf

    def wedge_mask(
        self,
        shape: Tuple[int],
        tilt_angles: NDArray,
        sigma: float = 0,
        omit_negative_frequencies: bool = True,
    ) -> NDArray:
        """
        Create a wedge mask with the same shape as template by rotating a
        plane according to tilt angles. The DC component of the filter is at the origin.

        Parameters
        ----------
        shape : Tuple of ints
            Shape of the output wedge array.
        tilt_angles : NDArray
            Tilt angles in format d dimensions N tilts [d x N].
        sigma : float, optional
            Standard deviation for Gaussian kernel used for smoothing the wedge.
        omit_negative_frequencies : bool, optional
            Whether the wedge mask should omit negative frequencies, i.e. be
            applicable to non hermitian-symmetric fourier transforms.

        Returns
        -------
        NDArray
            A numpy array containing the wedge mask.

        Notes
        -----
        The axis perpendicular to the tilts is the leftmost closest axis
        with minimal tilt.

        Examples
        --------
        >>> import numpy as np
        >>> from tme import Preprocessor
        >>> angles = np.zeros((3, 10))
        >>> angles[2, :] = np.linspace(-50, 55, 10)
        >>> wedge = Preprocessor().wedge_mask(
        >>>    shape = (50,50,50),
        >>>    tilt_angles = angles,
        >>>    omit_negative_frequencies = True
        >>>    )
        >>> wedge = np.fft.fftshift(wedge)

        This will create a wedge that is open along axis 1, tilted
        around axis 2 and propagated along axis 0. The code above would
        be equivalent to the following

        >>> wedge = Preprocessor().continuous_wedge_mask(
        >>>    shape = (50,50,50),
        >>>    start_tilt = 50,
        >>>    stop_tilt=55,
        >>>    tilt_axis=1,
        >>>    omit_negative_frequencies=False,
        >>>    infinite_plane=False
        >>>    )
        >>> wedge = np.fft.fftshift(wedge)

        with the difference being that :py:meth:`Preprocessor.continuous_wedge_mask`
        does not consider individual plane tilts.

        See Also
        --------
        :py:meth:`Preprocessor.continuous_wedge_mask`
        """
        plane = np.zeros(shape, dtype=np.float32)
        opening_axis = np.argmax(np.abs(tilt_angles), axis=0)
        slices = tuple(slice(a, a + 1) for a in np.divide(shape, 2).astype(int))
        plane_rotated = np.zeros_like(plane)
        wedge_volume = np.zeros_like(plane)
        for index in range(tilt_angles.shape[1]):
            potential_axes, *_ = np.where(
                np.abs(tilt_angles[:, index]) == np.abs(tilt_angles[:, index]).min()
            )
            largest_tilt = np.argmax(np.abs(tilt_angles[:, index]))
            opening_axis_index = np.argmin(np.abs(potential_axes - largest_tilt))
            opening_axis = potential_axes[opening_axis_index]
            rotation_matrix = euler_to_rotationmatrix(tilt_angles[:, index])
            plane_rotated.fill(0)
            plane.fill(0)
            subset = tuple(
                slice(None) if i != opening_axis else slices[opening_axis]
                for i in range(plane.ndim)
            )
            plane[subset] = 1
            Density.rotate_array(
                arr=plane,
                rotation_matrix=rotation_matrix,
                out=plane_rotated,
                use_geometric_center=True,
                order=1,
            )
            wedge_volume += plane_rotated

        wedge_volume = self.gaussian_filter(
            template=wedge_volume, sigma=sigma, fourier=False
        )
        wedge_volume = np.where(wedge_volume > np.exp(-2), 1, 0)
        wedge_volume = np.fft.ifftshift(wedge_volume)

        if omit_negative_frequencies:
            stop = 1 + (wedge_volume.shape[-1] // 2)
            wedge_volume = wedge_volume[..., :stop]

        return wedge_volume

    def continuous_wedge_mask(
        self,
        start_tilt: float,
        stop_tilt: float,
        shape: Tuple[int],
        tilt_axis: int = 1,
        sigma: float = 0,
        extrude_plane: bool = True,
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
            Axis that runs through the empty part of the wedge.
            - 0 for X-axis
            - 1 for Y-axis
            - 2 for Z-axis
        shape : Tuple of ints
            Shape of the output wedge array.
        sigma : float, optional
            Standard deviation for Gaussian kernel used for smoothing the wedge.
        extrude_plane : bool, optional
            Whether the tilted plane is extruded to 3D. By default, this represents
            the effect of rotating a plane in 3D yielding a cylinder with wedge
            insertion. If set to False, the returned mask has spherical shape,
            analogous to rotating a line in 3D.
        omit_negative_frequencies : bool, optional
            Whether the wedge mask should omit negative frequencies, i.e. be
            applicable to non hermitian-symmetric fourier transforms.
        infinite_plane : bool, optional
            Whether the plane should be considered to be larger than the shape. In this
            case the output wedge mask fill have no spheric component.

        Returns
        -------
        NDArray
            Array of the specified shape with the wedge created based on
            the tilt angles.

        Examples
        --------
        >>> wedge = create_wedge(30, 60, 1, (64, 64, 64))

        Notes
        -----
        The rotation plane is spanned by the tilt axis and the leftmost dimension
        that is not the tilt axis.

        See Also
        --------
        :py:meth:`Preprocessor.wedge_mask`
        """
        shape_center = np.divide(shape, 2).astype(int)

        opening_axis = tilt_axis
        base_axis = (tilt_axis + 1) % len(shape)

        grid = (np.indices(shape).T - shape_center).T

        start_radians = np.tan(np.radians(90 - start_tilt))
        stop_radians = np.tan(np.radians(-1 * (90 - stop_tilt)))
        max_tan_value = np.tan(np.radians(90)) + 1

        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(
                grid[opening_axis] == 0,
                max_tan_value,
                grid[base_axis] / grid[opening_axis],
            )

        wedge = np.logical_or(start_radians <= ratios, stop_radians >= ratios).astype(
            np.float32
        )

        if extrude_plane:
            distances = np.sqrt(grid[base_axis] ** 2 + grid[opening_axis] ** 2)
        else:
            distances = np.linalg.norm(grid, axis=0)

        if not infinite_plane:
            np.multiply(wedge, distances <= shape[opening_axis] // 2, out=wedge)

        wedge = self.gaussian_filter(template=wedge, sigma=sigma, fourier=False)
        wedge = np.fft.ifftshift(wedge > np.exp(-2))

        if omit_negative_frequencies:
            stop = 1 + (wedge.shape[-1] // 2)
            wedge = wedge[..., :stop]

        return wedge

    @staticmethod
    def _fourier_crop_mask(old_shape: NDArray, new_shape: NDArray) -> NDArray:
        """
        Generate a mask for Fourier cropping.

        Parameters
        ----------
        old_shape : NDArray
            The original shape of the array before cropping.
        new_shape : NDArray
            The new desired shape for the array after cropping.

        Returns
        -------
        NDArray
            The mask array for Fourier cropping.
        """
        mask = np.zeros(old_shape, dtype=bool)
        mask[tuple(np.indices(new_shape))] = 1
        box_shift = np.floor(np.divide(new_shape, 2)).astype(int)
        mask = np.roll(mask, shift=-box_shift, axis=range(len(old_shape)))
        return mask

    def fourier_crop(
        self,
        template: NDArray,
        reciprocal_template_filter: NDArray,
        crop_factor: float = 3 / 2,
    ) -> NDArray:
        """
        Perform Fourier uncropping on a given template.

        Parameters
        ----------
        template : NDArray
            The original template to be uncropped.
        reciprocal_template_filter : NDArray
            The filter to be applied in the Fourier space.
        crop_factor : float
            Cropping factor over reeciprocal_template_filter boundary.

        Returns
        -------
        NDArray
            The uncropped template.
        """
        new_boxsize = np.zeros(template.ndim, dtype=int)
        for i in range(template.ndim):
            slices = tuple(
                slice(0, 1) if j != i else slice(template.shape[i] // 2)
                for j in range(template.ndim)
            )
            filt = np.squeeze(reciprocal_template_filter[slices])
            new_boxsize[i] = np.ceil((np.max(np.where(filt > 0)) + 1) * crop_factor) * 2

        if np.any(np.greater(new_boxsize, template.shape)):
            new_boxsize = np.array(template.shape).copy()

        mask = self._fourier_crop_mask(old_shape=template.shape, new_shape=new_boxsize)
        arr_ft = np.fft.fftn(template)
        arr_ft *= np.prod(new_boxsize) / np.prod(template.shape)
        arr_ft = np.reshape(arr_ft[mask], new_boxsize)
        arr_cropped = np.real(np.fft.ifftn(arr_ft))
        return arr_cropped

    def fourier_uncrop(
        self, template: NDArray, reciprocal_template_filter: NDArray
    ) -> NDArray:
        """
        Perform an uncrop operation in the Fourier space.

        Parameters
        ----------
        template : NDArray
            The input array.
        reciprocal_template_filter : NDArray
            The filter to be applied in the Fourier space.

        Returns
        -------
        NDArray
            Uncropped template with shape reciprocal_template_filter.
        """
        mask = self._fourier_crop_mask(
            old_shape=reciprocal_template_filter.shape, new_shape=template.shape
        )
        ft_vol = np.zeros_like(mask)
        ft_vol[mask] = np.fft.fftn(template).ravel()
        ft_vol *= np.divide(np.prod(mask.shape), np.prod(template.shape)).astype(
            ft_vol.dtype
        )
        reciprocal_template_filter = reciprocal_template_filter.astype(ft_vol.dtype)
        np.multiply(ft_vol, reciprocal_template_filter, out=ft_vol)
        ret = np.real(np.fft.ifftn(ft_vol))
        return ret
