from functools import partial
from abc import ABC, abstractmethod
from typing import Tuple, Type, Optional

import numpy as np

from ..types import BackendArray, NDArray
from .peaks import PeakCaller
from ..backends import backend as be


class AbstractKernel(ABC):
    """Specification for kernel fitting kernels."""

    @abstractmethod
    def fit_batch(self, data, *args, **kwargs) -> tuple:
        """
        Fit kernel to given input data.

        Parameters
        ----------
        data : BackendArray
            Data with leading bactch dimension to fit kernel to.
        **kwargs:
            Kernel specific keyword arguments.

        Returns
        -------
        tuple
            Tuple of parameters.
        """

    @abstractmethod
    def score_batch(self, observation, *args, **kwargs) -> BackendArray:
        """
        Compare the observed parameters to the internal baseline

        Parameters
        ----------
        observation : tuple
            Parameters from :py:meth:`AbstractKernel.estimate_parameters_batch`.

        Returns
        -------
        BackendArray
            Similarity metric comparing for each element in observation.
        """

    @abstractmethod
    def rotate_masked(self, rotation_matrix) -> None:
        """
        Rotate the internal kernel parameters and applies an optional Fourier mask.

        Parameters
        ----------
        rotation_matrix : BackendArray
            Rotation matrix (d,d).
        """


def _make_grid(shape: Tuple[int, ...]) -> Tuple[BackendArray, ...]:
    from tme.filters._utils import fftfreqn

    grid = fftfreqn(shape, sampling_rate=0.5, fftshift=True)
    return grid.reshape(len(shape), -1).T


class GaussianKernel(AbstractKernel):

    def __init__(self, reference, ft_mask=None):
        if ft_mask is None:
            ft_mask = np.full(reference.shape, dtype=be._float_dtype, fill_value=1)

        self.ft_mask = be.to_numpy_array(ft_mask)

        self.nls = GaussianNLS()
        self.bayes = GaussianBayes()
        self.fit_rot = be.add(reference, 0)

        self.reference = reference

        params = self.nls.fit(be.to_numpy_array(reference))

        self.params_ft = self._ft_params(params)
        self.params_natural = self._natural_params(self.params_ft)

        # from tme import Density
        # Density(be.to_numpy_array(self.fit_rot)).to_file("baseline.mrc")
        # baseline_fit = self.nls.sample(params, self.ft_mask.shape)
        # Density(be.to_numpy_array(baseline_fit)).to_file("baseline_fit.mrc")
        # print(self.params_ft[2], "init", self.params_ft[2].max())

    @staticmethod
    def _natural_params(params):
        return be.to_backend_array(get_natural_param(*params))

    @staticmethod
    def _ft_params(params):
        return (params[0], params[1], np.linalg.inv(params[2]))

    def fit_batch(self, data, *args, **kwargs):
        return self.bayes.fit_batch(
            data,
            like_prec=data * self.fit_rot,
            standardize=False,
            *args,
            **kwargs,
        )

    def score_batch(self, observation: Tuple, **kwargs) -> BackendArray:
        return self.bayes.score_batch(
            (self.params_natural, None), observation, **kwargs
        )

    @staticmethod
    def _dist(data, target):
        data = data * target
        fit_rot_sq = target * target

        axes = tuple(i for i in range(1, data.ndim))

        def _norm(x, axis=-1):
            return be.sqrt(be.sum(be.square(x), axis=axis))

        err = be.sum(be.square(data - fit_rot_sq), axis=axes)
        # err = be.sum(be.multiply(data, fit_rot_sq), axis=axes)
        norm = _norm(data, axis=axes) * _norm(fit_rot_sq, axis=None)
        err = be.where(norm > 1e-8, err / norm, 1e10)
        return err

    def dist(self, data: BackendArray, **kwargs) -> BackendArray:
        return self._dist(data, self.fit_rot)

    def dist_xd(self, data: BackendArray, **kwargs) -> BackendArray:
        return self._dist(data, self.reference)

    def rotate_masked(self, rotation_matrix) -> None:
        params_ft_rot = self.nls.rotate(
            be.to_numpy_array(rotation_matrix), self.params_ft
        )

        nls_scale = (self.ft_mask.shape[0] // 3) / params_ft_rot[2].max()
        params_ft_rot = (
            params_ft_rot[0],
            params_ft_rot[1],
            params_ft_rot[2] * nls_scale,
        )

        # from tme import Density
        # fit_rot = self.nls.sample(params_ft_rot, self.ft_mask.shape)
        # Density(fit_rot).to_file("params_ft_rot.mrc")

        ft_rot = self.nls.sample(params_ft_rot, self.ft_mask.shape)
        ft_rot *= self.ft_mask

        # Density(be.to_numpy_array(ft_rot)).to_file("ft_rot.mrc")

        fit_ft_rot = self.nls.fit(data=ft_rot, p0=params_ft_rot)
        fit_ft_rot = (fit_ft_rot[0], fit_ft_rot[1], fit_ft_rot[2] / nls_scale)

        self.params_natural = self._natural_params(fit_ft_rot)

        # Sample data for like prec
        fit_rot = self._ft_params(fit_ft_rot)
        fit_rot = self.nls.sample(fit_rot, self.ft_mask.shape)
        self.fit_rot = be.to_backend_array(fit_rot)
        # Density(fit_rot).to_file("fit_rot.mrc")


class GaussianBayes:
    @staticmethod
    def _build_design_matrix(shape):
        ndim = len(shape)
        grid = _make_grid(shape)
        grid = be.to_backend_array(grid)

        cov = grid[..., np.newaxis] @ grid[:, np.newaxis]
        cov = cov.reshape(grid.shape[0], -1)

        row_indices, col_indices = be.triu_indices(ndim)
        flattened_indices = row_indices * ndim + col_indices

        cov_scaling = be.full(
            be.size(flattened_indices), fill_value=-1, dtype=be._float_dtype
        )
        cov_scaling[be.to_backend_array(row_indices == col_indices)] = -0.5

        phi = be.ones((grid.shape[0], 1 + ndim + be.size(flattened_indices)))
        phi[:, 1 : (ndim + 1)] = grid
        phi[:, (ndim + 1) :] = cov_scaling * cov[:, flattened_indices]

        return be.to_backend_array(phi)

    @staticmethod
    def _standardize(data, phi, like_prec=None):
        # Remove intercept
        phi = phi[..., 1:]

        has_batch = phi.ndim == 3
        param_axes = 0 + int(has_batch)
        data_axes = tuple(range(1, data.ndim)) if has_batch else None
        shape = (data.shape[0], -1, 1) if has_batch else (-1, 1)

        if like_prec is not None:
            weights = like_prec.reshape(*shape)
            weight = be.sum(weights, axis=param_axes, keepdims=has_batch)
            means = be.sum(phi * weights, axis=param_axes, keepdims=has_batch)
            means = be.divide(means, weight, out=means)
            data_mean = be.mean(like_prec * data, axis=data_axes, keepdims=has_batch)

        else:
            means = be.mean(phi, axis=param_axes)
            data_mean = be.mean(data, axis=data_axes, keepdims=has_batch)

        data = be.subtract(data, data_mean, out=data)

        stds = be.std(phi, axis=param_axes)
        phi = be.subtract(phi, means)
        phi = be.divide(phi, stds, out=phi)

        if has_batch and like_prec is not None:
            # n x d
            means = means[:, 0, :]

        return phi, data, data_mean.reshape(-1), means, stds

    @staticmethod
    def _cho_solve(a, b):
        transpose_axis = list(range(a.ndim))
        transpose_axis[-2:] = transpose_axis[-2:][::-1]

        return be.solve_triangular(
            be.transpose(a, transpose_axis),
            be.solve_triangular(a, b, lower=True),
            lower=False,
        )

    def fit(self, data, *args, **kwargs):
        if kwargs.get("like_prec", None) is not None:
            kwargs["like_prec"] = kwargs["like_prec"][None]

        params = self.fit_batch(data[None], *args, **kwargs)
        return tuple(x[0] for x in params)

    def fit_batch(
        self,
        data,
        prior_mean=None,
        prior_prec=None,
        like_prec=None,
        standardize=True,
        overwrite_data: bool = False,
    ):
        data = be.maximum(data, 0, out=data if overwrite_data else None)
        norm = be.sum(data, axis=tuple(range(1, data.ndim)), keepdims=True)
        data = be.divide(data, be.where(norm < 1e-6, 1, norm), out=data)
        data = be.log(data + 1e-10, out=data)

        shape = data.shape[1:]
        ndim = len(shape)
        n_params = 1 + ndim + (ndim * (ndim + 1) // 2)

        if prior_mean is None:
            prior_mean = be.zeros(n_params)
        if prior_prec is None:
            prior_prec = be.zeros(n_params)

        if like_prec is None:
            like_prec = be.exp(data) - 1e-10

        like_prec = be.maximum(like_prec, 0, out=like_prec)

        phi = self._build_design_matrix(shape)[None]
        if standardize:
            phi, data, data_mean, means, stds = self._standardize(
                data, phi, like_prec=like_prec
            )
            prior_inter = be.add(
                prior_mean[0], be.sum(means * prior_mean[None, 1:], axis=(1))
            )
            prior_mean = prior_mean[..., 1:] * stds
            prior_prec = prior_prec[1:]

        N = be.size(data) // int(data.shape[0])
        like_prec = like_prec.reshape(data.shape[0], 1, -1)

        transpose_axis = (0, 2, 1)
        phi_tprec = be.transpose(phi, transpose_axis) * like_prec

        # b,k,n @ 1,n,k -> b,k,k
        gram = be.einsum("bin,tnj->bij", phi_tprec, phi)
        gram = be.multiply(gram, 1 / N, out=gram)
        gram = be.add(gram, be.diag(prior_prec)[None], out=gram)

        chol_gram = be.cholesky(gram, lower=True)
        b = be.add(
            (phi_tprec @ data.reshape(data.shape[0], -1, 1)) / N,
            (prior_prec * prior_mean).reshape(1, -1, 1),
        )
        w_opt = self._cho_solve(chol_gram, b).reshape(data.shape[0], -1)
        if standardize:
            # Intercept including prior
            w_opt = be.divide(w_opt, stds)
            weight_avg = be.mean(like_prec, axis=(1, 2))

            # n x 1
            w_inter = be.multiply(
                be.divide(1, weight_avg + prior_prec[0]),
                data_mean + prior_prec[0] * prior_inter,
            )
            w_inter = w_inter.reshape(-1)
            w_inter = be.subtract(w_inter, be.sum(means * w_opt, axis=(1)))
            w_opt = be.concatenate((w_inter[..., None], w_opt), axis=1)

            # Precision rescaling
            gram = be.multiply(stds[..., None], gram, out=gram)
            gram = be.multiply(gram, stds[None, ...], out=gram)

            # Add intercept covariance
            sigma_std = be.cholesky(gram, lower=True)
            sigma_rec = be.zeros((data.shape[0], n_params, n_params))

            sigma_mean = self._cho_solve(sigma_std, means)
            sigma_rec[:, 0, 0] = be.add(
                be.divide(1, weight_avg + prior_prec[0]), means[0].T @ sigma_mean[0]
            )
            sigma_rec[:, 1:, 1:] = self._cho_solve(
                sigma_std, be.eye(sigma_std.shape[1])[None]
            )
            sigma_rec[:, 0, 1:] = -sigma_mean

            sigma_rec = be.add(
                be.triu(sigma_rec, 0),
                be.transpose(be.triu(sigma_rec, 1), transpose_axis),
                out=sigma_rec,
            )
            gram = be.linalg.inv(sigma_rec)

        return w_opt, gram

    @staticmethod
    def score_batch(params1: Tuple, params2: Tuple, **kwargs) -> BackendArray:
        omega1, _ = params1
        omega2, _ = params2

        d = int((-3 + np.sqrt(1 + 8 * be.size(omega1))) / 2)
        _prec1, _prec2 = omega1[(d + 1) :], omega2[:, (d + 1) :]

        # n_kernels, d_parameter
        _prec1 = be.reshape(_prec1, (1, -1))

        def _norm(x, axis=-1):
            return be.sqrt(be.sum(be.square(x), axis=axis))

        err = be.sum(be.square(_prec1 - _prec2), axis=-1)
        norm = _norm(_prec1, axis=-1) * _norm(_prec2, axis=-1)
        return be.where(norm > 1e-8, err / norm, 1e10)

    @staticmethod
    def rotate(rotation_matrix: BackendArray, params) -> Tuple:
        omega, precision = params
        d = be.size(omega)
        ndim = int((-3 + np.sqrt(1 + 8 * d)) / 2)
        w0, mean, prec = omega[0], omega[1 : (ndim + 1)], omega[(ndim + 1) :]

        row_indices, col_indices = be.triu_indices(ndim)
        prec_matrix = be.zeros((ndim, ndim))
        prec_matrix[row_indices, col_indices] = prec
        prec_matrix[col_indices, row_indices] = prec

        prec_rotated = rotation_matrix @ prec_matrix @ rotation_matrix.T
        mean_clean = be.linalg.inv(prec_matrix) @ mean
        mean_clean = prec_rotated @ mean_clean

        prec_rotated = prec_rotated[row_indices, col_indices]
        omega = be.concatenate([be.to_backend_array([w0]), mean_clean, prec_rotated])
        return omega, precision

    def sample(self, params, shape):
        omega, precision = params
        phi = self._build_design_matrix(shape)
        return be.reshape(be.exp(phi @ omega), shape)


class GaussianNLS:

    @staticmethod
    def _pack_params(height: float, mean: NDArray, L: NDArray) -> NDArray:
        D = mean.size
        i, j = np.tril_indices(D)
        L_params = L[i, j]
        diag_mask = i == j
        L_params[diag_mask] = np.log(L_params[diag_mask])
        return np.concatenate(([height], mean.ravel(), L_params))

    @staticmethod
    def _unpack_params(p: NDArray, D: int):
        height = p[0]
        mean = p[1 : 1 + D]
        L_params = p[1 + D :]
        i, j = np.tril_indices(D)
        L = np.zeros((D, D))
        L[i, j] = L_params
        diag_mask = i == j

        # Exponentiate diagonal to ensure positive values
        L[i[diag_mask], j[diag_mask]] = np.exp(L[i[diag_mask], j[diag_mask]])
        return height, mean, L

    def _jacobian(
        self,
        p: NDArray,
        x: NDArray,
        y: NDArray,
        D: int,
        buffer_d: NDArray = None,
        buffer_q: NDArray = None,
        buffer_J: NDArray = None,
    ) -> NDArray:
        """
        Analytical Jacobian of the residuals with respect to parameters.

        Returns
        -------
        jacobian : np.ndarray
            (N, n_params) where n_params = 1 + D + D*(D+1)/2

        Notes
        -----
        Model definition
        f(x) = height * exp(-0.5 * (x - mean)^T @ prec @ (x - mean))

        Partial derivatives
        ∂f/∂height = exp(-0.5 * q)
        ∂f/∂mean_k = height * exp(-0.5 * q) * (prec @ d)_k
        ∂f/∂L_ij = δ_ki * L_lj + δ_li * L_kj
                 = -0.5 * height * exp(-0.5 * q) * 2 * d_i * (L.T @ d)_j

        where
        q = (x - mean)^T @ prec @ (x - mean)
        d = x - mean
        prec = L @ L.T

        """
        height, mean, L = self._unpack_params(p, D)
        prec = L @ L.T

        N = x.shape[0]
        n_L_params = D * (D + 1) // 2
        n_params = 1 + D + n_L_params

        d = np.subtract(x, mean[None, :], out=buffer_d)
        q = np.sum(d @ prec * d, axis=1, out=buffer_q)
        g = np.exp(-0.5 * q)

        if buffer_J is None:
            buffer_J = np.zeros((N, n_params))

        # ∂f/∂height
        buffer_J[:, 0] = g

        # ∂f/∂mean_k = height * g * (prec @ d)_k
        prec_d = prec @ d.T
        buffer_J[:, 1 : 1 + D] = height * g[:, None] * prec_d.T

        i_tril, j_tril = np.tril_indices(D)

        # ∂f/∂L_ij = -0.5 * height * exp(-0.5 * q) * 2 * d_i * (L.T @ d)_j
        L_T_d = L.T @ d.T
        for idx, (i, j) in enumerate(zip(i_tril, j_tril)):
            # Compute ∂q/∂L_ij = 2 * d_i * (L.T @ d)_j
            dq_dL = 2.0 * d[:, i] * L_T_d[j, :]

            if i == j:
                # ∂L_ii/∂log(L_ii) = L_ii
                dq_dL *= L[i, i]

            # ∂f/∂param = -0.5 * height * g * ∂q/∂param
            buffer_J[:, 1 + D + idx] = -0.5 * height * g * dq_dL
        return buffer_J

    def _residuals(
        self,
        p: NDArray,
        x: NDArray,
        y: NDArray,
        D: int,
        buffer_d: NDArray = None,
        buffer_q: NDArray = None,
    ) -> NDArray:
        height, mean, L = self._unpack_params(p, D)
        pred = self.model(x, height, mean, L @ L.T, buffer_d, buffer_q)
        return pred - y

    def fit(
        self,
        data: BackendArray,
        p0: Optional[Tuple[float, NDArray, NDArray]] = None,
        max_nfev: Optional[int] = None,
        method: str = "trf",
        **kwargs,
    ) -> Tuple[float, BackendArray, BackendArray]:
        from scipy.optimize import least_squares

        grid = _make_grid(data.shape)

        data = np.maximum(data, 0)
        data = np.divide(data, data.sum()).ravel()

        if grid.ndim != 2:
            raise ValueError("grid must be (N, D)")
        if data.ndim != 1 or data.shape[0] != data.shape[0]:
            raise ValueError("data must be (N,) and match x_vec rows")

        _, D = grid.shape

        if p0 is None:
            height0 = float(np.max(data)) if np.max(data) > 0 else 1.0
            w = data.clip(min=0)
            wsum = float(w.sum()) if w.sum() > 0 else 1.0
            mean0 = (grid * w[:, None]).sum(axis=0) / wsum
            xc = grid - mean0[None, :]
            cov0 = (xc * w[:, None]).T @ xc / wsum + 1e-6 * np.eye(D)
            prec0 = np.linalg.inv(cov0)
            L0 = np.linalg.cholesky(prec0)
            p_init = self._pack_params(height0, mean0, L0)
        else:
            height0, mean0, prec0 = p0
            L0 = np.linalg.cholesky(prec0)
            p_init = self._pack_params(
                float(height0), np.asarray(mean0, dtype=float), L0
            )

        n_params = 1 + D + D * (D + 1) // 2

        buffer_d = np.zeros_like(grid)
        buffer_J = np.zeros((grid.shape[0], n_params), dtype=grid.dtype)
        buffer_q = np.zeros_like(data)

        _residuals_buf = partial(self._residuals, buffer_d=buffer_d, buffer_q=buffer_q)

        _jacobian_buf = partial(
            self._jacobian,
            buffer_d=buffer_d,
            buffer_q=buffer_q,
            buffer_J=buffer_J,
        )

        res = least_squares(
            fun=_residuals_buf,
            # jac=_jacobian_buf,
            x0=p_init,
            args=(grid, data, D),
            method=method,
            max_nfev=max_nfev,
        )
        height, mean, L = self._unpack_params(res.x, D)
        return (height, mean, L @ L.T)

    @staticmethod
    def rotate(rotation_matrix: BackendArray, params) -> Tuple:
        prec = rotation_matrix @ (params[2] @ rotation_matrix.T)
        return (params[0], params[1], prec)

    @staticmethod
    def model(
        x: BackendArray,
        height: float,
        mean: BackendArray,
        prec: BackendArray,
        buffer_d: BackendArray = None,
        buffer_q: BackendArray = None,
    ) -> BackendArray:
        d = np.subtract(x, mean[None, :], out=buffer_d)
        q = np.sum(d @ prec * d, axis=1, out=buffer_q)
        q = np.multiply(q, -0.5, out=q)
        q = np.exp(q, out=q)
        return np.multiply(q, height, out=q)

    def sample(self, params, shape):
        grid = _make_grid(shape)
        return self.model(grid, *params).reshape(shape)


def get_natural_param(height, mean, sigma, grid_mean=None, grid_scale=None):
    """
    Map original gauss parameters, viz. height, mean, sigma
    onto the natural parameters of the log-fit
    fits: w0 * exp(-1/2 * (x - mean).T @ Sigma^{-1} @ (x - mean))
    """
    import scipy

    ndim = len(mean)
    if grid_mean is not None:
        mean -= grid_mean

    if grid_scale is not None:
        mean *= grid_scale
        sigma = sigma * (grid_scale[None].T @ grid_scale[None])

    prec = np.linalg.inv(sigma)

    # Precision adjusted mean
    cho_sigma = scipy.linalg.cho_factor(sigma, lower=True)
    prec_mean = scipy.linalg.cho_solve(cho_sigma, mean)

    # Constant
    w_const = np.log(height) - 0.5 * mean.T @ prec_mean

    # Upper triangular of precision
    # diagonals enter as -0.5 and off diagonal entries as -1
    # Design matrix then comes with x @ x.T
    row_idx, col_idx = np.triu_indices(ndim)
    w_conv = prec[row_idx, col_idx]

    return np.concatenate(([w_const], prec_mean, w_conv), axis=0)


def get_original_param(w, ndim, grid_mean=None, grid_scale=None):
    """
    Map natural paramters back onto the real parameters of the model,
    viz. w -> height, mean, cov (or precision)
    """
    log_height = w[0]
    prec_mean = w[1 : (ndim + 1)]
    prec_triug = w[(ndim + 1) :]

    # Extract covariance matrix
    # Diagonal entries put with -1/2 and off-digonal ones with -1
    prec = np.zeros((ndim, ndim))
    prec[np.triu_indices(ndim)] = prec_triug
    prec += prec.T

    scaling = np.ones_like(prec)
    np.fill_diagonal(scaling, val=0.5)
    prec *= scaling

    cov = np.linalg.inv(prec)
    mean = cov @ prec_mean
    height = np.exp(log_height + 0.5 * mean.T @ prec_mean)

    if grid_scale is not None:
        mean /= grid_scale
        cov /= grid_scale[None].T @ grid_scale[None]

    if grid_mean is not None:
        mean += grid_mean

    return height, mean, cov


class _KernelFit(PeakCaller):
    """
    Initialize the KernelFitting object. This class is not intended to be used
    directly but saves as basis for KernelFitting to support dynamic inheritance
    of different peak caller strategies.

    Parameters
    ----------
    kernel : Type[Any]
        Class type of the kernel to be used.
    kernel_params : tuple
        Parameters of the kernel.
    **kwargs:
        Keyword arguments passed to PeakCaller.__init__
    """

    def __init__(
        self,
        reference: BackendArray,
        metric: str,
        ft_mask: BackendArray = None,
        kernel: Type[AbstractKernel] = GaussianKernel,
        **kwargs,
    ):
        super().__init__(**kwargs)

        reference = be.to_backend_array(reference)
        if ft_mask is not None:
            ft_mask = be.to_backend_array(ft_mask)

        self.index = -1
        self.kernel = kernel(reference=reference, ft_mask=ft_mask)
        self.extraction_shape = be.to_backend_array(reference.shape)
        self.left_pad = be.astype(be.divide(self.extraction_shape, 2), int)

        self.metric = metric

    def call_peaks(
        self, scores: BackendArray, rotation_matrix: BackendArray, **kwargs
    ) -> Tuple[BackendArray, BackendArray]:
        peaks, _ = super().call_peaks(scores=scores, rotation_matrix=rotation_matrix)
        valid_peaks = self._get_peak_mask(peaks=peaks, scores=scores)
        if valid_peaks is None:
            return None, None

        peaks = be.astype(peaks, int)

        peaks = peaks[valid_peaks,]
        starts = be.subtract(peaks, self.left_pad)

        # Extract score subvolumes and wrap boundaries around array
        ret, (n, d), shape = [], starts.shape, self.extraction_shape
        for i in range(d):
            indices = starts[:, slice(i, i + 1)] + be.arange(shape[i])[None]
            indices = be.mod(indices, scores.shape[i], out=indices)
            indices_shape = (n, *tuple(1 if k != i else -1 for k in range(d)))
            ret.append(be.reshape(indices, indices_shape))
        data = scores[*ret]

        self.kernel.rotate_masked(rotation_matrix)

        # params_obs = []

        peak_details, batch_size = be.zeros((peaks.shape[0],), be._float_dtype), 250
        for i in range(0, data.shape[0], batch_size):
            end = min(i + batch_size, data.shape[0])
            if self.metric == "l2":
                params_observation = self.kernel.fit_batch(data[i:end])
                peak_details[i:end] = -self.kernel.score_batch(params_observation)
                # params_obs.append(params_observation[0])
            elif self.metric == "dist":
                peak_details[i:end] = -self.kernel.dist(data[i:end])
            elif self.metric == "score":
                peak_details[i:end] = scores[tuple(peaks[i:end].T)]
            elif self.metric == "dist_xd":
                peak_details[i:end] = -self.kernel.dist_xd(data[i:end])

        # params_obs = be.concatenate(params_obs)

        # import pickle

        # self.index += 1

        # with open(
        #     f"/scratch/vmaurer/temp/params/{self.index}.pickle", mode="wb"
        # ) as ofile:
        #     pickle.dump(
        #         (
        #             params_obs,
        #             self.kernel.params_natural,
        #             peaks,
        #             scores[tuple(peaks.T)],
        #             rotation_matrix,
        #         ),
        #         ofile,
        #     )

        return peaks, peak_details


class KernelFit(_KernelFit):
    # def __new__(cls, peak_caller, *args, **kwargs):
    def __new__(cls, *args, **kwargs):
        from .peaks import PeakCallerMaximumFilter

        peak_caller = PeakCallerMaximumFilter
        dynamic_class = type(
            f"KernelFitting_{peak_caller.__name__}", (_KernelFit, peak_caller), {}
        )
        return dynamic_class(**kwargs)
