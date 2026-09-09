"""
Implements class Wedge and WedgeReconstructed.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Literal

import numpy as np

from ..types import NDArray
from ..backends import backend as be
from .compose import ComposableFilter
from ..parser import XMLParser, StarParser, MDOCParser
from ._utils import (
    frequency_grid_at_angle,
    compute_tilt_shape,
    fftfreqn,
    shift_fourier,
)

__all__ = ["Wedge", "WedgeReconstructed"]


@dataclass
class Wedge(ComposableFilter):
    """
    Create per-tilt wedge mask for tomographic data.
    """

    #: Tilt angles in degrees.
    angles: Tuple[float] = None
    #: The weights corresponding to each tilt angle, default to 1.
    weights: Tuple[float] = None
    #: Whether tilts should be used or not, defaults to True.
    use_tilt: Tuple[bool] = None
    #: Axis the plane is tilted over, defaults to 0.
    tilt_axis: int = 0
    #: The projection axis, defaults to 2 (z).
    opening_axis: int = 2
    #: The type of weighting to apply, defaults to None.
    weight_type: Optional[Literal["angle", "relion", "grigorieff"]] = None
    #: The sampling rate, defaults to 1 Ångstrom / voxel.
    sampling_rate: Tuple[float] = 1

    @classmethod
    def from_file(cls, filename: str, **kwargs) -> "Wedge":
        """
        Generate a :py:class:`Wedge` instance by reading tilt angles and weights.
        Supported extensions are:

            +-------+---------------------------------------------------------+
            | .star | Tomostar STAR file                                      |
            +-------+---------------------------------------------------------+
            | .xml  | WARP/M XML file                                         |
            +-------+---------------------------------------------------------+
            | .mdoc | SerialEM file                                           |
            +-------+---------------------------------------------------------+
            | .*    | Tab-separated file with optional column names           |
            +-------+---------------------------------------------------------+

        Parameters
        ----------
        filename : str
            The path to the file containing tilt angles and weights.

        Returns
        -------
        :py:class:`Wedge`
           Class instance instance initialized with angles and weights from the file.
        """
        func = _from_text
        if filename.lower().endswith("xml"):
            func = _from_xml
        elif filename.lower().endswith("star"):
            func = _from_star
        elif filename.lower().endswith("mdoc"):
            func = _from_mdoc

        data = func(filename)
        angles, weights = data.get("angles", None), data.get("weights", None)
        if angles is None:
            raise ValueError(f"Could not find colum angles in {filename}")

        if weights is None:
            weights = (1,) * len(angles)

        if len(weights) != len(angles):
            raise ValueError("Length of weights and angles differ.")

        use_tilt = data.get("use_tilt", None)
        if use_tilt is None:
            use_tilt = (True,) * len(angles)

        return cls(
            tilt_axis=0,
            opening_axis=2,
            angles=np.array(angles, dtype=np.float32),
            weights=np.array(weights, dtype=np.float32),
            use_tilt=use_tilt,
            **kwargs,
        )

    def _evaluate(
        self, shape: Tuple[int, ...], weight_type: str = None, **kwargs: Dict
    ) -> NDArray:
        """Returns a Wedge stack of chosen parameters."""
        weight_types = {
            None: weight_uniform,
            "angle": weight_angle,
            "relion": weight_relion,
            "grigorieff": weight_grigorieff,
        }

        func = weight_types.get(weight_type, None)
        if func is None:
            raise ValueError(
                f"Supported weight_types are {','.join(list(weight_types.keys()))}"
            )
        ret = func(shape=shape, **kwargs)

        # Warp style tilt masking
        use_tilt = kwargs.get("use_tilt", None)
        if use_tilt is not None and len(use_tilt) == ret.shape[0]:
            scale = np.where(use_tilt, 1.0, 0.0001)
            ret = ret * np.expand_dims(scale, axis=tuple(range(1, ret.ndim)))

        ret = be.to_backend_array(ret, be._float)
        return {"data": ret, "shape": shape}


@dataclass
class WedgeReconstructed(Wedge):
    """
    Create wedge mask for tomographic reconstructions.
    """

    #: Tilt angles in degrees.
    angles: Tuple[float] = None
    #: Weights to assign to individual wedge components. Not considered for continuous wedge
    weights: Tuple[float] = None
    #: Whether individual wedge components should be weighted.
    use_tilt: Tuple[bool] = None
    #: Whether tilts should be used or not.
    weight_wedge: bool = False
    #: Whether to create a continous wedge or a per-component wedge.
    create_continuous_wedge: bool = False
    #: Frequency cutoff of filter
    frequency_cutoff: float = 0.5
    #: Axis the plane is tilted over, defaults to 0 (x).
    tilt_axis: int = 0
    #: The projection axis, defaults to 2 (z).
    opening_axis: int = 2

    def _evaluate(self, shape: Tuple[int, ...], **kwargs) -> Dict:
        """
        Generate a reconstructed wedge.

        Parameters
        ----------
        shape : tuple of int
            The shape to build the filter for.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            data: BackendArray
                The filter mask.
            shape: tuple of ints
                The requested filter shape
        """
        func = step_wedge
        angles = kwargs.pop("angles", (0,))
        if kwargs.get("create_continuous_wedge", False):
            func = continuous_wedge
            if len(angles) != 2:
                angles = (min(angles), max(angles))

        weights = kwargs.pop("weights", None)
        weight_wedge = kwargs.get("weight_wedge", False)
        if weight_wedge and weights is None:
            weights = np.cos(np.radians(be.to_numpy_array(angles)))

        if not weight_wedge:
            weights = None

        ret = func(shape=shape, angles=angles, weights=weights, **kwargs)

        # Move DC component to origin
        if func == continuous_wedge:
            ret = shift_fourier(ret, shape_is_real_fourier=False)
        else:
            # Warp style tilt masking
            use_tilt = kwargs.get("use_tilt", None)
            if use_tilt is not None and len(use_tilt) == ret.shape[0]:
                scale = np.where(use_tilt, 1.0, 0.0001)
                ret = ret * np.expand_dims(scale, axis=tuple(range(1, ret.ndim)))

        frequency_cutoff = kwargs.get("frequency_cutoff", None)
        if frequency_cutoff is not None:
            freq = fftfreqn(
                shape=shape,
                sampling_rate=1,
                compute_euclidean_norm=True,
                shape_is_real_fourier=False,
                fftshift=False,
            )
            ret = np.multiply(ret, freq <= frequency_cutoff, out=ret)

        if not weight_wedge:
            ret = (ret > 0) * 1.0
        ret = be.to_backend_array(ret, be._float)
        return {"data": ret, "shape": shape}


def continuous_wedge(
    shape: Tuple[int, ...],
    angles: Tuple[float, float],
    opening_axis: int,
    tilt_axis: int,
    **kwargs: Dict,
) -> NDArray:
    """
    Generate a continous wedge mask with DC component at the center.

    Parameters
    ----------
    shape : tuple of int
        The shape of the reconstruction volume.
    angles : tuple of float
        Start and stop tilt angle in degrees.
    opening_axis : int
        The axis around which the wedge is opened.
    tilt_axis : int
        The axis along which the tilt is applied.

    Returns
    -------
    NDArray
        Wedge mask.
    """
    angles = np.abs(np.asarray(angles))
    aspect_ratio = shape[opening_axis] / shape[tilt_axis]
    angles = np.degrees(np.arctan(np.tan(np.radians(angles)) * aspect_ratio))

    start_radians = np.tan(np.radians(90 - angles[0]))
    stop_radians = np.tan(np.radians(-1 * (90 - angles[1])))

    grid = fftfreqn(shape, sampling_rate=None, fftshift=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(
            grid[opening_axis] == 0,
            np.tan(np.radians(90)) + 1,
            grid[tilt_axis] / grid[opening_axis],
        )

    wedge = np.logical_or(start_radians <= ratios, stop_radians >= ratios)
    return wedge.astype(np.float32)


def step_wedge(
    shape: Tuple[int, ...],
    angles: Tuple[float, ...],
    opening_axis: int,
    tilt_axis: int,
    weights: Tuple[float, ...] = None,
    reconstruction_filter: str = None,
    reconstruction_method: str = "gridding",
    interpolation_order: int = 1,
    **kwargs: Dict,
) -> NDArray:
    """
    Generate a per-angle wedge shape with DC component at the origin.

    Parameters
    ----------
    shape : tuple of int
        The shape of the reconstruction volume.
    angles : tuple of float
        The tilt angles in degrees.
    opening_axis : int
        The axis around which the wedge is opened.
    tilt_axis : int
        The axis along which the tilt is applied.
    weights : tuple of float, optional
        Weights to assign to individual tilts. Defaults to 1.
    reconstruction_filter : str
        Filter window applied during reconstruction.
        See :py:meth:`create_reconstruction_filter` for available options.
    reconstruction_method : str
        Reconstruction method: "rotation" or "gridding".

    Returns
    -------
    NDArray
        Wedge mask.
    """
    from .reconstruction import ReconstructFromTilt

    n_tilts = len(angles)
    shape = tuple(int(x) for x in shape)
    if weights is None:
        weights = np.ones(n_tilts, dtype=np.float32)

    weights = np.asarray(weights, dtype=np.float32)
    weights = np.repeat(weights, n_tilts // weights.shape[0], axis=0)

    rot_axis = min(i for i in range(len(shape)) if i not in (tilt_axis, opening_axis))

    wedge_shape = tuple(1 if i == rot_axis else x for i, x in enumerate(shape))
    slice_data = np.ones((n_tilts, shape[tilt_axis]), dtype=np.float32)
    for i in range(n_tilts):
        slice_data[i] *= weights[i]

    slice_dims = tuple(x for i, x in enumerate(wedge_shape) if i != opening_axis)
    slice_data = slice_data.reshape((n_tilts, *slice_dims))

    rec = ReconstructFromTilt(
        angles=angles,
        opening_axis=opening_axis,
        tilt_axis=tilt_axis,
        reconstruction_filter=reconstruction_filter,
        method=reconstruction_method,
        interpolation_order=interpolation_order,
    )

    wedge = rec(
        data=be.to_backend_array(slice_data),
        shape=wedge_shape,
        multiply_interpweights=kwargs.get("multiply_interpweights", True),
    )["data"]
    # wedge = rec(data=be.to_backend_array(slice_data), shape=wedge_shape)["data"]
    wedge = be.to_numpy_array(wedge)
    tile_dimensions = tuple(shape[i] if i == rot_axis else 1 for i in range(len(shape)))
    return np.tile(wedge, tile_dimensions)


def weight_uniform(angles: Tuple[float, ...], *args, **kwargs) -> NDArray:
    """
    Generate uniform weighted wedges.
    """
    return weight_angle(angles=np.zeros_like(angles), *args, **kwargs)


def weight_angle(
    shape: Tuple[int, ...],
    angles: Tuple[float, ...],
    opening_axis: int,
    tilt_axis: int,
    **kwargs,
) -> NDArray:
    """
    Generate weighted wedges based on the cosine of the current angle.
    """
    tilt_shape = compute_tilt_shape(
        shape=shape, opening_axis=opening_axis, reduce_dim=True
    )
    wedges = np.zeros((len(angles), *tilt_shape))
    for index, angle in enumerate(angles):
        wedges[index] = np.cos(np.radians(angle))
    return wedges


def weight_relion(
    shape: Tuple[int, ...],
    angles: Tuple[float, ...],
    weights: Tuple[float, ...],
    opening_axis: int,
    tilt_axis: int,
    sampling_rate: float = 1.0,
    **kwargs,
) -> NDArray:
    """
    Generate weighted wedges based on the RELION 1.4 formalism, weighting each tilt
    by the cosine of its angle and a Gaussian lowpass of its exposure.

    Returns
    -------
    NDArray
        Weighted wedges.
    """
    tilt_shape = compute_tilt_shape(
        shape=shape, opening_axis=opening_axis, reduce_dim=True
    )
    wedges = np.zeros((len(angles), *tilt_shape))
    for index, angle in enumerate(angles):
        freq_grid = frequency_grid_at_angle(
            shape=shape,
            opening_axis=opening_axis,
            tilt_axis=tilt_axis,
            angle=angle,
            sampling_rate=sampling_rate,
            fftshift=False,
        )
        freq_grid = np.square(freq_grid, out=freq_grid)
        freq_grid = np.multiply(-weights[index], freq_grid, out=freq_grid)
        freq_grid = np.exp(freq_grid, out=freq_grid)
        wedges[index] = np.multiply(freq_grid, np.cos(np.radians(angle)))

    return wedges


def weight_grigorieff(
    shape: Tuple[int, ...],
    angles: Tuple[float, ...],
    weights: Tuple[float, ...],
    opening_axis: int,
    tilt_axis: int,
    amplitude: float = 0.245,
    power: float = -1.665,
    offset: float = 2.81,
    sampling_rate: float = 1.0,
    **kwargs,
) -> NDArray:
    """
    Generate weighted wedges based on the formalism introduced in [1]_.

    Returns
    -------
    NDArray
        Weighted wedges.

    References
    ----------
    .. [1]  Timothy Grant, Nikolaus Grigorieff (2015), eLife 4:e06980.
    """
    tilt_shape = compute_tilt_shape(
        shape=shape, opening_axis=opening_axis, reduce_dim=True
    )

    wedges = np.zeros((len(angles), *tilt_shape), dtype=be._float)
    for index, angle in enumerate(angles):
        freq_grid = frequency_grid_at_angle(
            shape=shape,
            opening_axis=opening_axis,
            tilt_axis=tilt_axis,
            angle=angle,
            sampling_rate=sampling_rate,
            fftshift=False,
        )

        with np.errstate(divide="ignore"):
            np.power(freq_grid, power, out=freq_grid)
            np.multiply(amplitude, freq_grid, out=freq_grid)
            np.add(freq_grid, offset, out=freq_grid)
            np.multiply(-2, freq_grid, out=freq_grid)
            np.divide(weights[index], freq_grid, out=freq_grid)
        wedges[index] = np.exp(freq_grid)

    return wedges


def _from_xml(filename: str, **kwargs) -> Dict:
    """
    Read tilt data from a WARP/M XML file.

    Parameters
    ----------
    filename : str
        The path to the text file.

    Returns
    -------
    Dict
        A dictionary with one key for each column.
    """
    data = XMLParser(filename)
    return {"angles": data["Angles"], "weights": data["Dose"]}


def _from_star(filename: str, **kwargs) -> Dict:
    """
    Read tilt data from a STAR file.

    Parameters
    ----------
    filename : str
        The path to the text file.

    Returns
    -------
    Dict
        A dictionary with one key for each column.
    """
    data = StarParser(filename, delimiter=None)
    if "data_stopgap_wedgelist" in data:
        angles = data["data_stopgap_wedgelist"]["_tilt_angle"]
        weights = data["data_stopgap_wedgelist"]["_exposure"]
    else:
        try:
            # Warp format
            angles = data["data_"]["_wrpAxisAngle"]
            weights = data["data_"]["_wrpDose"]
        except KeyError:
            # Relion format
            potential_keys = [x for x in data.keys() if x.startswith("data_")]
            if len(potential_keys) != 1:
                raise ValueError(
                    f"Expected one 'data_*' field, got {len(potential_keys)} {potential_keys}"
                )
            key = potential_keys[0]
            angles = data[key]["_rlnTomoNominalStageTiltAngle"]
            weights = data[key]["_rlnMicrographPreExposure"]
    return {"angles": angles, "weights": weights}


def _from_mdoc(filename: str, **kwargs) -> Dict:
    """
    Read tilt data from a SerialEM MDOC file.

    Parameters
    ----------
    filename : str
        The path to the text file.

    Returns
    -------
    Dict
        A dictionary with one key for each column.
    """
    data = MDOCParser(filename)
    cumulative_exposure = np.multiply(np.add(1, data["ZValue"]), data["ExposureDose"])
    return {"angles": data["TiltAngle"], "weights": cumulative_exposure}


def _from_text(filename: str, **kwargs) -> Dict:
    """
    Read column data from a text file.

    Parameters
    ----------
    filename : str
        The path to the text file.

    Returns
    -------
    Dict
        A dictionary with keys angles and weights if available.
    """
    header = None
    try:
        data = np.loadtxt(filename)
    except Exception:
        # Probably has header
        data = np.loadtxt(filename, skiprows=1)
        with open(filename, mode="r", encoding="utf-8") as infile:
            header = infile.readline().strip().split()

    if header is not None:
        angles, weights = None, None
        if "angles" in header:
            angles = data[:, header.index("angles")]
        if "weights" in header:
            weights = data[:, header.index("weights")]
        return {"angles": angles, "weights": weights}

    # Perhaps AreTomo TLT file (angle, index, exposure)
    if data.ndim == 2 and data.shape[1] == 3:
        order = np.argsort(data[:, 1])
        if np.allclose(data[:, 1][order], np.arange(data.shape[0]) + 1):
            angles = data[:, 0]
            cumulative_exposure = np.cumsum(data[:, 2][order])
            cumulative_exposure = cumulative_exposure[np.argsort(order)]
            return {"angles": angles, "weights": cumulative_exposure}

    if data.ndim == 2:
        data = data[:, 0]
    return {"angles": data}
