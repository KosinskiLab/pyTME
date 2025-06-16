"""
Utility functions for cupy backend.

The functions spline_filter, _prepad_for_spline_filter, _filter_input,
_get_coord_affine_batched and affine_transform are largely copied from
cupyx.scipy.ndimage which operates under the following license

Copyright (c) 2015 Preferred Infrastructure, Inc.
Copyright (c) 2015 Preferred Networks, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

I have since extended the functionality of the cupyx.scipy.ndimage functions
in question to support batched inputs.

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import numpy
import cupy

from cupy import _core
from cupyx.scipy.ndimage import (
    _util,
    _interp_kernels,
    _interpolation,
    spline_filter1d,
    _spline_prefilter_core,
    _spline_kernel_weights,
)

spline_weights_inline = _spline_kernel_weights.spline_weights_inline


math_constants_preamble = r"""
// workaround for HIP: line begins with #include
#include <cupy/math_constants.h>
"""


def _prepad_for_spline_filter(input, mode, cval, batched=True):
    """
    Prepad the input array for spline filtering.

    Parameters
    ----------
    input : CupyArray
        The input array to be padded.
    mode : str
        Determines how input points outside the boundaries are handled.
    cval : scalar
        Constant value to use for padding if mode is 'grid-constant'.
    batched : bool, optional
        Whether the input has a leading batch dimension, by default False.

    Returns
    -------
    padded : CupyArray
        The padded input array.
    npad : int or tuple of tuples
        The amount of padding applied.
    """
    if mode in ["nearest", "grid-constant"]:
        # empirical factor chosen by SciPy
        npad = tuple(
            (0, 0) if batched and i == 0 else (12, 12) for i in range(input.ndim)
        )
        if mode == "grid-constant":
            kwargs = dict(mode="constant", constant_values=cval)
        else:
            kwargs = dict(mode="edge")
        padded = cupy.pad(input, npad, **kwargs)
    else:
        npad = 0
        padded = input
    return padded, npad


def spline_filter(input, order=3, output=cupy.float64, mode="mirror", batched=True):
    """Multidimensional spline filter.

    Parameters
    ----------
    input : CupyArray
        The input array.
    order : int, optional
        The order of the spline interpolation, default is 3. Must be in the range 0-5.
    output : CupyArray or dtype, optional
        The array in which to place the output, or the dtype of the returned array.
    mode : str, optional
        Determines how input points outside the boundaries are handled.
    batched : bool, optional
        Whether the input has a leading batch dimension. Default is False.

    Returns
    -------
    CupyArray
        The result of prefiltering the input.

    See Also
    --------
    :obj:`scipy.ndimage.spline_filter1d`
    """
    if order < 2 or order > 5:
        raise RuntimeError("spline order not supported")

    x = input
    ibatch = int(batched)
    temp, data_dtype, output_dtype = _interpolation._get_spline_output(x, output)
    if order not in [0, 1] and input.ndim > 0:
        for axis in range(ibatch, x.ndim):
            spline_filter1d(x, order, axis, output=temp, mode=mode)
            x = temp
    if isinstance(output, cupy.ndarray):
        _core.elementwise_copy(temp, output)
    else:
        output = temp
    if output.dtype != output_dtype:
        output = output.astype(output_dtype)
    return output


def _filter_input(image, prefilter, mode, cval, order, batched=True):
    """
    Perform spline prefiltering on the input image if requested.

    Parameters
    ----------
    image : CupyArray
        The input image to be filtered.
    prefilter : bool
        Whether to apply prefiltering or not.
    mode : str
        The boundary mode to use. See `cupy.pad` for details.
    cval : scalar
        Value to fill past edges of input if mode is 'constant'.
    order : int
        The order of the spline interpolation. Must be in the range 0-5.
    batched : bool, optional
        Whether the input has a leading batch dimension. Default is False.

    Returns
    -------
    filtered_image : ndarray
        The filtered image as a contiguous array.
    npad : int
        The amount of padding applied at each edge of the array.
    """
    if not prefilter or order < 2:
        return (cupy.ascontiguousarray(image), 0)
    padded, npad = _prepad_for_spline_filter(image, mode, cval, batched=batched)
    float_dtype = cupy.promote_types(image.dtype, cupy.float32)
    filtered = spline_filter(
        padded, order, output=float_dtype, mode=mode, batched=batched
    )
    return cupy.ascontiguousarray(filtered), npad


def _get_coord_affine_batched(ndim, nprepad=0):
    """
    Compute target coordinate based on a homogeneous transformation matrix.

    Parameters
    ----------
    ndim : int
        Number of dimensions for the coordinate system.
    nprepad : int, optional
        Number of elements to prepad, by default 0.

    Returns
    -------
    list of str
        A list of string operations representing the coordinate computation.

    Notes
    -----
    This function assumes the following variables have been initialized on the device:

    - mat (array): Array containing the (ndim, ndim + 1) transform matrix.
    - in_coords (array): Coordinates of the input.

    For example, in 2D

        c_0 = in_coords[0]
        c_1 = mat[0] * in_coords[1] + mat[1] * in_coords[2] + mat[2]
        c_2 = mat[3] * in_coords[1] + mat[4] * in_coords[2] + mat[5]

    """
    ops = []
    pre = f" + (W){nprepad}" if nprepad > 0 else ""

    batched, ibatched = True, 1
    ncol = ndim + 1 - ibatched
    for j in range(ndim):
        if batched:
            ops.append(
                f"""
                W c_{j} = (W)in_coord[{j}];"""
            )
            batched = False
            continue
        ops.append(
            f"""
            W c_{j} = (W)0.0;"""
        )
        j_batch = j - ibatched
        for k in range(ibatched, ndim):
            ops.append(
                f"""
                c_{j} += mat[{ncol * j_batch + k - ibatched}] * (W)in_coord[{k}];"""
            )
        ops.append(
            f"""
            c_{j} += mat[{ncol * j_batch + ndim - ibatched}]{pre};"""
        )
    return ops


def _generate_interp_custom(
    coord_func,
    ndim,
    large_int,
    yshape,
    mode,
    cval,
    order,
    name="",
    integer_output=False,
    nprepad=0,
    omit_in_coord=False,
    batched=False,
):
    """
    Args:
        coord_func (function): generates code to do the coordinate
            transformation. See for example, `_get_coord_shift`.
        ndim (int): The number of dimensions.
        large_int (bool): If true use Py_ssize_t instead of int for indexing.
        yshape (tuple): Shape of the output array.
        mode (str): Signal extension mode to use at the array boundaries
        cval (float): constant value used when `mode == 'constant'`.
        name (str): base name for the interpolation kernel
        integer_output (bool): boolean indicating whether the output has an
            integer type.
        nprepad (int): integer indicating the amount of prepadding at the
            boundaries.

    Returns:
        operation (str): code body for the ElementwiseKernel
        name (str): name for the ElementwiseKernel
    """

    ops = []
    internal_dtype = "double" if integer_output else "Y"
    ops.append(f"{internal_dtype} out = 0.0;")

    if large_int:
        uint_t = "size_t"
        int_t = "ptrdiff_t"
    else:
        uint_t = "unsigned int"
        int_t = "int"

    # determine strides for x along each axis
    for j in range(ndim):
        ops.append(f"const {int_t} xsize_{j} = x.shape()[{j}];")
    ops.append(f"const {uint_t} sx_{ndim - 1} = 1;")
    for j in range(ndim - 1, 0, -1):
        ops.append(f"const {uint_t} sx_{j - 1} = sx_{j} * xsize_{j};")

    if not omit_in_coord:
        # create in_coords array to store the unraveled indices
        ops.append(_interp_kernels._unravel_loop_index(yshape, uint_t))

    # compute the transformed (target) coordinates, c_j
    ops = ops + coord_func(ndim, nprepad)

    if cval is numpy.nan:
        cval = "(Y)CUDART_NAN"
    elif cval == numpy.inf:
        cval = "(Y)CUDART_INF"
    elif cval == -numpy.inf:
        cval = "(Y)(-CUDART_INF)"
    else:
        cval = f"({internal_dtype}){cval}"

    if mode == "constant":
        # use cval if coordinate is outside the bounds of x
        _cond = " || ".join(
            [f"(c_{j} < 0) || (c_{j} > xsize_{j} - 1)" for j in range(ndim)]
        )
        ops.append(
            f"""
        if ({_cond})
        {{
            out = {cval};
        }}
        else
        {{"""
        )

    if order == 0:
        if mode == "wrap":
            ops.append("double dcoord;")  # mode 'wrap' requires this to work
        for j in range(ndim):
            # determine nearest neighbor
            if mode == "wrap":
                ops.append(
                    f"""
                dcoord = c_{j};"""
                )
            else:
                ops.append(
                    f"""
                {int_t} cf_{j} = ({int_t})floor((double)c_{j} + 0.5);"""
                )

            # handle boundary
            if mode != "constant":
                if mode == "wrap":
                    ixvar = "dcoord"
                    float_ix = True
                else:
                    ixvar = f"cf_{j}"
                    float_ix = False
                ops.append(
                    _util._generate_boundary_condition_ops(
                        mode, ixvar, f"xsize_{j}", int_t, float_ix
                    )
                )
                if mode == "wrap":
                    ops.append(
                        f"""
                {int_t} cf_{j} = ({int_t})floor(dcoord + 0.5);"""
                    )

            # sum over ic_j will give the raveled coordinate in the input
            ops.append(
                f"""
            {int_t} ic_{j} = cf_{j} * sx_{j};"""
            )
        _coord_idx = " + ".join([f"ic_{j}" for j in range(ndim)])
        if mode == "grid-constant":
            _cond = " || ".join([f"(ic_{j} < 0)" for j in range(ndim)])
            ops.append(
                f"""
            if ({_cond}) {{
                out = {cval};
            }} else {{
                out = ({internal_dtype})x[{_coord_idx}];
            }}"""
            )
        else:
            ops.append(
                f"""
            out = ({internal_dtype})x[{_coord_idx}];"""
            )

    elif order == 1:
        if batched:
            ops.append(
                """
                int ic_0 = (int) c_0 * sx_0;"""
            )
        for j in range(int(batched), ndim):
            # get coordinates for linear interpolation along axis j
            ops.append(
                f"""
            {int_t} cf_{j} = ({int_t})floor((double)c_{j});
            {int_t} cc_{j} = cf_{j} + 1;
            {int_t} n_{j} = (c_{j} == cf_{j}) ? 1 : 2;  // points needed
            """
            )

            if mode == "wrap":
                ops.append(
                    f"""
                double dcoordf = c_{j};
                double dcoordc = c_{j} + 1;"""
                )
            else:
                # handle boundaries for extension modes.
                ops.append(
                    f"""
                {int_t} cf_bounded_{j} = cf_{j};
                {int_t} cc_bounded_{j} = cc_{j};"""
                )

            if mode != "constant":
                if mode == "wrap":
                    ixvar = "dcoordf"
                    float_ix = True
                else:
                    ixvar = f"cf_bounded_{j}"
                    float_ix = False
                ops.append(
                    _util._generate_boundary_condition_ops(
                        mode, ixvar, f"xsize_{j}", int_t, float_ix
                    )
                )

                ixvar = "dcoordc" if mode == "wrap" else f"cc_bounded_{j}"
                ops.append(
                    _util._generate_boundary_condition_ops(
                        mode, ixvar, f"xsize_{j}", int_t, float_ix
                    )
                )
                if mode == "wrap":
                    ops.append(
                        f"""
                    {int_t} cf_bounded_{j} = ({int_t})floor(dcoordf);;
                    {int_t} cc_bounded_{j} = ({int_t})floor(dcoordf + 1);;
                    """
                    )

            ops.append(
                f"""
            for (int s_{j} = 0; s_{j} < n_{j}; s_{j}++)
                {{
                    W w_{j};
                    {int_t} ic_{j};
                    if (s_{j} == 0)
                    {{
                        w_{j} = (W)cc_{j} - c_{j};
                        ic_{j} = cf_bounded_{j} * sx_{j};
                    }} else
                    {{
                        w_{j} = c_{j} - (W)cf_{j};
                        ic_{j} = cc_bounded_{j} * sx_{j};
                    }}"""
            )
    elif order > 1:
        if mode == "grid-constant":
            spline_mode = "constant"
        elif mode == "nearest":
            spline_mode = "nearest"
        else:
            spline_mode = _spline_prefilter_core._get_spline_mode(mode)

        # wx, wy are temporary variables used during spline weight computation
        ops.append(
            f"""
            W wx, wy;
            {int_t} start;"""
        )

        if batched:
            ops.append(
                """
                int ic_0 = (int) c_0 * sx_0;"""
            )
        for j in range(int(batched), ndim):
            # determine weights along the current axis
            ops.append(
                f"""
            W weights_{j}[{order + 1}];"""
            )
            ops.append(spline_weights_inline[order].format(j=j, order=order))

            # get starting coordinate for spline interpolation along axis j
            if mode in ["wrap"]:
                ops.append(f"double dcoord = c_{j};")
                coord_var = "dcoord"
                ops.append(
                    _util._generate_boundary_condition_ops(
                        mode, coord_var, f"xsize_{j}", int_t, True
                    )
                )
            else:
                coord_var = f"(double)c_{j}"

            if order & 1:
                op_str = """
                start = ({int_t})floor({coord_var}) - {order_2};"""
            else:
                op_str = """
                start = ({int_t})floor({coord_var} + 0.5) - {order_2};"""
            ops.append(
                op_str.format(int_t=int_t, coord_var=coord_var, order_2=order // 2)
            )

            # set of coordinate values within spline footprint along axis j
            ops.append(f"""{int_t} ci_{j}[{order + 1}];""")
            for k in range(order + 1):
                ixvar = f"ci_{j}[{k}]"
                ops.append(
                    f"""
                {ixvar} = start + {k};"""
                )
                ops.append(
                    _util._generate_boundary_condition_ops(
                        spline_mode, ixvar, f"xsize_{j}", int_t
                    )
                )

            # loop over the order + 1 values in the spline filter
            ops.append(
                f"""
            W w_{j};
            {int_t} ic_{j};
            for (int k_{j} = 0; k_{j} <= {order}; k_{j}++)
                {{
                    w_{j} = weights_{j}[k_{j}];
                    ic_{j} = ci_{j}[k_{j}] * sx_{j};
            """
            )

    if order > 0:
        _weight = " * ".join([f"w_{j}" for j in range(int(batched), ndim)])
        _coord_idx = " + ".join([f"ic_{j}" for j in range(ndim)])
        if mode == "grid-constant" or (order > 1 and mode == "constant"):
            _cond = " || ".join([f"(ic_{j} < 0)" for j in range(ndim)])
            ops.append(
                f"""
            if ({_cond}) {{
                out += {cval} * ({internal_dtype})({_weight});
            }} else {{
                {internal_dtype} val = ({internal_dtype})x[{_coord_idx}];
                out += val * ({internal_dtype})({_weight});
            }}"""
            )
        else:
            ops.append(
                f"""
            {internal_dtype} val = ({internal_dtype})x[{_coord_idx}];
            out += val * ({internal_dtype})({_weight});"""
            )

        ops.append("}" * (ndim - int(batched)))

    if mode == "constant":
        ops.append("}")

    if integer_output:
        ops.append("y = (Y)rint((double)out);")
    else:
        ops.append("y = (Y)out;")
    operation = "\n".join(ops)

    mode_str = mode.replace("-", "_")  # avoid hyphen in kernel name
    name = "cupyx_scipy_ndimage_interpolate_{}_order{}_{}_{}d_y{}".format(
        name,
        order,
        mode_str,
        ndim,
        "_".join([f"{j}" for j in yshape]),
    )
    if uint_t == "size_t":
        name += "_i64"
    return operation, name


@cupy._util.memoize(for_each_device=True)
def _get_batched_affine_kernel(
    ndim, large_int, yshape, mode, cval=0.0, order=1, integer_output=False, nprepad=0
):
    in_params = "raw X x, raw W mat"
    out_params = "Y y"
    operation, name = _generate_interp_custom(
        coord_func=_get_coord_affine_batched,
        ndim=ndim,
        large_int=large_int,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="affine_batched",
        integer_output=integer_output,
        nprepad=nprepad,
        batched=True,
    )
    return cupy.ElementwiseKernel(
        in_params,
        out_params,
        operation,
        name,
        preamble=math_constants_preamble,
    )


def affine_transform_batch(
    input,
    matrix,
    offset=0.0,
    output_shape=None,
    output=None,
    order=3,
    mode="constant",
    cval=0.0,
    prefilter=True,
    *,
    batched=True,
):
    """
    Apply an affine transformation.

    Parameters
    ----------
    input : CupyArray
        The input array.
    matrix : CupyArray
        The inverse coordinate transformation matrix, mapping output coordinates
        to input coordinates. The shape of the matrix depends on the dimensions
        of the input:

        - ``(ndim, ndim)``: linear transformation matrix for each output coordinate.
        - ``(ndim,)``: assume a diagonal 2D transformation matrix.
        - ``(ndim + 1, ndim + 1)``: assume homogeneous coordinates (ignores `offset`).
        - ``(ndim, ndim + 1)``: as above, but omits the bottom row
          ``[0, 0, ..., 1]``.

    offset : float or sequence, optional
        The offset into the array where the transform is applied. If a float,
        `offset` is the same for each axis. If a sequence, `offset` should
        contain one value for each axis. Default is 0.0.
    output_shape : tuple of ints, optional
        Shape tuple of the output.
    output : CupyArray or dtype, optional
        The array in which to place the output, or the dtype of the returned array.
    order : int, optional
        The order of the spline interpolation. Must be in the range 0-5.
        Default is 3.
    mode : str, optional
        Determines how input points outside the boundaries are handled.
        Default is 'constant'. Available options are

        +---------------+----------------------------------------------------+
        | 'constant'    | Fill with a constant value                         |
        +---------------+----------------------------------------------------+
        | 'nearest'     | Use the nearest pixel's value                      |
        +---------------+----------------------------------------------------+
        | 'mirror'      | Mirror the pixels at the boundary                  |
        +---------------+----------------------------------------------------+
        | 'reflect'     | Reflect the pixels at the boundary                 |
        +---------------+----------------------------------------------------+
        | 'wrap'        | Wrap the pixels at the boundary                    |
        +---------------+----------------------------------------------------+
        | 'grid-mirror' | Mirror the grid at the boundary                    |
        +---------------+----------------------------------------------------+
        | 'grid-wrap'   | Wrap the grid at the boundary                      |
        +---------------+----------------------------------------------------+
        | 'grid-        | Use a constant value for grid points outside the   |
        | constant'     | boundary                                           |
        +---------------+----------------------------------------------------+
        | 'opencv'      | OpenCV border mode                                 |
        +---------------+----------------------------------------------------+
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'`` or ``mode='opencv'``. Default is 0.0.
    prefilter : bool, optional
        Whether to prefilter the input array with `spline_filter` before
        interpolation. Default is True.
    batched : bool, optional
        Whether the input has a leading batch dimension. Default is False.

    Returns
    -------
    CupyArray or None
        The transformed input. If `output` is given as a parameter,
        None is returned.

    Notes
    -----
    When `prefilter` is True and `order > 1`, a temporary `float64` array
    of filtered values is created. If `prefilter` is False and `order > 1`,
    the output may be slightly blurred unless the input is prefiltered.

    When `batched` is True, the function treats the first dimension of the
    input as a batch dimension.
    """
    _interpolation._check_parameter("affine_transform", order, mode)

    offset = _util._fix_sequence_arg(offset, input.ndim, "offset", float)

    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise RuntimeError("no proper affine matrix provided")

    if matrix.shape[0] == matrix.shape[1] - 1:
        offset = matrix[:, -1]
        matrix = matrix[:, :-1]
    elif matrix.shape[0] == input.ndim + 1 - int(batched):
        offset = matrix[:-1, -1]
        matrix = matrix[:-1, :-1]

    if output_shape is None:
        output_shape = input.shape

    matrix = matrix.astype(cupy.float64, copy=False)
    ndim = input.ndim
    output = _util._get_output(output, input, shape=output_shape)
    if input.dtype.kind in "iu":
        input = input.astype(cupy.float32)
    filtered, nprepad = _filter_input(input, prefilter, mode, cval, order, batched)

    integer_output = output.dtype.kind in "iu"
    _util._check_cval(mode, cval, integer_output)
    large_int = (
        max(_core.internal.prod(input.shape), _core.internal.prod(output_shape))
        > 1 << 31
    )

    kernel = _interp_kernels._get_affine_kernel
    if batched:
        kernel = _get_batched_affine_kernel
    kern = kernel(
        ndim,
        large_int,
        output_shape,
        mode,
        cval=cval,
        order=order,
        integer_output=integer_output,
        nprepad=nprepad,
    )
    m = cupy.zeros((ndim - int(batched), ndim + 1 - int(batched)), dtype=cupy.float64)
    m[:, :-1] = matrix
    m[:, -1] = cupy.asarray(offset, dtype=cupy.float64)
    kern(filtered, m, output)
    return output
