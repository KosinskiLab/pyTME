"""
Strategy pattern to allow for flexible array / FFT backends.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from abc import ABC, abstractmethod
from multiprocessing import shared_memory
from typing import Tuple, Callable, List, Any, Union, Optional, Generator

from ..types import BackendArray, NDArray, Scalar, shm_type


def _create_metafunction(func_name: str) -> Callable:
    """
    Returns a wrapper of ``self._array_backend.func_name``.
    """

    def metafunction(self, *args, **kwargs) -> Any:
        backend_func = getattr(self._array_backend, func_name)
        return backend_func(*args, **kwargs)

    return metafunction


class MatchingBackend(ABC):
    """
    A strategy class for template matching backends.

    This class provides an interface to enable users to swap between different
    array and fft implementations while preserving the remaining functionalities
    of the API. The objective is to maintain a numpy like interface that generalizes
    across various backends.

    By delegating attribute access to the provided backend, users can access
    functions/methods of the backend as if they were directly part of this class.

    Parameters
    ----------
    array_backend : object
        The backend object providing array functionalities.
    float_dtype : type
        Data type of float array instances, e.g. np.float32.
    complex_dtype : type
        Data type of complex array instances, e.g. np.complex64.
    int_dtype : type
        Data type of integer array instances, e.g. np.int32.
    overflow_safe_dtype : type
        Data type than can be used in reduction operations to avoid overflows.

    Attributes
    ----------
    _array_backend : object
        The backend object providing array functionalities.
    _float_dtype : type
        Data type of float array instances, e.g. np.float32.
    _complex_dtype : type
        Data type of complex array instances, e.g. np.complex64.
    _int_dtype : type
        Data type of integer array instances, e.g. np.int32.
    _overflow_safe_dtype : type
        Data type than can be used in reduction operations to avoid overflows.
    _fundamental_dtypes : Dict
        Maps int, float and cmoplex python types to backend specific data types.

    Examples
    --------
    >>> import numpy as np
    >>> from tme.backends import NumpyFFTWBackend
    >>> backend = NumpyFFTWBackend(
    >>>     array_backend=np,
    >>>     float_dtype=np.float32,
    >>>     complex_dtype=np.complex64,
    >>>     int_dtype=np.int32
    >>> )
    >>> arr = backend.array([1, 2, 3])
    >>> print(arr)
    [1 2 3]

    Notes
    -----
    Developers should be aware of potential naming conflicts between methods and
    attributes of this class and those of the provided backend.
    """

    def __init__(
        self,
        array_backend,
        float_dtype: type,
        complex_dtype: type,
        int_dtype: type,
        overflow_safe_dtype: type,
    ):
        self._array_backend = array_backend
        self._float_dtype = float_dtype
        self._complex_dtype = complex_dtype
        self._int_dtype = int_dtype
        self._overflow_safe_dtype = overflow_safe_dtype

        self._fundamental_dtypes = {
            int: self._int_dtype,
            float: self._float_dtype,
            complex: self._complex_dtype,
        }

    def __getattr__(self, name: str):
        """
        Delegate attribute access to the underlying backend.

        Parameters
        ----------
        name : str
            The name of the attribute to access.

        Returns
        -------
        attribute
            The attribute from the underlying backend.

        Raises
        ------
        AttributeError
            If the attribute is not found in the backend.
        """
        return getattr(self._array_backend, name)

    def __dir__(self) -> List:
        """
        Return a list of attributes available in this object,
        including those from the backend.

        Returns
        -------
        list
            Sorted list of attributes.
        """
        base_attributes = []
        base_attributes.extend(dir(self.__class__))
        base_attributes.extend(self.__dict__.keys())
        base_attributes.extend(dir(self._array_backend))
        return sorted(base_attributes)

    @abstractmethod
    def to_backend_array(self, arr: NDArray) -> BackendArray:
        """
        Convert a numpy array instance to backend array type.

        Parameters
        ----------
        arr : NDArray
            The numpy array instance to be converted.

        Returns
        -------
        BackendArray
            An array of the specified backend.

        See Also
        --------
        :py:meth:`MatchingBackend.to_cpu_array`
        :py:meth:`MatchingBackend.to_numpy_array`
        """

    @abstractmethod
    def to_numpy_array(self, arr: BackendArray) -> NDArray:
        """
        Convert an array of given backend to a numpy array.

        Parameters
        ----------
        arr : BackendArray
            The array instance to be converted.

        Returns
        -------
        NDArray
            The numpy array equivalent of arr.

        See Also
        --------
        :py:meth:`MatchingBackend.to_cpu_array`
        :py:meth:`MatchingBackend.to_backend_array`
        """

    @abstractmethod
    def to_cpu_array(self, arr: BackendArray) -> BackendArray:
        """
        Convert an array of a given backend to a CPU array of that backend.

        Parameters
        ----------
        arr : BackendArray
            The array instance to be converted.

        Returns
        -------
        BackendArray
            The CPU array equivalent of arr.

        See Also
        --------
        :py:meth:`MatchingBackend.to_numpy_array`
        :py:meth:`MatchingBackend.to_backend_array`
        """

    def get_fundamental_dtype(self, arr: BackendArray) -> type:
        """
        Given an array instance, returns the corresponding fundamental python type,
        i.e., int, float or complex.

        Parameters
        ----------
        arr : BackendArray
            Input data.

        Returns
        -------
        type
            Data type.
        """

    @abstractmethod
    def free_cache(self):
        """
        Free cached objects allocated by backend.
        """

    @abstractmethod
    def add(
        self, arr1: BackendArray, arr2: BackendArray, out: BackendArray = None
    ) -> BackendArray:
        """
        Element-wise addition of arrays.

        Parameters
        ----------
        arr1 : BackendArray
            Input array.
        arr2 : BackendArray
            Input array.
        out : BackendArray, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        BackendArray
            Element-wise sum of the input arrays.
        """

    @abstractmethod
    def subtract(
        self, arr1: BackendArray, arr2: BackendArray, out: BackendArray = None
    ) -> BackendArray:
        """
        Element-wise subtraction of arrays.

        Parameters
        ----------
        arr1 : BackendArray
            The minuend array.
        arr2 : BackendArray
            The subtrahend array.
        out : BackendArray, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        BackendArray
            Element-wise difference of the input arrays.
        """

    @abstractmethod
    def multiply(
        self, arr1: BackendArray, arr2: BackendArray, out: BackendArray = None
    ) -> BackendArray:
        """
        Element-wise multiplication of arrays.

        Parameters
        ----------
        arr1 : BackendArray
            Input array.
        arr2 : BackendArray
            Input array.
        out : BackendArray, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        BackendArray
            Element-wise product of the input arrays.
        """

    @abstractmethod
    def divide(
        self, arr1: BackendArray, arr2: BackendArray, out: BackendArray = None
    ) -> BackendArray:
        """
        Element-wise division of arrays.

        Parameters
        ----------
        arr1 : BackendArray
            The dividend array.
        arr2 : BackendArray
            The divisor array.
        out : BackendArray, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        BackendArray
            Element-wise quotient of the input arrays.
        """

    @abstractmethod
    def mod(
        self, arr1: BackendArray, arr2: BackendArray, out: BackendArray = None
    ) -> BackendArray:
        """
        Element-wise modulus of arrays.

        Parameters
        ----------
        arr1 : BackendArray
            The dividend array.
        arr2 : BackendArray
            The divisor array.
        out : BackendArray, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        BackendArray
            Element-wise modulus of the input arrays.
        """

    @abstractmethod
    def einsum(
        self, arr1: BackendArray, arr2: BackendArray, out: BackendArray = None
    ) -> BackendArray:
        """
        Compute the einstein notation based summation.

        Parameters
        ----------
        subscripts : str
            Specifies the subscripts for summation (see  :obj:`numpy.einsum`).
        arr1, arr2 : BackendArray
            Input data.
        out : BackendArray, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        BackendArray
            Einsum of input arrays.
        """

    @abstractmethod
    def sum(
        self, arr: BackendArray, axis: Tuple[int] = None
    ) -> Union[BackendArray, Scalar]:
        """
        Compute the sum of array elements.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        axis : int or tuple of ints, optional
            Axis or axes to perform the operation on. Default is all.

        Returns
        -------
        Union[BackendArray, Scalar]
            Sum of ``arr``.
        """

    @abstractmethod
    def mean(
        self, arr: BackendArray, axis: Tuple[int] = None
    ) -> Union[BackendArray, Scalar]:
        """
        Compute the mean of array elements.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        axis : int or tuple of ints, optional
            Axis or axes to perform the operation on. Default is all.

        Returns
        -------
        Union[BackendArray, Scalar]
            Mean of ``arr``.
        """

    @abstractmethod
    def std(
        self, arr: BackendArray, axis: Tuple[int] = None
    ) -> Union[BackendArray, Scalar]:
        """
        Compute the standad deviation of array elements.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        axis : int or tuple of ints, optional
            Axis or axes to perform the operation on. Default is all.

        Returns
        -------
        Union[BackendArray, Scalar]
            Standard deviation of ``arr``.
        """

    @abstractmethod
    def max(
        self, arr: BackendArray, axis: Tuple[int] = None
    ) -> Union[BackendArray, Scalar]:
        """
        Compute the maximum of array elements.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        axis : int or tuple of ints, optional
            Axis or axes to perform the operation on. Default is all.

        Returns
        -------
        Union[BackendArray, Scalar]
            Maximum of ``arr``.
        """

    @abstractmethod
    def min(
        self, arr: BackendArray, axis: Tuple[int] = None
    ) -> Union[BackendArray, Scalar]:
        """
        Compute the minimum of array elements.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        axis : int or tuple of ints, optional
            Axis or axes to perform the operation on. Default is all.

        Returns
        -------
        Union[BackendArray, Scalar]
            Minimum of ``arr``.
        """

    @abstractmethod
    def maximum(
        self, arr1: BackendArray, arr2: BackendArray, out: BackendArray = None
    ) -> BackendArray:
        """
        Compute the element wise maximum of arr1 and arr2.

        Parameters
        ----------
        arr1, arr2 : BackendArray
            Input data.
        out : BackendArray, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        BackendArray
            Element wise maximum of ``arr1`` and ``arr2``.
        """

    @abstractmethod
    def minimum(
        self, arr1: BackendArray, arr2: BackendArray, out: BackendArray = None
    ) -> BackendArray:
        """
        Compute the element wise minimum of arr1 and arr2.

        Parameters
        ----------
        arr1, arr2 : BackendArray
            Input data.
        out : BackendArray, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        BackendArray
            Element wise minimum of arr1 and arr2.
        """

    @abstractmethod
    def sqrt(self, arr: BackendArray, out: BackendArray = None) -> BackendArray:
        """
        Compute the square root of array elements.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        out : BackendArray, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        BackendArray
            Square root of ``arr``.
        """

    @abstractmethod
    def square(self, arr: BackendArray, out: BackendArray = None) -> BackendArray:
        """
        Compute the square of array elements.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        out : BackendArray, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        BackendArray
            Square of ``arr``.
        """

    @abstractmethod
    def abs(self, arr: BackendArray, out: BackendArray = None) -> BackendArray:
        """
        Compute the absolute of array elements.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        out : BackendArray, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        BackendArray
            Absolute value of ``arr``.
        """

    @abstractmethod
    def transpose(self, arr: BackendArray) -> BackendArray:
        """
        Compute the transpose of arr.

        Parameters
        ----------
        arr : BackendArray
            Input data.

        Returns
        -------
        BackendArray
            Transpose of ``arr``.
        """

    def power(
        self,
        arr: BackendArray = None,
        power: BackendArray = None,
        out: BackendArray = None,
        *args,
        **kwargs,
    ) -> BackendArray:
        """
        Compute the n-th power of an array.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        power : BackendArray
            Power to raise ``arr`` to.
        arr : BackendArray
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        BackendArray
            N-th power of ``arr``.
        """

    def tobytes(self, arr: BackendArray) -> str:
        """
        Compute the bytestring representation of arr.

        Parameters
        ----------
        arr : BackendArray
            Input data.

        Returns
        -------
        str
            Bytestring representation of ``arr``.
        """

    @abstractmethod
    def size(self, arr: BackendArray) -> int:
        """
        Compute the number of elements of arr.

        Parameters
        ----------
        arr : BackendArray
            Input data.

        Returns
        -------
        int
            Number of elements in ``arr``.
        """

    @abstractmethod
    def fill(self, arr: BackendArray, value: Scalar) -> None:
        """
        Fills ``arr`` in-place with a given value.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        value : Scalar
            The value to fill the array with.
        """

    @abstractmethod
    def zeros(self, shape: Tuple[int], dtype: type) -> BackendArray:
        """
        Returns an aligned array of zeros with specified shape and dtype.

        Parameters
        ----------
        shape : tuple of ints.
            Desired shape for the array.
        dtype : type
            Desired data type for the array.

        Returns
        -------
        BackendArray
            Byte-aligned array of zeros with specified shape and dtype.
        """

    @abstractmethod
    def full(self, shape: Tuple[int], dtype: type, fill_value: Scalar) -> BackendArray:
        """
        Returns an array filled with fill_value of specified shape and dtype.

        Parameters
        ----------
        shape : tuple of ints.
            Desired shape for the array.
        dtype : type
            Desired data type for the array.

        Returns
        -------
        BackendArray
            Byte-aligned array of zeros with specified shape and dtype.
        """

    @abstractmethod
    def eps(self, dtype: type) -> Scalar:
        """
        Returns the minimal difference representable by dtype.

        Parameters
        ----------
        dtype : type
            Data type for which eps should be returned.

        Returns
        -------
        Scalar
            The eps for the given data type.
        """

    @abstractmethod
    def datatype_bytes(self, dtype: type) -> int:
        """
        Return the number of bytes occupied by a given datatype.

        Parameters
        ----------
        dtype : type
            Data type to determine the bytesize of.

        Returns
        -------
        int
            Number of bytes occupied by the datatype.
        """

    @abstractmethod
    def clip(
        self, arr: BackendArray, a_min: Scalar, a_max: Scalar, out: BackendArray = None
    ) -> BackendArray:
        """
        Clip elements of arr.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        a_min : Scalar
            Lower bound.
        a_max : Scalar
            Upper bound.
        out : BackendArray, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        BackendArray
            Clipped ``arr``.
        """

    @abstractmethod
    def astype(arr: BackendArray, dtype: type) -> BackendArray:
        """
        Change the datatype of arr.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        dtype : type
            Target data type.

        Returns
        -------
        BackendArray
            Freshly allocated array containing the data of ``arr`` in ``dtype``.
        """

    @abstractmethod
    def at(arr, idx, value) -> NDArray:
        """
        Assign value to arr at idx, for compatibility with immutable array structures
        such as jax.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        idx : BackendArray
            Indices to change.
        value: BackendArray
            Values to assign to arr at idx.

        Returns
        -------
        BackendArray
            Modified input data.
        """

    @abstractmethod
    def arange(
        self, stop: Scalar, start: Scalar = 0, step: Scalar = 1, *args, **kwargs
    ) -> BackendArray:
        """
        Arange values in evenly spaced interval.

        Parameters
        ----------
        stop : Scalar
            End of the interval.
        start : Scalar
            Start of the interval, zero by default.
        step : Scalar
            Interval step size, one by default.

        Returns
        -------
        BackendArray
            Array of evenly spaced values in specified interval.
        """

    def stack(self, *args, **kwargs) -> BackendArray:
        """
        Join a sequence of objects along a new axis.

        Parameters
        ----------
        arr : BackendArray
            Sequence of arrays.
        axis : int, optional
            Axis along which to stack the input arrays.

        Returns
        -------
        BackendArray
            Stacked input data.
        """

    @abstractmethod
    def concatenate(self, *args, **kwargs) -> BackendArray:
        """
        Join a sequence of objects along an existing axis.

        Parameters
        ----------
        arr : BackendArray
            Sequence of arrays.
        axis : int
            Axis along which to stack the input arrays.

        Returns
        -------
        BackendArray
            Concatenated input data.
        """

    @abstractmethod
    def repeat(self, *args, **kwargs) -> BackendArray:
        """
        Repeat each array element a specified number of times.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        repeats : int or tuple of ints
            Number of each repetitions along axis.

        Returns
        -------
        BackendArray
            Repeated ``arr``.
        """

    @abstractmethod
    def topk_indices(self, arr: NDArray, k: int) -> BackendArray:
        """
        Determinces the indices of largest elements.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        k : int
            Number of maxima to determine.

        Returns
        -------
        BackendArray
            Indices of ``k`` largest elements in ``arr``.
        """

    @abstractmethod
    def ssum(self, arr, *args, **kwargs) -> BackendArray:
        """
        Compute the sum of squares of ``arr``.

        Returns
        -------
        BackendArray
            Sum of squares with shape ().
        """

    def indices(self, *args, **kwargs) -> BackendArray:
        """
        Creates an array representing the index grid of an input.

        Returns
        -------
        BackendArray
            The index grid.
        """

    @abstractmethod
    def roll(self, *args, **kwargs) -> BackendArray:
        """
        Roll array elements along a specified axis.

        Parameters
        ----------
        a : BackendArray
            Input data.
        shift : int or tuple of ints, optional
            Shift along each axis.

        Returns
        -------
        BackendArray
            Array with elements rolled.
        """

    @abstractmethod
    def where(condition, *args) -> BackendArray:
        """
        Return elements from input depending on ``condition``.

        Parameters
        ----------
        condition : BackendArray
            Binary condition array.
        *args : BackendArray
            Values to choose from.

        Returns
        -------
        BackendArray
            Array of elements according to ``condition``.
        """

    @abstractmethod
    def unique(
        self,
        arr: BackendArray,
        return_index: bool = False,
        return_inverse: bool = False,
        return_counts: bool = False,
        axis: Tuple[int] = None,
        *args,
        **kwargs,
    ) -> Tuple[BackendArray]:
        """
        Find the unique elements of an array.

        Parameters
        ----------
        arr : BackendArray
            Input data.
        return_index : bool, optional
            Return indices that resulted in unique array, False by default.
        return_inverse : bool, optional
            Return indices to reconstruct the input, False by default.
        return_counts : bool, optional
            Return number of occurences of each unique element, False by default.
        axis : int or tuple of ints, optional
            Axis or axes to perform the operation on. Default is all.

        Returns
        -------
        BackendArray or tuple of BackendArray
            If `return_index`, `return_inverse`, and `return_counts` keyword
            arguments are all False (the default), this will be an BackendArray object
            of the sorted unique values. Otherwise, it's a tuple with one
            or more arrays as specified by those keyword arguments.
        """

    @abstractmethod
    def argsort(self, *args, **kwargs) -> BackendArray:
        """
        Compute the indices to sort a given input array.

        Parameters
        ----------
        arr : BackendArray
            Input array.
        dtype : type
            Target data type.

        Returns
        -------
        BackendArray
            Indices that would sort the input data.
        """

    @abstractmethod
    def unravel_index(
        self, indices: BackendArray, shape: Tuple[int]
    ) -> Tuple[BackendArray]:
        """
        Convert flat index to array indices.

        Parameters
        ----------
        indices : BackendArray
            Input data.
        shape : tuple of ints
            Shape of the array used for unraveling.

        Returns
        -------
        BackendArray
            Array indices.
        """

    @abstractmethod
    def tril_indices(self, *args, **kwargs) -> BackendArray:
        """
        Compute indices of upper triangular matrix

        Parameters
        ----------
        arr : BackendArray
            Input array.
        dtype : type
            Target data type.

        Returns
        -------
        BackendArray
            Flipped version of arr.
        """

    @abstractmethod
    def max_filter_coordinates(
        self, score_space: BackendArray, min_distance: Tuple[int]
    ) -> BackendArray:
        """
        Identifies local maxima in score_space separated by min_distance.

        Parameters
        ----------
        score_space : BackendArray
            Input score space.
        min_distance : tuple of ints
            Minimum distance along each array axis.

        Returns
        -------
        BackendArray
            Identified local maxima.
        """

    @abstractmethod
    def from_sharedarr(
        self, shape: Tuple[int], dtype: str, shm: shared_memory.SharedMemory
    ) -> BackendArray:
        """
        Returns an array of given shape and dtype from shared memory location.

        Parameters
        ----------
        shape : tuple
            Tuple of integers specifying the shape of the array.
        dtype : str
            String specifying the dtype of the array.
        shm : shared_memory.SharedMemory
            Shared memory object where the array is stored.

        Returns
        -------
        BackendArray
            Array of the specified shape and dtype from the shared memory location.
        """

    @abstractmethod
    def to_sharedarr(self, arr: type, shared_memory_handler: type = None) -> shm_type:
        """
        Converts an array to an object shared in memory. The output of this function
        will only be passed to :py:meth:`from_sharedarr`, hence the return values can
        be modified in particular backends to match the expected input data.

        Parameters
        ----------
        arr : BackendArray
            Numpy array to convert.
        shared_memory_handler : type, optional
            The type of shared memory handler. Default is None.

        Returns
        -------
        Tuple[shared_memory.SharedMemory, tuple of ints, dtype]
            The shared memory object containing the numpy array, its shape and dtype.
        """

    @abstractmethod
    def topleft_pad(
        self, arr: BackendArray, shape: Tuple[int], padval: int = 0
    ) -> BackendArray:
        """
        Returns an array that has been padded to a specified shape with a padding
        value at the top-left corner.

        Parameters
        ----------
        arr : BackendArray
            Input array to be padded.
        shape : Tuple[int]
            Desired shape for the output array.
        padval : int, optional
            Value to use for padding, default is 0.

        Returns
        -------
        BackendArray
            Array that has been padded to the specified shape.
        """

    @abstractmethod
    def rfftn(self, **kwargs):
        """Perform an n-D real FFT."""

    @abstractmethod
    def irfftn(self, **kwargs):
        """Perform an n-D real inverse FFT."""

    @abstractmethod
    def compute_convolution_shapes(
        self, arr1_shape: Tuple[int], arr2_shape: Tuple[int]
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Computes regular, optimized and fourier convolution shape.

        Parameters
        ----------
        arr1_shape : tuple of int
            Tuple of integers corresponding to array1 shape.
        arr2_shape : tuple of int
            Tuple of integers corresponding to array2 shape.

        Returns
        -------
        tuple
            Tuple with regular convolution shape, convolution shape optimized for faster
            fourier transform, shape of the forward fourier transform
            (see :py:meth:`build_fft`).
        """

    @abstractmethod
    def rigid_transform(
        self,
        arr: BackendArray,
        rotation_matrix: BackendArray,
        arr_mask: Optional[BackendArray] = None,
        translation: Optional[BackendArray] = None,
        use_geometric_center: bool = True,
        out: Optional[BackendArray] = None,
        out_mask: Optional[BackendArray] = None,
        order: int = 3,
        **kwargs,
    ) -> Tuple[BackendArray, Optional[BackendArray]]:
        """
        Performs a rigid transformation.

        Parameters
        ----------
        arr : BackendArray
            The input array to be rotated.
        arr_mask : BackendArray, optional
            The mask of `arr` that will be equivalently rotated.
        rotation_matrix : BackendArray
            The rotation matrix to apply (d, d).
        translation : BackendArray
            The translation to apply (d,).
        use_geometric_center : bool, optional
            Whether rotation should be performed over the center of mass.
        out : BackendArray, optional
            Location into which the rotation of ``arr`` is written.
        out_mask : BackendArray, optional
            Location into which the rotation of ``arr_mask`` is written.
        order : int, optional
            Interpolation order, one is linear and three is cubic. Specific
            meaning depends on backend.
        kwargs : dict, optional
            Keyword arguments relevant to particular backend implementations.

        Returns
        -------
        out, out_mask : BackendArray or None
            The rotated arrays.
        """

    @abstractmethod
    def get_available_memory(self) -> int:
        """
        Returns the available memory available for computations in bytes. For CPU
        operations this corresponds to available RAM. For GPU operations the function
        is expected to return the available GPU memory.
        """

    @abstractmethod
    def reverse(arr: BackendArray) -> BackendArray:
        """
        Reverse the order of elements in an array along all its axes.

        Parameters
        ----------
        arr : NDArray
            Input array.
        axis : tuple of int
            Axis to reverse, all by default.

        Returns
        -------
        NDArray
            Reversed array.
        """

    @abstractmethod
    def set_device(device_index: int) -> Generator:
        """
        Context manager that sets active compute device device for operations.

        Parameters
        ----------
        device_index : int
            Index of the device to be set as active.
        """

    @abstractmethod
    def device_count() -> int:
        """
        Return the number of available compute devices considered by the backend.

        Returns
        -------
        int
            Number of available devices.
        """
