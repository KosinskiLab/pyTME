""" Strategy pattern to allow for flexible array / FFT backends.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from abc import ABC, abstractmethod
from typing import Tuple, Callable, List, Any
from multiprocessing import shared_memory

from numpy.typing import NDArray
from ..types import ArrayLike, Scalar, shm_type


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
    >>>     array_backend = np,
    >>>     float_dtype = np.float32,
    >>>     complex_dtype = np.complex64,
    >>>     int_dtype = np.int32
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
    def to_backend_array(self, arr: NDArray) -> ArrayLike:
        """
        Convert a numpy array instance to backend array type.

        Parameters
        ----------
        arr : NDArray
            The numpy array instance to be converted.

        Returns
        -------
        ArrayLike
            An array of the specified backend.

        See Also
        --------
        :py:meth:`MatchingBackend.to_cpu_array`
        :py:meth:`MatchingBackend.to_numpy_array`
        """

    @abstractmethod
    def to_numpy_array(self, arr: ArrayLike) -> NDArray:
        """
        Convert an array of given backend to a numpy array.

        Parameters
        ----------
        arr : NDArray
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
    def to_cpu_array(self, arr: ArrayLike) -> ArrayLike:
        """
        Convert an array of a given backend to a CPU array of that backend.

        Parameters
        ----------
        arr : NDArray
            The array instance to be converted.

        Returns
        -------
        ArrayLike
            The CPU array equivalent of arr.

        See Also
        --------
        :py:meth:`MatchingBackend.to_numpy_array`
        :py:meth:`MatchingBackend.to_backend_array`
        """

    @abstractmethod
    def free_cache(self):
        """
        Free cached objects allocated by backend.
        """

    @abstractmethod
    def add(self, arr1: ArrayLike, arr2: ArrayLike, out: ArrayLike = None) -> ArrayLike:
        """
        Interface for element-wise addition of arrays.

        Parameters
        ----------
        arr1 : ArrayLike
            Input array.
        arr2 : ArrayLike
            Input array.
        out : ArrayLike, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        ArrayLike
            Element-wise sum of the input arrays.
        """

    @abstractmethod
    def subtract(
        self, arr1: ArrayLike, arr2: ArrayLike, out: ArrayLike = None
    ) -> ArrayLike:
        """
        Interface for element-wise subtraction of arrays.

        Parameters
        ----------
        arr1 : ArrayLike
            The minuend array.
        arr2 : ArrayLike
            The subtrahend array.
        out : ArrayLike, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        ArrayLike
            Element-wise difference of the input arrays.
        """

    @abstractmethod
    def multiply(
        self, arr1: ArrayLike, arr2: ArrayLike, out: ArrayLike = None
    ) -> ArrayLike:
        """
        Interface for element-wise multiplication of arrays.

        Parameters
        ----------
        arr1 : ArrayLike
            Input array.
        arr2 : ArrayLike
            Input array.
        out : ArrayLike, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        ArrayLike
            Element-wise product of the input arrays.
        """

    @abstractmethod
    def divide(
        self, arr1: ArrayLike, arr2: ArrayLike, out: ArrayLike = None
    ) -> ArrayLike:
        """
        Interface for element-wise division of arrays.

        Parameters
        ----------
        arr1 : ArrayLike
            The dividend array.
        arr2 : ArrayLike
            The divisor array.
        out : ArrayLike, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        ArrayLike
            Element-wise quotient of the input arrays.
        """

    @abstractmethod
    def mod(self, arr1: ArrayLike, arr2: ArrayLike, out: ArrayLike = None) -> ArrayLike:
        """
        Interface for element-wise modulus of arrays.

        Parameters
        ----------
        arr1 : ArrayLike
            The dividend array.
        arr2 : ArrayLike
            The divisor array.
        out : ArrayLike, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        ArrayLike
            Element-wise modulus of the input arrays.
        """

    @abstractmethod
    def einsum(
        self, arr1: ArrayLike, arr2: ArrayLike, out: ArrayLike = None
    ) -> ArrayLike:
        """
        Interface for einstein notation based summation.

        Parameters
        ----------
        subscripts : str
            Specifies the subscripts for summation (see  :obj:`numpy.einsum`).
        arr1, arr2 : ArrayLike
            Input data.
        out : ArrayLike, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        ArrayLike
            Einsum of input arrays.
        """

    @abstractmethod
    def sum(self, arr: ArrayLike, axis: Tuple[int] = None) -> ArrayLike:
        """
        Compute the sum of array elements.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        axis : int or tuple of ints, optional
            Axis or axes to perform the operation on. Default is all.

        Returns
        -------
        ArrayLike
            Sum of ``arr``.
        """

    @abstractmethod
    def mean(self, arr: ArrayLike, axis: Tuple[int] = None) -> ArrayLike:
        """
        Compute the mean of array elements.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        axis : int or tuple of ints, optional
            Axis or axes to perform the operation on. Default is all.

        Returns
        -------
        ArrayLike
            Mean of ``arr``.
        """

    @abstractmethod
    def std(self, arr: ArrayLike, axis: Tuple[int] = None) -> ArrayLike:
        """
        Compute the standad deviation of array elements.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        axis : int or tuple of ints, optional
            Axis or axes to perform the operation on. Default is all.

        Returns
        -------
        ArrayLike
            Standard deviation of ``arr``.
        """

    @abstractmethod
    def max(self, arr: ArrayLike, axis: Tuple[int] = None) -> ArrayLike:
        """
        Compute the maximum of array elements.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        axis : int or tuple of ints, optional
            Axis or axes to perform the operation on. Default is all.

        Returns
        -------
        ArrayLike
            Maximum of ``arr``.
        """

    @abstractmethod
    def min(self, arr: ArrayLike, axis: Tuple[int] = None) -> ArrayLike:
        """
        Compute the minimum of array elements.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        axis : int or tuple of ints, optional
            Axis or axes to perform the operation on. Default is all.

        Returns
        -------
        ArrayLike
            Minimum of ``arr``.
        """

    @abstractmethod
    def maximum(
        self, arr1: ArrayLike, arr2: ArrayLike, out: ArrayLike = None
    ) -> Scalar:
        """
        Compute the element wise maximum of arr1 and arr2.

        Parameters
        ----------
        arr1, arr2 : ArrayLike
            Input data.
        out : ArrayLike, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        ArrayLike
            Element wise maximum of ``arr1`` and ``arr2``.
        """

    @abstractmethod
    def minimum(
        self, arr1: ArrayLike, arr2: ArrayLike, out: ArrayLike = None
    ) -> Scalar:
        """
        Compute the element wise minimum of arr1 and arr2.

        Parameters
        ----------
        arr1, arr2 : ArrayLike
            Input data.
        out : ArrayLike, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        ArrayLike
            Element wise minimum of arr1 and arr2.
        """

    @abstractmethod
    def sqrt(self, arr: ArrayLike, out: ArrayLike = None) -> ArrayLike:
        """
        Compute the square root of array elements.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        out : ArrayLike, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        ArrayLike
            Square root of ``arr``.
        """

    @abstractmethod
    def square(self, arr: ArrayLike, out: ArrayLike = None) -> ArrayLike:
        """
        Compute the square of array elements.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        out : ArrayLike, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        ArrayLike
            Square of ``arr``.
        """

    @abstractmethod
    def abs(self, arr: ArrayLike, out: ArrayLike = None) -> ArrayLike:
        """
        Compute the absolute of array elements.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        out : ArrayLike, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        ArrayLike
            Absolute value of ``arr``.
        """

    @abstractmethod
    def transpose(self, arr: ArrayLike) -> ArrayLike:
        """
        Compute the transpose of arr.

        Parameters
        ----------
        arr : ArrayLike
            Input data.

        Returns
        -------
        ArrayLike
            Transpose of ``arr``.
        """

    def power(
        self,
        arr: ArrayLike = None,
        power: ArrayLike = None,
        out: ArrayLike = None,
        *args,
        **kwargs,
    ) -> ArrayLike:
        """
        Compute the n-th power of an array.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        power : ArrayLike
            Power to raise ``arr`` to.
        arr : ArrayLike
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        ArrayLike
            N-th power of ``arr``.
        """

    def tobytes(self, arr: ArrayLike) -> str:
        """
        Compute the bytestring representation of arr.

        Parameters
        ----------
        arr : ArrayLike
            Input data.

        Returns
        -------
        str
            Bytestring representation of ``arr``.
        """

    @abstractmethod
    def size(self, arr: ArrayLike) -> int:
        """
        Compute the number of elements of arr.

        Parameters
        ----------
        arr : ArrayLike
            Input data.

        Returns
        -------
        int
            Number of elements in ``arr``.
        """

    @abstractmethod
    def fill(self, arr: ArrayLike, value: Scalar) -> None:
        """
        Fills ``arr`` in-place with a given value.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        value : Scalar
            The value to fill the array with.
        """

    @abstractmethod
    def zeros(self, shape: Tuple[int], dtype: type) -> ArrayLike:
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
        ArrayLike
            Byte-aligned array of zeros with specified shape and dtype.
        """

    @abstractmethod
    def full(self, shape: Tuple[int], dtype: type, fill_value: Scalar) -> ArrayLike:
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
        ArrayLike
            Byte-aligned array of zeros with specified shape and dtype.
        """

    @abstractmethod
    def eps(self, dtype: type) -> Scalar:
        """
        Returns the eps defined as diffeerence between 1.0 and the next
        representable floating point value larger than 1.0.

        Parameters
        ----------
        dtype : type
            Data type for which eps should be returned.

        Returns
        -------
        Scalar
            The eps for the given data type
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

        Examples
        --------
        >>> MatchingBackend.datatype_bytes(np.float32)
        4
        """

    @abstractmethod
    def clip(
        self, arr: ArrayLike, a_min: Scalar, a_max: Scalar, out: ArrayLike = None
    ) -> ArrayLike:
        """
        Clip elements of arr.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        a_min : Scalar
            Lower bound.
        a_max : Scalar
            Upper bound.
        out : ArrayLike, optional
            Output array to write the result to. Returns a new array by default.

        Returns
        -------
        ArrayLike
            Clipped ``arr``.
        """

    @abstractmethod
    def flip(self, arr: ArrayLike, axis: Tuple[int] = None) -> ArrayLike:
        """
        Flip the elements of arr.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        axis : int or tuple of ints, optional
            Axis or axes to perform the operation on. Default is all.

        Returns
        -------
        ArrayLike
            Flipped version of ``arr``.
        """

    @abstractmethod
    def astype(arr: ArrayLike, dtype: type) -> ArrayLike:
        """
        Change the datatype of arr.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        dtype : type
            Target data type.

        Returns
        -------
        ArrayLike
            Freshly allocated array containing the data of ``arr`` in ``dtype``.
        """

    @abstractmethod
    def arange(
        self, stop: Scalar, start: Scalar = 0, step: Scalar = 1, *args, **kwargs
    ) -> ArrayLike:
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
        ArrayLike
            Array of evenly spaced values in specified interval.
        """

    def stack(self, *args, **kwargs) -> ArrayLike:
        """
        Join a sequence of objects along a new axis.

        Parameters
        ----------
        arr : ArrayLike
            Sequence of arrays.
        axis : int, optional
            Axis along which to stack the input arrays.

        Returns
        -------
        ArrayLike
            Stacked input data.
        """

    @abstractmethod
    def concatenate(self, *args, **kwargs) -> ArrayLike:
        """
        Join a sequence of objects along an existing axis.

        Parameters
        ----------
        arr : ArrayLike
            Sequence of arrays.
        axis : int
            Axis along which to stack the input arrays.

        Returns
        -------
        ArrayLike
            Concatenated input data.
        """

    @abstractmethod
    def repeat(self, *args, **kwargs) -> ArrayLike:
        """
        Repeat each array element a specified number of times.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        repeats : int or tuple of ints
            Number of each repetitions along axis.

        Returns
        -------
        ArrayLike
            Repeated ``arr``.
        """

    @abstractmethod
    def topk_indices(self, arr: NDArray, k: int) -> ArrayLike:
        """
        Determinces the indices of largest elements.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        k : int
            Number of maxima to determine.

        Returns
        -------
        ArrayLike
            Indices of ``k`` largest elements in ``arr``.
        """

    def indices(self, *args, **kwargs) -> ArrayLike:
        """
        Creates an array representing the index grid of an input.

        Returns
        -------
        ArrayLike
            The index grid.
        """

    @abstractmethod
    def roll(self, *args, **kwargs) -> ArrayLike:
        """
        Roll array elements along a specified axis.

        Parameters
        ----------
        a : ArrayLike
            Input data.
        shift : int or tuple of ints, optional
            Shift along each axis.

        Returns
        -------
        ArrayLike
            Array with elements rolled.
        """

    @abstractmethod
    def where(condition, *args) -> ArrayLike:
        """
        Return elements from input depending on ``condition``.

        Parameters
        ----------
        condition : ArrayLike
            Binary condition array.
        *args : ArrayLike
            Values to choose from.

        Returns
        -------
        ArrayLike
            Array of elements according to ``condition``.
        """

    @abstractmethod
    def unique(
        self,
        arr: ArrayLike,
        return_index: bool = False,
        return_inverse: bool = False,
        return_counts: bool = False,
        axis: Tuple[int] = None,
        *args,
        **kwargs,
    ) -> Tuple[ArrayLike]:
        """
        Find the unique elements of an array.

        Parameters
        ----------
        arr : ArrayLike
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
        ArrayLike or tuple of ArrayLike
            If `return_index`, `return_inverse`, and `return_counts` keyword
            arguments are all False (the default), this will be an ArrayLike object
            of the sorted unique values. Otherwise, it's a tuple with one
            or more arrays as specified by those keyword arguments.
        """

    @abstractmethod
    def argsort(self, *args, **kwargs) -> ArrayLike:
        """
        Compute the indices to sort a given input array.

        Parameters
        ----------
        arr : ArrayLike
            Input array.
        dtype : type
            Target data type.

        Returns
        -------
        ArrayLike
            Indices that would sort the input data.
        """

    @abstractmethod
    def unravel_index(self, indices: ArrayLike, shape: Tuple[int]) -> Tuple[ArrayLike]:
        """
        Convert flat index to array indices.

        Parameters
        ----------
        indices : ArrayLike
            Input data.
        shape : tuple of ints
            Shape of the array used for unraveling.

        Returns
        -------
        ArrayLike
            Array indices.
        """

    @abstractmethod
    def tril_indices(self, *args, **kwargs) -> ArrayLike:
        """
        Compute indices of upper triangular matrix

        Parameters
        ----------
        arr : ArrayLike
            Input array.
        dtype : type
            Target data type.

        Returns
        -------
        ArrayLike
            Flipped version of arr.
        """

    @abstractmethod
    def max_filter_coordinates(
        self, score_space: ArrayLike, min_distance: Tuple[int]
    ) -> ArrayLike:
        """
        Identifies local maxima in score_space separated by min_distance.

        Parameters
        ----------
        score_space : ArrayLike
            Input score space.
        min_distance : tuple of ints
            Minimum distance along each array axis.

        Returns
        -------
        ArrayLike
            Identified local maxima.
        """

    @abstractmethod
    def from_sharedarr(
        self, shape: Tuple[int], dtype: str, shm: shared_memory.SharedMemory
    ) -> ArrayLike:
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
        ArrayLike
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
        arr : ArrayLike
            Numpy array to convert.
        shared_memory_handler : type, optional
            The type of shared memory handler. Default is None.

        Returns
        -------
        Tupleshared_memory.SharedMemory, tuple of ints, dtype]
            The shared memory object containing the numpy array, its shape and dtype.
        """

    @abstractmethod
    def topleft_pad(
        self, arr: ArrayLike, shape: Tuple[int], padval: int = 0
    ) -> ArrayLike:
        """
        Returns an array that has been padded to a specified shape with a padding
        value at the top-left corner.

        Parameters
        ----------
        arr : ArrayLike
            Input array to be padded.
        shape : Tuple[int]
            Desired shape for the output array.
        padval : int, optional
            Value to use for padding, default is 0.

        Returns
        -------
        ArrayLike
            Array that has been padded to the specified shape.
        """

    @abstractmethod
    def build_fft(
        self,
        fast_shape: Tuple[int],
        fast_ft_shape: Tuple[int],
        real_dtype: type,
        complex_dtype: type,
        **kwargs,
    ) -> Tuple[Callable, Callable]:
        """
        Build forward and inverse real fourier transform functions. The returned
        callables have two parameters ``arr`` and ``out`` which correspond to the
        input and output of the Fourier transform. The methods return the output
        of the respective function call, regardless of ``out`` being provided or not,
        analogous to most numpy functions.

        Parameters
        ----------
        fast_shape : tuple
            Tuple of integers corresponding to fast convolution shape
            (see `compute_convolution_shapes`).
        fast_ft_shape : tuple
            Tuple of integers corresponding to the shape of the fourier
            transform array (see `compute_convolution_shapes`).
        real_dtype : dtype
            Numpy dtype of the inverse fourier transform.
        complex_dtype : dtype
            Numpy dtype of the fourier transform.
        inverse_fast_shape : tuple, optional
            Output shape of the inverse Fourier transform. By default fast_shape.
        fftargs : dict, optional
            Dictionary passed to pyFFTW builders.
        temp_real : NDArray, optional
            Temporary real numpy array, by default None.
        temp_fft : NDArray, optional
            Temporary fft numpy array, by default None.

        Returns
        -------
        tuple
            Tuple of callables for forward and inverse real Fourier transform.
        """

    def extract_center(self, arr: ArrayLike, newshape: Tuple[int]) -> ArrayLike:
        """
        Extract the centered portion of an array based on a new shape.

        Parameters
        ----------
        arr : ArrayLike
            Input data.
        newshape : tuple
            Desired shape for the central portion.

        Returns
        -------
        ArrayLike
            Central portion of the array with shape ``newshape``.

        References
        ----------
        .. [1] https://github.com/scipy/scipy/blob/v1.11.2/scipy/signal/_signaltools.py
        """

    @abstractmethod
    def compute_convolution_shapes(
        self, arr1_shape: Tuple[int], arr2_shape: Tuple[int]
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Computes regular, optimized and fourier convolution shape.

        Parameters
        ----------
        arr1_shape : tuple
            Tuple of integers corresponding to array1 shape.
        arr2_shape : tuple
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
        arr: NDArray,
        rotation_matrix: NDArray,
        arr_mask: NDArray = None,
        translation: NDArray = None,
        use_geometric_center: bool = True,
        out: NDArray = None,
        out_mask: NDArray = None,
        order: int = 3,
        **kwargs,
    ) -> None:
        """
        Performs a rigid transformation.

        Parameters
        ----------
        arr : ArrayLike
            The input array to be rotated.
        arr_mask : ArrayLike, optional
            The mask of `arr` that will be equivalently rotated.
        rotation_matrix : ArrayLike
            The rotation matrix to apply (d, d).
        translation : ArrayLike
            The translation to apply (d,).
        use_geometric_center : bool, optional
            Whether rotation should be performed over the center of mass.
        out : ArrayLike, optional
            Location into which the rotation of ``arr`` is written.
        out_mask : ArrayLike, optional
            Location into which the rotation of ``arr_mask`` is written.
        order : int, optional
            Interpolation order, one is linear and three is cubic. Specific
            meaning depends on backend.
        kwargs : dict, optional
            Keyword arguments relevant to particular backend implementations.
        """

    @abstractmethod
    def get_available_memory(self) -> int:
        """
        Returns the available memory available for computations in bytes. For CPU
        operations this corresponds to available RAM. For GPU operations the function
        is expected to return the available GPU memory.
        """

    @abstractmethod
    def reverse(arr: ArrayLike) -> ArrayLike:
        """
        Reverse the order of elements in an array along all its axes.

        Parameters
        ----------
        arr : ArrayLike
            Input array.

        Returns
        -------
        ArrayLike
            Reversed array.
        """

    @abstractmethod
    def set_device(device_index: int):
        """
        Set the active GPU device as a context.

        This method sets the active GPU device for operations within the context.

        Parameters
        ----------
        device_index : int
            Index of the GPU device to be set as active.

        Yields
        ------
        None
            Operates as a context manager, yielding None and providing
            the set GPU context for enclosed operations.
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
