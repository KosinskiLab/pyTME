""" Strategy pattern to allow for flexible array / FFT backends.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from abc import ABC, abstractmethod
from typing import Tuple, Callable, List
from multiprocessing import shared_memory

from numpy.typing import NDArray
from ..types import ArrayLike, Scalar


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
    default_dtype : type
        Data type of real array instances, e.g. np.float32.
    complex_dtype : type
        Data type of complex array instances, e.g. np.complex64.
    default_dtype_int : type
        Data type of integer array instances, e.g. np.int32.

    Attributes
    ----------
    _array_backend : object
        The backend object used to delegate method and attribute calls.
    _default_dtype : type
        Data type of real array instances, e.g. np.float32.
    _complex_dtype : type
        Data type of complex array instances, e.g. np.complex64.
    _default_dtype_int : type
        Data type of integer array instances, e.g. np.int32.

    Examples
    --------
    >>> import numpy as np
    >>> backend = MatchingBackend(
        array_backend = np,
        default_dtype = np.float32,
        complex_dtype = np.complex64,
        default_dtype_int = np.int32
    )
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
        default_dtype: type,
        complex_dtype: type,
        default_dtype_int: type,
    ):
        self._array_backend = array_backend
        self._default_dtype = default_dtype
        self._complex_dtype = complex_dtype
        self._default_dtype_int = default_dtype_int

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

    @staticmethod
    def free_sharedarr(link: shared_memory.SharedMemory):
        """
        Free shared array at link.

        Parameters
        ----------
        link : shared_memory.SharedMemory
            The shared memory link to be freed.
        """
        if type(link) is not shared_memory.SharedMemory:
            return None
        try:
            link.close()
            link.unlink()
        # Shared memory has been freed already
        except FileNotFoundError:
            pass

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
    def to_cpu_array(self, arr: ArrayLike) -> NDArray:
        """
        Convert an array of a given backend to a CPU array of that backend.

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
            Optional output array to store the result. If provided, it must have a shape
            that the inputs broadcast to.

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
            Optional output array to store the result. If provided, it must have a shape
            that the inputs broadcast to.

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
            Optional output array to store the result. If provided, it must have a shape
            that the inputs broadcast to.

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
            The numerator array.
        arr2 : ArrayLike
            The denominator array.
        out : ArrayLike, optional
            Optional output array to store the result. If provided, it must have a shape
            that the inputs broadcast to.

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
            The numerator array.
        arr2 : ArrayLike
            The denominator array.
        out : ArrayLike, optional
            Optional output array to store the result. If provided, it must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ArrayLike
            Element-wise modulus of the input arrays.
        """

    @abstractmethod
    def sum(self, arr: ArrayLike) -> Scalar:
        """
        Compute the sum of array elements.

        Parameters
        ----------
        arr : ArrayLike
            The array whose sum should be computed.

        Returns
        -------
        Scalar
            Sum of the arr.
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
            Specifies the subscripts for summation as comma separated
            list of subscript label
        arr1 : ArrayLike
            Input array.
        arr2 : ArrayLike
            Input array.
        out : ArrayLike, optional
            Optional output array to store the result. If provided, it must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ArrayLike
            Element-wise sum of the input arrays.
        """

    @abstractmethod
    def mean(self, arr: ArrayLike) -> Scalar:
        """
        Compute the mean of array elements.

        Parameters
        ----------
        arr : ArrayLike
            The array whose mean should be computed.

        Returns
        -------
        Scalar
            Mean value of the arr.
        """

    @abstractmethod
    def std(self, arr: ArrayLike) -> Scalar:
        """
        Compute the standad deviation of array elements.

        Parameters
        ----------
        arr : Scalar
            The array whose standard deviation should be computed.

        Returns
        -------
        Scalar
            The standard deviation of arr.
        """

    @abstractmethod
    def max(self, arr: ArrayLike) -> Scalar:
        """
        Compute the maximum of array elements.

        Parameters
        ----------
        arr : Scalar
            The array whose maximum should be computed.

        Returns
        -------
        Scalar
            The maximum of arr.
        """

    @abstractmethod
    def min(self, arr: ArrayLike) -> Scalar:
        """
        Compute the minimum of array elements.

        Parameters
        ----------
        arr : Scalar
            The array whose maximum should be computed.

        Returns
        -------
        Scalar
            The maximum of arr.
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
            Arrays for which element wise maximum will be computed.
        out : ArrayLike, optional
            Output array.

        Returns
        -------
        ArrayLike
            Element wise maximum of arr1 and arr2.
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
            Arrays for which element wise minimum will be computed.
        out : ArrayLike, optional
            Output array.

        Returns
        -------
        ArrayLike
            Element wise minimum of arr1 and arr2.
        """

    @abstractmethod
    def sqrt(self, arr: ArrayLike) -> ArrayLike:
        """
        Compute the square root of array elements.

        Parameters
        ----------
        arr : ArrayLike
            The array whose square root should be computed.

        Returns
        -------
        ArrayLike
            The squared root of the array.
        """

    @abstractmethod
    def square(self, arr: ArrayLike) -> ArrayLike:
        """
        Compute the square of array elements.

        Parameters
        ----------
        arr : ArrayLike
            The array that should be squared.

        Returns
        -------
        ArrayLike
            The squared arr.
        """

    @abstractmethod
    def abs(self, arr: ArrayLike) -> ArrayLike:
        """
        Compute the absolute of array elements.

        Parameters
        ----------
        arr : ArrayLike
            The array whose absolte should be computed.

        Returns
        -------
        ArrayLike
            The absolute of the array.
        """

    @abstractmethod
    def transpose(self, arr: ArrayLike) -> ArrayLike:
        """
        Compute the transpose of arr.

        Parameters
        ----------
        arr : ArrayLike
            Input array.

        Returns
        -------
        ArrayLike
            The transpose of arr.
        """

    def power(self, *args, **kwargs) -> ArrayLike:
        """
        Compute the n-th power of an array.

        Returns
        -------
        ArrayLike
            The n-th power of the array
        """

    def tobytes(self, arr: ArrayLike) -> str:
        """
        Compute the bytestring representation of arr.

        Parameters
        ----------
        arr : ArrayLike
            Input array.

        Returns
        -------
        str
            Bytestring representation of arr.
        """

    @abstractmethod
    def size(self, arr: ArrayLike) -> int:
        """
        Compute the number of elements of arr.

        Parameters
        ----------
        arr : ArrayLike
            Input array.

        Returns
        -------
        int
            The number of elements in arr.
        """

    @abstractmethod
    def fill(self, arr: ArrayLike, value: float) -> ArrayLike:
        """
        Fill arr with value.

        Parameters
        ----------
        arr : ArrayLike
            The array that should be filled.
        value : float
            The value with which to fill the array.

        Returns
        -------
        ArrayLike
            The array filled with the given value.
        """

    @abstractmethod
    def zeros(self, shape: Tuple[int], dtype: type) -> ArrayLike:
        """
        Returns an aligned array of zeros with specified shape and dtype.

        Parameters
        ----------
        shape : Tuple[int]
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
        shape : Tuple[int]
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
            Datatype for which the number of bytes is to be determined.
            This is typically a data type like `np.float32` or `np.int64`.

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
            Input array.
        a_min : Scalar
            Lower bound.
        a_max : Scalar
            Upper bound.
        out : ArrayLike, optional
            Output array.

        Returns
        -------
        ArrayLike
            Clipped arr.
        """

    @abstractmethod
    def flip(
        self, arr: ArrayLike, a_min: Scalar, a_max: Scalar, out: ArrayLike = None
    ) -> ArrayLike:
        """
        Flip the elements of arr.

        Parameters
        ----------
        arr : ArrayLike
            Input array.

        Returns
        -------
        ArrayLike
            Flipped version of arr.
        """

    @abstractmethod
    def astype(arr: ArrayLike, dtype: type) -> ArrayLike:
        """
        Change the datatype of arr.

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
    def arange(self, *args, **kwargs) -> ArrayLike:
        """
        Arange values in increasing order.

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

    def stack(self, *args, **kwargs) -> ArrayLike:
        """
        Join a sequence of ArrayLike objects along a specified axis.

        Returns
        -------
        ArrayLike
            The joined input objects.
        """

    @abstractmethod
    def concatenate(self, *args, **kwargs) -> ArrayLike:
        """
        Concatenates arrays along axis.

        Parameters
        ----------
        arr * : ArrayLike
            Input arrays

        Returns
        -------
        ArrayLike
            Concatenated input arrays
        """

    @abstractmethod
    def repeat(self, *args, **kwargs) -> ArrayLike:
        """
        Repeat each array elements after themselves.

        Parameters
        ----------
        arr : ArrayLike
            Input array

        Returns
        -------
        ArrayLike
            Repeated input array
        """

    @abstractmethod
    def topk_indices(self, arr: NDArray, k: int) -> ArrayLike:
        """
        Compute indices of largest elements.

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
        *args : tuple
            Generic arguments.
        **kwargs : dict
            Generic keyword arguments.

        Returns
        -------
        ArrayLike
            Array with elements rolled.

        See Also
        --------
        numpy.roll : For more detailed documentation on arguments and behavior.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> np.roll(x, 2)
        array([4, 5, 1, 2, 3])
        """

    @abstractmethod
    def unique(self, *args, **kwargs) -> Tuple[ArrayLike]:
        """
        Find the unique elements of an array.

        Parameters
        ----------
        *args : tuple
            Generic arguments.
        **kwargs : dict
            Generic keyword arguments.

        Returns
        -------
        ArrayLike or tuple of ArrayLike
            If `return_index`, `return_inverse`, and `return_counts` keyword
            arguments are all False (the default), this will be an ArrayLike object
            of the sorted unique values. Otherwise, it's a tuple with one
            or more arrays as specified by those keyword arguments.

        See Also
        --------
        numpy.unique : For more detailed documentation on arguments and behavior.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.array([1, 2, 3, 2, 3, 4])
        >>> np.unique(x)
        array([1, 2, 3, 4])
        """

    @abstractmethod
    def argsort(self, *args, **kwargs) -> ArrayLike:
        """
        Perform argsort of arr.

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
    def unravel_index(self, indices: ArrayLike, shape: Tuple[int]) -> Tuple[ArrayLike]:
        """
        Convert flat index to array indices.

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
    def preallocate_array(self, shape: Tuple[int], dtype: type) -> ArrayLike:
        """
        Returns an aligned array of zeros with specified shape and dtype.

        Parameters
        ----------
        shape : Tuple[int]
            Desired shape for the array.
        dtype : type
            Desired data type for the array.

        Returns
        -------
        ArrayLike
            Byte-aligned array of zeros with specified shape and dtype.
        """

    @abstractmethod
    def sharedarr_to_arr(
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
    def arr_to_sharedarr(
        self, arr: type, shared_memory_handler: type = None
    ) -> shared_memory.SharedMemory:
        """
        Converts an array to an object shared in memory.

        Parameters
        ----------
        arr : ArrayLike
            Numpy array to convert.
        shared_memory_handler : type, optional
            The type of shared memory handler. Default is None.

        Returns
        -------
        shared_memory.SharedMemory
            The shared memory object containing the numpy array.
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
        Build forward and inverse real fourier transform functions.

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
        fftargs : dict, optional
            Dictionary passed to pyFFTW builders.
        temp_real : ArrayLike, optional
            Temporary real numpy array, by default None.
        temp_fft : ArrayLike, optional
            Temporary fft numpy array, by default None.

        Returns
        -------
        tuple
            Tuple containing function pointers for forward and inverse real
            fourier transform
        """

    def extract_center(self, arr: ArrayLike, newshape: Tuple[int]) -> ArrayLike:
        """
        Extract the centered portion of an array based on a new shape.

        Parameters
        ----------
        arr : ArrayLike
            Input array.
        newshape : tuple
            Desired shape for the central portion.

        Returns
        -------
        ArrayLike
            Central portion of the array with shape `newshape`.
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
    def rotate_array(self, *args, **kwargs):
        """
        Perform a rigid transform of arr.

        Parameters
        ----------
        arr : ArrayLike
            Input array.

        Returns
        -------
        ArrayLike
            Transformed version of arr.
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
        Set the active device context and operate as context manager for it
        """

    @abstractmethod
    def device_count() -> int:
        """
        Return the number of available compute devices.
        """
