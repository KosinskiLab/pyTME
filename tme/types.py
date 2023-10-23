from typing import Union, TypeVar

NDArray = TypeVar("numpy.ndarray")
CupyArray = TypeVar("cupy.ndarray")
TorchTensor = TypeVar("torch.Tensor")

Scalar = Union[int, float, complex]

ArrayLike = TypeVar("array_like")
CallbackClass = TypeVar("callback_class")
