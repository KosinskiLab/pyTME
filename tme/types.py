from typing import Union, TypeVar, Tuple

BackendArray = TypeVar("backend_array")
NDArray = TypeVar("numpy.ndarray")
CupyArray = TypeVar("cupy.ndarray")
TorchTensor = TypeVar("torch.Tensor")
MlxArray = TypeVar("mlx.core.array")
ArrayLike = TypeVar("array_like")

Scalar = Union[int, float, complex]
shm_type = Union[Tuple[type, Tuple[int], type], BackendArray]
MatchingData = TypeVar("MatchingData")
CallbackClass = TypeVar("callback_class")
