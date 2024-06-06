import pytest
import numpy as np

from multiprocessing.managers import SharedMemoryManager

from tme.backends import MatchingBackend, NumpyFFTWBackend, BackendManager


BACKEND_CLASSES = ["NumpyFFTWBackend", "PytorchBackend", "CupyBackend", "MLXBackend"]
BACKENDS_TO_TEST = []
for backend_class in BACKEND_CLASSES:
    try:
        BackendClass = getattr(
            __import__("tme.backends", fromlist=[backend_class]), backend_class
        )
        BACKENDS_TO_TEST.append(BackendClass(device="cpu"))
    except ImportError:
        print(f"Couldn't import {backend_class}. Skipping...")


METHODS_TO_TEST = MatchingBackend.__abstractmethods__


class TestBackendManager:
    def setup_method(self):
        self.manager = BackendManager()

    def test_initialization(self):
        manager = BackendManager()
        backend_name = manager._backend_name
        assert f"<BackendManager: using {backend_name}>" == str(manager)

    def test_dir(self):
        _ = dir(self.manager)
        for method in METHODS_TO_TEST:
            assert hasattr(self.manager, method)

    def test_add_backend(self):
        self.manager.add_backend(backend_name="test", backend_class=NumpyFFTWBackend)

    def test_add_backend_error(self):
        class _Bar:
            def __init__(self):
                pass

        with pytest.raises(ValueError):
            self.manager.add_backend(backend_name="test", backend_class=_Bar)

    def test_change_backend_error(self):
        with pytest.raises(NotImplementedError):
            self.manager.change_backend(backend_name=None)


class TestBackends:
    def setup_method(self):
        self.backend = NumpyFFTWBackend()
        self.x1 = np.random.rand(30, 30).astype(np.float32)
        self.x2 = np.random.rand(30, 30).astype(np.float32)

    def teardown_method(self):
        self.backend = None

    def test_initialization_errors(self):
        with pytest.raises(TypeError):
            _ = MatchingBackend()

    @pytest.mark.parametrize("backend", [type(x) for x in BACKENDS_TO_TEST])
    def test_initialization(self, backend):
        _ = backend()

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize(
        "method_name",
        ("add", "subtract", "multiply", "divide", "minimum", "maximum", "mod"),
    )
    def test_arithmetic_operations(self, method_name, backend):
        base = getattr(self.backend, method_name)(self.x1, self.x2)
        x1 = backend.to_backend_array(self.x1)
        x2 = backend.to_backend_array(self.x2)
        other = getattr(backend, method_name)(x1, x2)

        assert np.allclose(base, backend.to_numpy_array(other))

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize(
        "method_name", ("sum", "mean", "std", "max", "min", "unique")
    )
    @pytest.mark.parametrize("axis", ((0), (1)))
    def test_reduction_operations(self, method_name, backend, axis):
        base = getattr(self.backend, method_name)(self.x1, axis=axis)
        other = getattr(backend, method_name)(
            backend.to_backend_array(self.x1), axis=axis
        )
        # Account for bessel function correction in pytorch
        rtol = 0.01 if method_name != "std" else 0.5
        assert np.allclose(base, backend.to_numpy_array(other), rtol=rtol)

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize(
        "method_name",
        ("sqrt", "square", "abs", "transpose", "tobytes", "size"),
    )
    def test_array_manipulation(self, method_name, backend):
        base = getattr(self.backend, method_name)(self.x1)
        other = getattr(backend, method_name)(backend.to_backend_array(self.x1))

        if type(base) == np.ndarray:
            assert np.allclose(base, backend.to_numpy_array(other), rtol=0.01)
        else:
            assert base == other

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize("shape", ((10, 15), (10, 15, 20)))
    def test_zeros(self, shape, backend):
        base = self.backend.zeros(shape)
        other = backend.zeros(shape)
        assert np.allclose(base, backend.to_numpy_array(other), rtol=0.01)

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize("shape", ((10, 15), (10, 15, 20)))
    @pytest.mark.parametrize(
        "dtype", (("_float_dtype", "_complex_dtype", "_int_dtype"))
    )
    def test_preallocate_array(self, shape, backend, dtype):
        dtype_base = getattr(self.backend, dtype)
        dtype_backend = getattr(backend, dtype)
        base = self.backend.preallocate_array(shape, dtype=dtype_base)
        other = backend.preallocate_array(shape, dtype=dtype_backend)
        assert np.allclose(base, backend.to_numpy_array(other), rtol=0.01)

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize("shape", ((10, 15), (10, 15, 20)))
    @pytest.mark.parametrize("fill_value", (-1, 0, 1))
    def test_full(self, shape, backend, fill_value):
        base = self.backend.full(shape, fill_value=fill_value)
        other = backend.full(shape, fill_value=fill_value)
        assert np.allclose(base, backend.to_numpy_array(other), rtol=0.01)

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize("power", (0.5, 1, 2))
    def test_power(self, backend, power):
        base = self.backend.power(self.x1, power)
        other = backend.power(backend.to_backend_array(self.x1), power)
        assert np.allclose(base, backend.to_numpy_array(other), rtol=0.01)

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize("shift", (-5, 0, 10))
    @pytest.mark.parametrize("axis", (0, 1))
    def test_roll(self, backend, shift, axis):
        base = self.backend.roll(self.x1, (shift,), (axis,))
        other = backend.roll(backend.to_backend_array(self.x1), (shift,), (axis,))
        assert np.allclose(base, backend.to_numpy_array(other), rtol=0.01)

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize("shape", ((10, 15), (10, 15, 20)))
    @pytest.mark.parametrize("fill_value", (-1, 0, 1))
    def test_fill(self, shape, backend, fill_value):
        base = self.backend.full(shape, fill_value=fill_value)
        other = backend.full(shape, fill_value=20)
        backend.fill(other, fill_value)
        assert np.allclose(base, backend.to_numpy_array(other), rtol=0.01)

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize("min_distance", (1, 5, 10))
    def test_max_filter_coordinates(self, backend, min_distance):
        coordinates = backend.max_filter_coordinates(
            backend.to_backend_array(self.x1), min_distance=min_distance
        )
        if len(coordinates):
            assert coordinates.shape[1] == self.x1.ndim
        assert True

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize(
        "dtype", (("_float_dtype", "_complex_dtype", "_int_dtype"))
    )
    @pytest.mark.parametrize(
        "dtype_target", (("_int_dtype", "_complex_dtype", "_float_dtype"))
    )
    def test_astype(self, dtype, backend, dtype_target):
        dtype_base = getattr(backend, dtype)
        dtype_target = getattr(backend, dtype_target)

        base = backend.zeros((20, 20, 20), dtype=dtype_base)
        arr = backend.astype(base, dtype_target)

        assert arr.dtype == dtype_target

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize("N", (0, 15, 30))
    def test_arange(self, backend, N):
        base = self.backend.arange(N)
        other = getattr(backend, "arange")(
            N,
        )
        assert np.allclose(base, backend.to_numpy_array(other), rtol=0.1)

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize("return_inverse", (False, True))
    @pytest.mark.parametrize("return_counts", (False, True))
    @pytest.mark.parametrize("return_index", (False, True))
    def test_unique(self, backend, return_inverse, return_counts, return_index):
        base = self.backend.unique(
            self.x1,
            return_inverse=return_inverse,
            return_counts=return_counts,
            return_index=return_index,
        )
        other = backend.unique(
            backend.to_backend_array(self.x1),
            return_inverse=return_inverse,
            return_counts=return_counts,
            return_index=return_index,
        )
        if type(base) != tuple:
            base, other = tuple(base), tuple(other)
        for k in range(len(base)):
            print(
                k,
                base[k].shape,
                other[k].shape,
                return_inverse,
                return_counts,
                return_index,
            )
            assert np.allclose(base[k], backend.to_numpy_array(other[k]), rtol=0.1)

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize("k", (0, 15, 30))
    def test_repeat(self, backend, k):
        base = self.backend.repeat(self.x1, k)
        other = backend.repeat(backend.to_backend_array(self.x1), k)
        assert np.allclose(base, backend.to_numpy_array(other), rtol=0.1)

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize("dim", (1, 3))
    @pytest.mark.parametrize("k", (0, 15, 30))
    def test_topk_indices(self, backend, k: int, dim: int):
        data = np.random.rand(*(50 for _ in range(dim)))
        base = self.backend.topk_indices(data, k)
        other = backend.topk_indices(backend.to_backend_array(data), k)

        for i in range(len(base)):
            np.allclose(
                base[i],
                backend.to_numpy_array(backend.to_backend_array(other[i])),
                rtol=0.1,
            )

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_indices(self, backend):
        base = self.backend.indices(self.x1.shape)
        other = backend.indices(backend.to_backend_array(self.x1).shape)
        print(base)
        print(other)
        assert np.allclose(base, backend.to_numpy_array(other), rtol=0.1)

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_get_available_memory(self, backend):
        mem = backend.get_available_memory()
        assert isinstance(mem, int)

    # @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    # def test_shared_memory(self, backend):
    #     shared_memory_handler = None
    #     base = backend.to_backend_array(self.x1)
    #     shared = backend.arr_to_sharedarr(
    #         arr=base, shared_memory_handler=shared_memory_handler
    #     )
    #     arr = backend.sharedarr_to_arr(shape=base.shape, dtype=base.dtype, shm=shared)
    #     assert np.allclose(backend.to_numpy_array(arr), backend.to_numpy_array(base))

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_shared_memory_managed(self, backend):
        with SharedMemoryManager() as shared_memory_handler:
            base = backend.to_backend_array(self.x1)
            shared = backend.arr_to_sharedarr(
                arr=base, shared_memory_handler=shared_memory_handler
            )
            arr = backend.sharedarr_to_arr(
                shape=base.shape, dtype=base.dtype, shm=shared
            )
            assert np.allclose(
                backend.to_numpy_array(arr), backend.to_numpy_array(base)
            )

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize("shape", ((10, 15, 100), (10, 15, 20)))
    @pytest.mark.parametrize("padval", (-1, 0, 1))
    def test_topleft_pad(self, backend, shape, padval):
        arr = np.random.rand(30, 30, 30)
        base = self.backend.topleft_pad(arr, shape=shape, padval=padval)
        other = backend.topleft_pad(
            backend.to_backend_array(arr), shape=shape, padval=padval
        )
        assert np.allclose(base, backend.to_numpy_array(other), rtol=0.01)

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize("fast_shape", ((10, 15, 100), (55, 23, 17)))
    def test_fft(self, backend, fast_shape):
        _, fast_shape, fast_ft_shape = backend.compute_convolution_shapes(
            fast_shape, (1 for _ in range(len(fast_shape)))
        )
        rfftn, irfftn = backend.build_fft(
            fast_shape=fast_shape,
            fast_ft_shape=fast_ft_shape,
            real_dtype=backend._float_dtype,
            complex_dtype=backend._complex_dtype,
        )
        arr = np.random.rand(*fast_shape)
        out = np.zeros(fast_ft_shape)

        real_arr = backend.astype(backend.to_backend_array(arr), backend._float_dtype)
        complex_arr = backend.astype(
            backend.to_backend_array(out), backend._complex_dtype
        )

        rfftn(
            backend.astype(backend.to_backend_array(arr), backend._float_dtype),
            complex_arr,
        )
        irfftn(complex_arr, real_arr)
        assert np.allclose(arr, backend.to_numpy_array(real_arr), rtol=0.3)

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_extract_center(self, backend):
        new_shape = np.divide(self.x1.shape, 2).astype(int)
        base = self.backend.extract_center(arr=self.x1, newshape=new_shape)
        other = backend.extract_center(
            arr=backend.to_backend_array(self.x1), newshape=new_shape
        )

        assert np.allclose(base, backend.to_numpy_array(other), rtol=0.01)

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_compute_convolution_shapes(self, backend):
        base = self.backend.compute_convolution_shapes(self.x1.shape, self.x2.shape)
        other = backend.compute_convolution_shapes(self.x1.shape, self.x2.shape)

        assert base == other

    @pytest.mark.parametrize("dim", (2, 3))
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize("create_mask", (False, True))
    def test_rotate_array(self, backend, dim, create_mask):
        shape = tuple(50 for _ in range(dim))
        arr = np.zeros(shape)
        if dim == 2:
            arr[20:25, 21:26] = 1
        elif dim == 3:
            arr[20:25, 21:26, 26:31] = 1

        rotation_matrix = np.eye(dim)
        rotation_matrix[0, 0] = -1

        out = np.zeros_like(arr)

        arr_mask, out_mask = None, None
        if create_mask:
            arr_mask = np.multiply(np.random.rand(*arr.shape) > 0.5, 1.0)
            out_mask = np.zeros_like(arr_mask)
            arr_mask = backend.to_backend_array(arr_mask)
            out_mask = backend.to_backend_array(out_mask)

        arr = backend.to_backend_array(arr)
        out = backend.to_backend_array(arr)

        rotation_matrix = backend.to_backend_array(rotation_matrix)

        backend.rotate_array(
            arr=arr,
            arr_mask=arr_mask,
            rotation_matrix=rotation_matrix,
            out=out,
            out_mask=out_mask,
        )

        assert np.round(arr.sum(), 3) == np.round(out.sum(), 3)

    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_datatype_bytes(self, backend):
        assert isinstance(backend.datatype_bytes(backend._float_dtype), int)
        assert isinstance(backend.datatype_bytes(backend._complex_dtype), int)
        assert isinstance(backend.datatype_bytes(backend._int_dtype), int)
