import pytest
import numpy as np
from tme.utils.serialization import serialize, deserialize


@pytest.fixture
def temp_file(tmp_path):
    """Fixture to provide temporary file paths."""

    def _make_temp(suffix=""):
        return str(tmp_path / f"test_file{suffix}")

    return _make_temp


@pytest.mark.parametrize("file_format", ["pickle", "hdf5"])
@pytest.mark.parametrize("use_compression", [False, True])
def test_serialize_roundtrip(temp_file, file_format, use_compression):
    """Test serialization roundtrip with different formats and compression."""
    kwargs = {}
    if file_format == "pickle":
        filename = temp_file(".pickle.gz" if use_compression else ".pickle")
    else:
        filename = temp_file(".h5")
        kwargs = {"compression": "gzip" if use_compression else None}

    data = [
        np.array([1, 2, 3], dtype=np.int32),
        {"key1": np.array([4, 5]), "key2": 10},
        (1, "test"),
        np.zeros((10, 20), dtype=np.float32),
    ]

    serialize(data, filename, **kwargs)
    loaded_data = deserialize(filename)

    assert len(loaded_data) == len(data)
    assert np.array_equal(loaded_data[0], data[0])
    assert np.array_equal(loaded_data[1]["key1"], data[1]["key1"])
    assert loaded_data[2] == data[2]
    assert np.array_equal(loaded_data[3], data[3])


@pytest.mark.parametrize("use_compression", [False, True])
def test_pickle_memmap(temp_file, use_compression):
    """Test pickle serialization with memmap."""
    memmap_file = temp_file(".mm")
    pickle_file = temp_file(".pickle.gz" if use_compression else ".pickle")

    data = np.memmap(memmap_file, dtype="float32", mode="w+", shape=(3,))
    data[:] = [1.1, 2.2, 3.3]
    data.flush()

    data = np.memmap(memmap_file, dtype="float32", mode="r", shape=(3,))

    serialize(data=data, filename=pickle_file, use_gzip=use_compression)
    loaded_data = deserialize(pickle_file)

    assert isinstance(loaded_data, np.memmap)
    assert np.array_equal(data, loaded_data)


def test_hdf5_lazy_loading(temp_file):
    """Test HDF5 lazy loading and unpacking."""
    filename = temp_file(".h5")

    data = [np.array([1, 2, 3]), {"key": np.array([4, 5])}, np.array([6, 7, 8])]

    serialize(data, filename, file_format="hdf5")
    loader = deserialize(filename, file_format="hdf5", lazy=True)
    arr1, dict_data, arr2 = loader

    assert np.array_equal(arr1, data[0])
    assert np.array_equal(dict_data["key"], data[1]["key"])
    assert np.array_equal(arr2, data[2])


def test_auto_format_detection(temp_file):
    """Test automatic format detection based on file extension."""
    data = [np.array([1, 2, 3]), np.array([4, 5, 6])]

    h5_file = temp_file(".h5")
    serialize(data, h5_file, file_format="auto")
    loaded_h5 = deserialize(h5_file, file_format="auto")

    pickle_file = temp_file(".pickle")
    serialize(data, pickle_file, file_format="auto")
    loaded_pickle = deserialize(pickle_file, file_format="auto")

    assert all(np.array_equal(a, b) for a, b in zip(loaded_h5, data))
    assert all(np.array_equal(a, b) for a, b in zip(loaded_pickle, data))


def test_extension_validation(temp_file):
    """Test that HDF5 format requires correct file extension."""
    data = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        serialize(data, temp_file(".pkl"), file_format="hdf5")

    with pytest.raises(ValueError):
        deserialize(temp_file(".pkl"), file_format="hdf5")
