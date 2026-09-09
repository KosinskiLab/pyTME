"""
Serialize/deserialize template matching results using pickle and hdf5.

Copyright (c) 2025 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import pickle

from shutil import copyfile
from os import makedirs, replace
from gzip import open as gzip_open
from os.path import exists, dirname
from typing import Any, List, Union, Optional
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

from ..matching_utils import generate_tempfile_name

__all__ = ["serialize", "deserialize", "is_gzipped"]


class HDF5Loader:
    """
    Lazy loader for HDF5 files with random access.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    """

    def __init__(self, filename: str):
        if not exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        self.filename = filename
        self._file = h5py.File(filename, "r")
        self._num_items = self._file.attrs.get("num_items", 0)

    def __del__(self):
        """Close file on deletion."""
        self.close()

    def close(self):
        """Close the HDF5 file."""
        try:
            self._file.close()
            self._file = None
        except Exception:
            pass

    def __getitem__(self, index: int):
        """Access item by index and load into memory.

        Parameters
        ----------
        index : int
            Index of item to access.

        Returns
        -------
        np.ndarray, dict, or object
            The loaded object.
        """
        if self._file is None:
            raise RuntimeError("HDF5Loader has been closed")

        if index < 0:
            index = self._num_items + index

        if index < 0 or index >= self._num_items:
            raise IndexError(f"Index {index} out of range for {self._num_items} items")

        item = self._file[f"item_{index}"]
        if isinstance(item, h5py.Group):
            item_type = item.attrs.get("type", "unknown")

            if item_type == "sparse_coordinate":
                shape = tuple(item.attrs["shape"])
                dtype = np.dtype(item.attrs["dtype"])
                result = np.zeros(shape, dtype=dtype)

                indices = item["indices"][:]
                values = item["values"][:]
                result[tuple(indices)] = values
                return result
            elif item_type == "pickle":
                pickled_data = item["data"][()].tobytes()
                return pickle.loads(pickled_data)
            elif item_type == "memmap":
                shape = tuple(item.attrs["shape"])
                dtype = np.dtype(item.attrs["dtype"])
                memmap_filename = item.attrs["filename"]
                return np.memmap(memmap_filename, dtype=dtype, shape=shape, mode="r")
        return item

    def __len__(self):
        """Get number of items in file."""
        return self._num_items

    def __iter__(self):
        """Iterator through items in file."""
        yield from (self[i] for i in range(len(self)))


def is_gzipped(filename: str) -> bool:
    """Check if a file is a gzip file by reading its magic number."""
    with open(filename, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def _stage_memmap(src: str, dst: str) -> None:
    """
    Relocate a memmap backing file to ``dst`` without invalidating any mmap the
    caller still holds on ``src``. An atomic rename is attempted first; on
    Windows this fails while the source is mapped, so a plain copy is used and
    the source is left in place for the caller to manage.
    """
    try:
        replace(src, dst)
    except OSError:
        copyfile(src, dst)


def write_pickle(data: object, filename: str, **kwargs) -> None:
    """
    Write data to a pickle file.

    Parameters
    ----------
    data : iterable or object
        The data to be serialized.
    filename : str
        The name of the file where the serialized data will be written.
        If filename ends with .gz, the output will be gzipped.

    Notes
    -----
    For numpy memmaps, only metadata is stored. The memmap file path is saved
    as a reference rather than copying the potentially huge data.
    """
    open_func = gzip_open if filename.endswith(".gz") else open
    with ThreadPoolExecutor() as executor:
        memmap_replacements = {}
        move_futures = []

        for i, item in enumerate(data):
            if isinstance(item, np.memmap):
                new_filename = generate_tempfile_name(
                    suffix=".mm", tmpdir=dirname(filename)
                )
                future = executor.submit(_stage_memmap, item.filename, new_filename)
                move_futures.append(future)
                memmap_replacements[i] = (
                    "np.memmap",
                    item.shape,
                    item.dtype,
                    new_filename,
                )

        for future in move_futures:
            future.result()

    with open_func(filename, "wb") as ofile:
        for i, item in enumerate(data):
            item = memmap_replacements.get(i, item)
            pickle.dump(item, ofile)


def load_pickle(filename: str) -> object:
    """
    Load data written by :py:meth:`write_pickle`.

    Parameters
    ----------
    filename : str
        The name of the file to read and deserialize data from.

    Returns
    -------
    object or iterable
        The deserialized data.
    """

    def _load_pickle(file_handle):
        try:
            while True:
                yield pickle.load(file_handle)
        except EOFError:
            pass

    def _is_pickle_memmap(data):
        ret = False
        if isinstance(data[0], str):
            if data[0] == "np.memmap":
                ret = True
        return ret

    items = []
    func = open if not is_gzipped(filename) else gzip_open

    with func(filename, "rb") as ifile:
        for data in _load_pickle(ifile):
            if isinstance(data, tuple):
                if _is_pickle_memmap(data):
                    _, shape, dtype, filename = data
                    data = np.memmap(filename, shape=shape, dtype=dtype)
            items.append(data)
    return items[0] if len(items) == 1 else items


def write_hdf5(
    data: Union[object, List, tuple, HDF5Loader],
    filename: str,
    compression: Optional[str] = "lzf",
    sparsity_threshold: float = 0.7,
    **kwargs,
) -> None:
    """
    Write data to HDF5 file.

    Parameters
    ----------
    data : object, list, or tuple
        Data to serialize. Can be numpy arrays, dicts, or pickleable objects.
    filename : str
        Output filename. Should end with .h5 or .hdf5.
    compression : str, optional
        Compression algorithm: 'gzip', 'lzf', or None. Default is 'lzf'.
    sparsity_threshold : float, optional
        If an array has more than this fraction of zeros, store in sparse coordinate
        format. Default is 0.7 (70% zeros).

    Notes
    -----
    For numpy memmaps, only metadata is stored. The memmap file path is saved
    as a reference rather than copying the potentially huge data.
    """
    compression_kwargs = {"compression": compression, "chunks": True}
    with h5py.File(filename, "w") as f:
        for i, item in enumerate(data):
            key = f"item_{i}"

            sparse = False
            if isinstance(item, np.ndarray):
                sparse = ((item == 0).sum() / item.size) > sparsity_threshold

            # As for pickle, we store memmap metadata as reference, NOT the data
            if isinstance(item, np.memmap):
                grp = f.create_group(key)
                grp.attrs["type"] = "memmap"
                grp.attrs["dtype"] = str(item.dtype)
                grp.attrs["shape"] = item.shape
                grp.attrs["filename"] = item.filename
            elif isinstance(item, np.ndarray) and not sparse:
                f.create_dataset(key, data=item, **compression_kwargs)
            elif isinstance(item, np.ndarray) and sparse:
                grp = f.create_group(key)
                grp.attrs["type"] = "sparse_coordinate"
                grp.attrs["dtype"] = str(item.dtype)
                grp.attrs["shape"] = item.shape

                # Non-flat indices seem to compress better
                nonzero_indices = np.where(item != 0)
                nonzero_values = item[nonzero_indices]

                max_index = max(item.shape)
                if max_index < 256:
                    index_dtype = np.uint8
                elif max_index < 65536:
                    index_dtype = np.uint16
                else:
                    index_dtype = np.uint32

                grp.create_dataset(
                    "indices",
                    data=np.array(nonzero_indices, dtype=index_dtype),
                    **compression_kwargs,
                )
                grp.create_dataset(
                    "values",
                    data=nonzero_values,
                    **compression_kwargs,
                )
            else:
                grp = f.create_group(key)
                grp.attrs["type"] = "pickle"
                pickled_data = pickle.dumps(item)
                grp.create_dataset(
                    "data",
                    data=np.void(pickled_data),
                    compression=None,
                )
        f.attrs["num_items"] = len(data)


def load_hdf5(filename: str, lazy: bool = False) -> HDF5Loader:
    """
    Load data written by :py:meth:`write_hdf5`.

    Parameters
    ----------
    filename : str
        Path to HDF5 file.

    Returns
    -------
    HDF5Loader
        HDF5Loader context manager.
    """
    return HDF5Loader(filename)


def _check_extension(filename: str, file_format: str) -> None:
    """Validate that filename extension matches the specified format.

    Parameters
    ----------
    filename : str
        The filename to validate.
    file_format : str
        The format ('hdf5', 'pickle' or 'auto'.).

    Raises
    ------
    ValueError
        If file_format is not supported or does not support the extension.
    """
    _extension = {
        "hdf5": (".h5", ".hdf5"),
        "pickle": (".pickle", ".pickle.gz"),
    }
    _extension["auto"] = tuple(ext for exts in _extension.values() for ext in exts)

    extensions = _extension.get(file_format)
    if extensions is None:
        _supported = ", ".join([str(x) for x in _extension.keys()])
        raise ValueError(f"Supported formats are {_supported}, got: {file_format}")

    valid = any([filename.endswith(x) for x in extensions])
    if not valid:
        _supported = ", ".join([str(x) for x in extensions])
        raise ValueError(f"{file_format} requires {_supported}, got: {filename}.")


def serialize(
    data: Union[object, List, tuple], filename: str, file_format: str = "auto", **kwargs
) -> None:
    """
    Serialize data to file.

    Parameters
    ----------
    data : object, list, or tuple
        Data to serialize.
    filename : str
        Output filename.
    format : {'pickle', 'hdf5', 'auto'}
        File format. Extension of filename determines format for 'auto'.
    **kwargs
        Keyword arguments passed to writer function.

    Notes
    -----
    Passing a filename with extension 'pickle.gz' will create a gzipped pickle.
    """
    _check_extension(filename, file_format)
    if file_format == "auto" and filename.endswith((".h5", ".hdf5")):
        file_format = "hdf5"

    func = write_pickle
    if file_format == "hdf5":
        func = write_hdf5

    if dir_name := dirname(filename):
        makedirs(dir_name, exist_ok=True)

    if not isinstance(data, (list, tuple)):
        data = (data,)
    return func(data, filename, **kwargs)


def deserialize(
    filename: str, file_format: str = "auto", **kwargs
) -> Union[List, Any, HDF5Loader]:
    """
    Deserialize data from file.

    Parameters
    ----------
    filename : str
        Input filename.
    format : {'pickle', 'hdf5', 'auto'}
        File format. Extension of filename determines format for 'auto'.
    **kwargs
        Keyword arguments passed to loader function.

    Returns
    -------
    list, object, or HDF5Loader
        Deserialized data or lazy loader (for HDF5 with lazy=True).
    """
    _check_extension(filename, file_format)
    if file_format == "auto" and filename.endswith((".h5", ".hdf5")):
        file_format = "hdf5"

    func = load_pickle
    if file_format == "hdf5":
        func = load_hdf5
    return func(filename, **kwargs)
