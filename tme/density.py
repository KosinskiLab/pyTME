""" Representation of N-dimensional densities

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import warnings
from io import BytesIO
from copy import deepcopy
from gzip import open as gzip_open
from typing import Tuple, Dict, Set
from os.path import splitext, basename

import h5py
import mrcfile
import numpy as np
import skimage.io as skio

from scipy.ndimage import (
    zoom,
    laplace,
    sobel,
    minimum_filter,
    binary_erosion,
    generic_gradient_magnitude,
)
from scipy.spatial import ConvexHull

from .types import NDArray
from .backends import NumpyFFTWBackend
from .structure import Structure
from .matching_utils import (
    array_to_memmap,
    memmap_to_array,
    minimum_enclosing_box,
)


class Density:
    """
    Abstract representation of N-dimensional densities.

    Parameters
    ----------
    data : array_like
        Array of densities.
    origin : array_like, optional
        Origin of the coordinate system, zero by default.
    sampling_rate : array_like, optional
        Sampling rate along data axis, one by default.
    metadata : dict, optional
        Dictionary with metadata information, empty by default.

    Raises
    ------
    ValueError
        If metadata is not a dictionary.

        If sampling rate / origin is not defined for a single or all axes.

    Examples
    --------
    The following achieves the minimal definition of a :py:class:`Density` instance

    >>> import numpy as np
    >>> from tme import Density
    >>> data = np.random.rand(50,70,40)
    >>> Density(data=data)

    Optional parameters ``origin`` correspond to the coordinate system reference,
    ``sampling_rate`` to the spatial length per axis element, and ``metadata`` to
    a dictionary with supplementary information. By default,
    :py:attr:`Density.origin` is set to zero, :py:attr:`Density.sampling_rate`
    to one, and :py:attr:`Density.metadata` is an empty dictionary. If provided,
    ``origin`` and ``sampling_rate`` either need to be a single value

    >>> Density(data=data, origin=0, sampling_rate=1)

    be specified along each data axis

    >>> Density(data=data, origin=(0, 0, 0), sampling_rate=(1.5, 1.1, 1.2))

    or be a combination of both

    >>> Density(data=data, origin=0, sampling_rate=(1.5, 1.1, 1.2))
    """

    def __init__(
        self,
        data: NDArray,
        origin: NDArray = None,
        sampling_rate: NDArray = None,
        metadata: Dict = {},
    ):
        origin = np.zeros(data.ndim) if origin is None else origin
        sampling_rate = 1 if sampling_rate is None else sampling_rate
        origin, sampling_rate = np.asarray(origin), np.asarray(sampling_rate)
        origin = np.repeat(origin, data.ndim // origin.size)
        sampling_rate = np.repeat(sampling_rate, data.ndim // sampling_rate.size)

        if sampling_rate.size != data.ndim:
            raise ValueError(
                "sampling_rate size should be 1 or "
                f"{data.ndim}, not {sampling_rate.size}."
            )
        if origin.size != data.ndim:
            raise ValueError(f"Expected origin size : {data.ndim}, got {origin.size}.")
        if not isinstance(metadata, dict):
            raise ValueError("Argument metadata has to be of class dict.")

        self.data, self.origin, self.sampling_rate = data, origin, sampling_rate
        self.metadata = metadata

    def __repr__(self):
        response = "Density object at {}\nOrigin: {}, Sampling Rate: {}, Shape: {}"
        return response.format(
            hex(id(self)),
            tuple(round(float(x), 3) for x in self.origin),
            tuple(round(float(x), 3) for x in self.sampling_rate),
            self.shape,
        )

    @classmethod
    def from_file(
        cls, filename: str, subset: Tuple[slice] = None, use_memmap: bool = False
    ) -> "Density":
        """
        Reads a file into a :py:class:`Density` instance.

        Parameters
        ----------
        filename : str
            Path to a file in CCP4/MRC, EM, HDF5 or a format supported by
            :obj:`skimage.io.imread`. The file can be gzip compressed.
        subset : tuple of slices, optional
            Slices representing the desired subset along each dimension.
        use_memmap : bool, optional
            Memory map the data contained in ``filename`` to save memory.

        Returns
        -------
        :py:class:`Density`
            Class instance representing the data in ``filename``.

        References
        ----------
        .. [1] Burnley T et al., Acta Cryst. D, 2017
        .. [2] Nickell S. et al, Journal of Structural Biology, 2005.
        .. [3] https://scikit-image.org/docs/stable/api/skimage.io.html

        Examples
        --------
        :py:meth:`Density.from_file` reads files in CCP4/MRC, EM, or a format supported
        by skimage.io.imread and converts them into a :py:class:`Density` instance. The
        following outlines how to read a file in the CCP4/MRC format [1]_:

        >>> from tme import Density
        >>> Density.from_file("/path/to/file.mrc")

        In some cases, you might want to read only a specific subset of the data.
        This can be achieved by passing a tuple of slices to the ``subset`` parameter.
        For example, to read only the first 50 voxels along each dimension:

        >>> subset_slices = (slice(0, 50), slice(0, 50), slice(0, 50))
        >>> Density.from_file("/path/to/file.mrc", subset=subset_slices)

        Memory mapping can be used to read the file from disk without loading
        it entirely into memory. This is particularly useful for large datasets
        or when working with limited memory resources:

        >>> Density.from_file("/path/to/large_file.mrc", use_memmap=True)

        Note that use_memmap will be ignored if the file is gzip compressed.

        If the input file has an `.em` or `.em.gz` extension, it will automatically
        be parsed as EM file [2]_.

        >>> Density.from_file("/path/to/file.em")
        >>> Density.from_file("/path/to/file.em.gz")

        If the file format is not CCP4/MRC or EM, :py:meth:`Density.from_file` attempts
        to use :obj:`skimage.io.imread` to read the file [3]_, which does not extract
        origin or sampling_rate information from the file:

        >>> Density.from_file("/path/to/other_format.tif")

        Notes
        -----
        If ``filename`` ends ".em" or ".h5" it will be parsed as EM or HDF5 file.
        Otherwise, the default reader is CCP4/MRC and on failure
        :obj:`skimage.io.imread` is used regardless of extension. The later does
        not extract origin or sampling_rate information from the file.

        See Also
        --------
        :py:meth:`Density.to_file`

        """
        try:
            func = cls._load_mrc
            if filename.endswith("em") or filename.endswith("em.gz"):
                func = cls._load_em
            elif filename.endswith("h5") or filename.endswith("h5.gz"):
                func = cls._load_hdf5
            data, origin, sampling_rate, meta = func(
                filename=filename, subset=subset, use_memmap=use_memmap
            )
        except ValueError:
            data, origin, sampling_rate, meta = cls._load_skio(filename=filename)
            if subset is not None:
                cls._validate_slices(slices=subset, shape=data.shape)
                data = data[subset].copy()

        return cls(data=data, origin=origin, sampling_rate=sampling_rate, metadata=meta)

    @classmethod
    def _load_mrc(
        cls, filename: str, subset: Tuple[int] = None, use_memmap: bool = False
    ) -> Tuple[NDArray, NDArray, NDArray, Dict]:
        """
        Extracts data from a CCP4/MRC file.

        Parameters
        ----------
        filename : str
            Path to a file in CCP4/MRC format.
        subset : tuple of slices, optional
            Slices representing the desired subset along each dimension.
        use_memmap : bool, optional
            Whether the Density objects data attribute should be memmory mapped.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, Dict]
            File data, coordinate origin, sampling rate array and metadata dictionary.

        References
        ----------
        .. [1] Burnley T, Palmer C & Winn M (2017) Recent developments in the
            CCP-EM software suite. Acta Cryst. D73:469–477.
            doi: 10.1107/S2059798317007859

        Raises
        ------
        ValueError
            If the mrcfile is malformatted.
            If the subset starts below zero, exceeds the data dimension or does not
            have the same length as the data dimensions.

        See Also
        --------
        :py:meth:`Density.from_file`

        """
        with mrcfile.open(filename, header_only=True) as mrc:
            data_shape = mrc.header.nz, mrc.header.ny, mrc.header.nx
            data_type = mrcfile.utils.data_dtype_from_header(mrc.header)

            # All map related parameters should be in zyx order
            origin = (
                mrc.header["origin"]
                .astype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
                .view(("<f4", 3))
            )
            origin = origin[::-1]

            # nx := column; ny := row; nz := section
            start = np.array(
                [
                    mrc.header["nzstart"],
                    mrc.header["nystart"],
                    mrc.header["nxstart"],
                ]
            )

            crs_index = (
                np.array(
                    [
                        int(mrc.header["mapc"]),
                        int(mrc.header["mapr"]),
                        int(mrc.header["maps"]),
                    ]
                )
                - 1
            )

            # mapc := column; mapr := row; maps := section;
            if not (0 in crs_index and 1 in crs_index and 2 in crs_index):
                raise ValueError(f"Malformatted CRS array in {filename}")

            sampling_rate = mrc.voxel_size.astype(
                [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
            ).view(("<f4", 3))
            sampling_rate = np.array(sampling_rate)[::-1]

            if np.allclose(origin, 0) and not np.allclose(start, 0):
                origin = np.multiply(start, sampling_rate)

            extended_header = mrc.header.nsymbt

            metadata = {
                "min": float(mrc.header.dmin),
                "max": float(mrc.header.dmax),
                "mean": float(mrc.header.dmean),
                "std": float(mrc.header.rms),
            }

        non_standard_crs = not np.all(crs_index == (0, 1, 2))
        if non_standard_crs:
            warnings.warn("Non standard MAPC, MAPR, MAPS, adapting data and origin.")

        if is_gzipped(filename):
            if use_memmap:
                warnings.warn(
                    f"Cannot open gzipped file {filename} as memmap."
                    f" Please run 'gunzip {filename}' to use memmap functionality."
                )
            use_memmap = False

        if subset is not None:
            subset = tuple(
                subset[i] if i < len(subset) else slice(0, data_shape[i])
                for i in crs_index
            )
            subset_shape = tuple(x.stop - x.start for x in subset)
            if np.allclose(subset_shape, data_shape):
                return cls._load_mrc(
                    filename=filename, subset=None, use_memmap=use_memmap
                )

            data = cls._read_binary_subset(
                filename=filename,
                slices=subset,
                data_shape=data_shape,
                dtype=data_type,
                header_size=1024 + extended_header,
            )
        elif subset is None and not use_memmap:
            with mrcfile.open(filename, header_only=False) as mrc:
                data = mrc.data.astype(np.float32, copy=False)
        else:
            with mrcfile.mrcmemmap.MrcMemmap(filename, header_only=False) as mrc:
                data = mrc.data

        if non_standard_crs:
            data = np.transpose(data, crs_index)
            origin = np.take(origin, crs_index)

        return data, origin, sampling_rate, metadata

    @classmethod
    def _load_em(
        cls, filename: str, subset: Tuple[int] = None, use_memmap: bool = False
    ) -> Tuple[NDArray, NDArray, NDArray, Dict]:
        """
        Extracts data from a EM file.

        Parameters
        ----------
        filename : str
            Path to a file in EM format.
        subset : tuple of slices, optional
            Slices representing the desired subset along each dimension.
        use_memmap : bool, optional
            Whether the Density objects data attribute should be memmory mapped.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, Dict]
            File data, coordinate origin, sampling rate array and metadata dictionary.

        References
        ----------
        .. [1] Nickell S. et al, Journal of Structural Biology, 2005.

        Warns
        -----
            If the sampling rate is zero.

        Notes
        -----
        A sampling rate of zero will be treated as missing value and changed to one. This
        function does not yet extract an origin like :py:meth:`Density._load_mrc`.

        See Also
        --------
        :py:meth:`Density.from_file`
        """
        DATA_TYPE_CODING = {
            1: np.byte,
            2: np.int16,
            3: np.int32,
            5: np.float32,
            6: np.float64,
            8: np.complex64,
            9: np.complex128,
        }

        func = gzip_open if is_gzipped(filename) else open
        with func(filename, mode="rb") as f:
            if is_gzipped(filename):
                f = BytesIO(f.read())

            f.seek(3, 1)
            data_type_code = np.frombuffer(f.read(1), dtype="<i1")[0]
            data_type = DATA_TYPE_CODING.get(data_type_code)

            data_shape = np.frombuffer(f.read(3 * 4), dtype="<i4")[::-1]

            f.seek(80, 1)
            user_params = np.frombuffer(f.read(40 * 4), dtype="<i4")

            pixel_size = user_params[6] / 1000.0
            f.seek(256, 1)

            if use_memmap and subset is None:
                data = np.memmap(f, dtype=data_type, mode="r", offset=f.tell()).reshape(
                    data_shape
                )
            elif subset is None:
                data_size = np.prod(data_shape) * np.dtype(data_type).itemsize
                data = np.frombuffer(f.read(data_size), dtype=data_type).reshape(
                    data_shape
                )
                data = data.astype(np.float32)
            else:
                subset_shape = [x.stop - x.start for x in subset]
                if np.allclose(subset_shape, data_shape):
                    return cls._load_em(
                        filename=filename, subset=None, use_memmap=use_memmap
                    )

                data = cls._read_binary_subset(
                    filename=filename,
                    slices=subset,
                    data_shape=data_shape,
                    dtype=data_type(),
                    header_size=f.tell(),
                )

        origin = np.zeros(3, dtype=data.dtype)

        if pixel_size == 0:
            warnings.warn(
                f"Got invalid sampling rate {pixel_size}, overwriting it to 1."
            )
            pixel_size = 1
        sampling_rate = np.repeat(pixel_size, data.ndim).astype(data.dtype)

        return data, origin, sampling_rate, {}

    @staticmethod
    def _validate_slices(slices: Tuple[slice], shape: Tuple[int]):
        """
        Validate whether the given slices fit within the provided data shape.

        Parameters
        ----------
        slices : Tuple[slice]
            A tuple of slice objects, one per dimension of the data.
        shape : Tuple[int]
            The shape of the data being sliced, as a tuple of integers.

        Raises
        ------
        ValueError
            - If the length of `slices` doesn't match the dimension of shape.
            - If any slice has a stop value exceeding any dimension in shape.
            - If any slice has a stop value that is negative.
        """

        n_dims = len(shape)
        if len(slices) != n_dims:
            raise ValueError(
                f"Expected length of slices : {n_dims}, got : {len(slices)}"
            )

        if any(
            [
                slices[i].stop > shape[i] or slices[i].start > shape[i]
                for i in range(n_dims)
            ]
        ):
            raise ValueError(f"Subset exceeds data dimensions ({shape}).")

        if any([slices[i].stop < 0 or slices[i].start < 0 for i in range(n_dims)]):
            raise ValueError("Subsets have to be non-negative.")

    @classmethod
    def _read_binary_subset(
        cls,
        filename: str,
        slices: Tuple[slice],
        data_shape: Tuple[int],
        dtype: type,
        header_size: int,
    ) -> NDArray:
        """
        Read a subset of data from a binary file with a header.

        Parameters
        ----------
        filename : str
            Path to the binary file.
        slices : tuple of slice objects
            Slices representing the desired subset in each dimension.
        data_shape : tuple of ints
            Shape of the complete dataset in the file.
        dtype : numpy dtype
            Data type of the dataset in the file.
        header_size : int
            Size of the file's header in bytes.

        Returns
        -------
        NDArray
            Subset of the dataset as specified by the slices.

        Raises
        ------
        NotImplementedError
            If the data is not three dimensional.

        See Also
        --------
        :py:meth:`Density._load_mrc`
        :py:meth:`Density._load_em`
        """
        n_dims = len(data_shape)
        if n_dims != 3:
            raise NotImplementedError("Only 3-dimensional data can be subsetted.")

        cls._validate_slices(slices=slices, shape=data_shape)
        bytes_per_item = dtype.itemsize

        subset_shape = [s.stop - s.start for s in slices]
        subset_data = np.empty(subset_shape, dtype=dtype)

        row_bytes = (slices[2].stop - slices[2].start) * bytes_per_item
        full_row_bytes = data_shape[2] * bytes_per_item
        x_offset = slices[2].start * bytes_per_item

        func = gzip_open if is_gzipped(filename) else open
        with func(filename, mode="rb") as f:
            if is_gzipped(filename):
                f = BytesIO(f.read())

            for z in range(slices[0].start, slices[0].stop):
                base_offset_z = header_size + z * data_shape[1] * full_row_bytes

                for y in range(slices[1].start, slices[1].stop):
                    offset = base_offset_z + y * full_row_bytes + x_offset
                    f.seek(offset)
                    row = np.frombuffer(f.read(row_bytes), dtype=dtype)
                    subset_data[z - slices[0].start, y - slices[1].start] = row

        return subset_data

    @staticmethod
    def _load_skio(filename: str) -> Tuple[NDArray, NDArray, NDArray, Dict]:
        """
        Uses :obj:`skimage.io.imread` to extract data from filename [1]_.

        Parameters
        ----------
        filename : str
            Path to a file whose format is supported by :obj:`skimage.io.imread`.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, Dict]
            File data, coordinate origin, sampling rate array and metadata dictionary.

        References
        ----------
        .. [1] https://scikit-image.org/docs/stable/api/skimage.io.html

        Warns
        -----
            Warns that origin and sampling_rate are not yet extracted from ``filename``.

        See Also
        --------
        :py:meth:`Density.from_file`
        """
        swap = filename
        if is_gzipped(filename):
            with gzip_open(filename, "rb") as infile:
                swap = BytesIO(infile.read())

        data = skio.imread(swap)
        warnings.warn(
            "origin and sampling_rate are not yet extracted from non CCP4/MRC files."
        )
        return data, np.zeros(data.ndim), np.ones(data.ndim), {}

    @staticmethod
    def _load_hdf5(
        filename: str, subset: Tuple[slice], use_memmap: bool = False, **kwargs
    ) -> "Density":
        """
        Extracts data from an H5 file.

        Parameters
        ----------
        filename : str
            Path to a file in CCP4/MRC format.
        subset : tuple of slices, optional
            Slices representing the desired subset along each dimension.
        use_memmap : bool, optional
            Whether the Density objects data attribute should be memmory mapped.

        Returns
        -------
        Density
            An instance of the Density class populated with the data from the HDF5 file.

        See Also
        --------
        :py:meth:`Density._save_hdf5`
        """
        subset = ... if subset is None else subset

        with h5py.File(filename, mode="r") as infile:
            data = infile["data"]
            data_attributes = [
                infile["data"].id.get_offset(),
                infile["data"].shape,
                infile["data"].dtype,
            ]
            origin = infile["origin"][...].copy()
            sampling_rate = infile["sampling_rate"][...].copy()
            metadata = {key: val for key, val in infile.attrs.items()}
            if not use_memmap:
                return data[subset], origin, sampling_rate, metadata

        offset, shape, dtype = data_attributes
        data = np.memmap(filename, dtype=dtype, shape=shape, offset=offset)[subset]

        return data, origin, sampling_rate, metadata

    @classmethod
    def from_structure(
        cls,
        filename_or_structure: str,
        shape: Tuple[int] = None,
        sampling_rate: NDArray = np.ones(1),
        origin: Tuple[float] = None,
        weight_type: str = "atomic_weight",
        weight_type_args: Dict = {},
        chain: str = None,
        filter_by_elements: Set = None,
        filter_by_residues: Set = None,
    ) -> "Density":
        """
        Reads in an atomic structure and converts it into a :py:class:`Density`
        instance.

        Parameters
        ----------
        filename_or_structure : str or :py:class:`tme.structure.Structure`
            Either :py:class:`tme.structure.Structure` instance or path to
            structure file that can be read by
            :py:meth:`tme.structure.Structure.from_file`.
        shape : tuple of int, optional
            Shape of the new :py:class:`Density` instance. By default,
            computes the minimum 3D box holding all atoms.
        sampling_rate : float, optional
            Sampling rate of the output array along each axis, in the same unit
            as the atoms in the structure. Defaults to one Ångstroms
            per axis unit.
        origin : tuple of float, optional
            Origin of the coordinate system. If provided, its expected to be in
            z, y, x form in the same unit as the atoms in the structure.
            By default, computes origin as distance between minimal coordinate
            and coordinate system origin.
        weight_type : str, optional
            Which weight should be given to individual atoms. For valid values
            see :py:meth:`tme.structure.Structure.to_volume`.
        weight_type_args : dict, optional
            Additional arguments for atom weight computation.
        chain : str, optional
            The chain that should be extracted from the structure. If multiple chains
            should be selected, they needto be a comma separated string,
            e.g. 'A,B,CE'. If chain None, all chains are returned. Default is None.
        filter_by_elements : set, optional
            Set of atomic elements to keep. Default is all atoms.
        filter_by_residues : set, optional
            Set of residues to keep. Default is all residues.

        Returns
        -------
        :py:class:`Density`
            Newly created :py:class:`Density` instance.

        References
        ----------
        .. [1]  Sorzano, Carlos et al (Mar. 2015). Fast and accurate conversion
            of atomic models into electron density maps. AIMS Biophysics
            2, 8–20.

        Examples
        --------
        The following outlines the minimal parameters needed to read in an
        atomic structure and convert it into a :py:class:`Density` instance. For
        specification on supported formats refer to
        :py:meth:`tme.structure.Structure.from_file`.

        >>> path_to_structure = "/path/to/structure.cif"
        >>> density = Density.from_structure(path_to_structure)

        :py:meth:`Density.from_structure` will automatically determine the appropriate
        density dimensions based on the structure. The origin will be computed as
        minimal distance required to move the closest atom of the structure to the
        coordinate system origin. Furthermore, all chains will be used and the atom
        densities will be represented by their atomic weight and accumulated
        on a per-voxel basis.

        The following will read in chain A of an atomic structure and discretize
        it on a grid of dimension 100 x 100 x 100 using a sampling rate of
        2.5 Angstrom per voxel.

        >>> density = Density.from_structure(
        >>>    filename_or_structure = path_to_structure,
        >>>    shape = (100, 100, 100),
        >>>    sampling_rate = 2.5,
        >>>    chain = "A"
        >>> )

        We can restrict the generated :py:class:`Density` instance to only contain
        specific elements like carbon and nitrogen:

        >>> density = Density.from_structure(
        >>>    filename_or_structure = path_to_structure,
        >>>    filter_by_elements = {"C", "N"}
        >>> )

        or specified residues such as polar amino acids:

        >>> density = Density.from_structure(
        >>>    filename_or_structure = path_to_structure,
        >>>    filter_by_residues = {"SER", "THR", "CYS", "ASN", "GLN", "TYR"}
        >>> )

        :py:meth:`Density.from_structure` supports a variety of methods to convert
        atoms into densities. In addition to 'atomic_weight', 'atomic_number',
        and 'van_der_waals_radius', its possible to use experimentally determined
        scattering factors from various sources:

        >>> density = Density.from_structure(
        >>>    filename_or_structure = path_to_structure,
        >>>    weight_type = "scattering_factors",
        >>>    weight_type_args={"source": "dt1969"}
        >>> )

        or a lowpass filtered representation introduced in [1]_:

        >>> density = Density.from_structure(
        >>>    filename_or_structure = path_to_structure,
        >>>    weight_type = "lowpass_scattering_factors",
        >>>    weight_type_args={"source": "dt1969"}
        >>> )

        See Also
        --------
        :py:meth:`tme.structure.Structure.from_file`
        :py:meth:`tme.structure.Structure.to_volume`
        """
        structure = filename_or_structure
        if isinstance(filename_or_structure, str):
            structure = Structure.from_file(
                filename=filename_or_structure,
                filter_by_elements=filter_by_elements,
                filter_by_residues=filter_by_residues,
            )

        volume, origin, sampling_rate = structure.to_volume(
            shape=shape,
            sampling_rate=sampling_rate,
            origin=origin,
            chain=chain,
            weight_type=weight_type,
            weight_type_args=weight_type_args,
        )

        return cls(
            data=volume,
            origin=origin,
            sampling_rate=sampling_rate,
            metadata=structure.metadata.copy(),
        )

    def to_file(self, filename: str, gzip: bool = False) -> None:
        """
        Writes class instance to disk.

        Parameters
        ----------
        filename : str
            Path to write to.
        gzip : bool, optional
            Gzip compress the output and add corresponding suffix to filename
            if not present. False by default.

        References
        ----------
        .. [1] Burnley T et al., Acta Cryst. D, 2017
        .. [2] Nickell S. et al, Journal of Structural Biology, 2005
        .. [3] https://scikit-image.org/docs/stable/api/skimage.io.html

        Examples
        --------
        The following creates a :py:class:`Density` instance `dens` holding
        random data values and writes it to disk:

        >>> import numpy as np
        >>> from tme import Density
        >>> data = np.random.rand(50,50,50)
        >>> dens = Density(data=data, origin=(0, 0, 0), sampling_rate=(1, 1, 1))
        >>> dens.to_file("example.mrc")

        The output file can also be directly ``gzip`` compressed. The corresponding
        ".gz" extension will be automatically added if absent [1]_.

        >>> dens.to_file("example.mrc", gzip=True)

        The :py:meth:`Density.to_file` method also supports writing EM files [2]_:

        >>> dens.to_file("example.em")

        In addition, a variety of image file formats are supported [3]_:

        >>> data = np.random.rand(50,50)
        >>> dens = Density(data=data, origin=(0, 0), sampling_rate=(1, 1))
        >>> dens.to_file("example.tiff")

        Notes
        -----
        If ``filename`` endswith ".em" or ".h5" a EM file or HDF5 file will be created.
        The default output format is CCP4/MRC and on failure, :obj:`skimage.io.imsave`
        is used.

        See Also
        --------
        :py:meth:`Density.from_file`
        """
        if gzip:
            filename = filename if filename.endswith(".gz") else f"{filename}.gz"

        try:
            func = self._save_mrc
            if filename.endswith("em") or filename.endswith("em.gz"):
                func = self._save_em
            elif filename.endswith("h5") or filename.endswith("h5.gz"):
                func = self._save_hdf5
            _ = func(filename=filename, gzip=gzip)
        except ValueError:
            _ = self._save_skio(filename=filename, gzip=gzip)

    def _save_mrc(self, filename: str, gzip: bool = False) -> None:
        """
        Writes class instance to disk as mrc file.

        Parameters
        ----------
        filename : str
            Path to write to.
        gzip : bool, optional
            If True, the output will be gzip compressed.

        References
        ----------
        .. [1] Burnley T et al., Acta Cryst. D, 2017
        """
        compression = "gzip" if gzip else None
        with mrcfile.new(filename, overwrite=True, compression=compression) as mrc:
            mrc.set_data(self.data.astype("float32"))
            mrc.header.nzstart, mrc.header.nystart, mrc.header.nxstart = np.rint(
                np.divide(self.origin, self.sampling_rate)
            )
            mrc.header.origin = tuple(x for x in self.origin)
            # mrcfile library expects origin to be in xyz format
            mrc.header.mapc, mrc.header.mapr, mrc.header.maps = (1, 2, 3)
            mrc.header["origin"] = tuple(self.origin[::-1])
            mrc.voxel_size = tuple(self.sampling_rate[::-1])

    def _save_em(self, filename: str, gzip: bool = False) -> None:
        """
        Writes data to disk as an .em file.

        Parameters
        ----------
        filename : str
            Path to write to.
        gzip : bool, optional
            If True, the output will be gzip compressed.

        References
        ----------
        .. [1] Nickell S. et al, Journal of Structural Biology, 2005.
        """
        DATA_TYPE_MAPPING = {
            np.dtype(np.int8): 1,
            np.dtype(np.int16): 2,
            np.dtype(np.int32): 3,
            np.dtype(np.float32): 5,
            np.dtype(np.float64): 6,
            np.dtype(np.complex64): 8,
            np.dtype(np.complex128): 9,
        }

        data_type_code = DATA_TYPE_MAPPING.get(self.data.dtype, 5)

        func = gzip_open if gzip else open
        with func(filename, "wb") as f:
            f.write(np.array([0], dtype=np.int8).tobytes())
            f.write(np.array([0, 0, data_type_code], dtype=np.int8).tobytes())
            f.write(np.array(self.data.shape, dtype="<i4").tobytes())
            f.write(b" " * 80)
            user_params = np.zeros(40, dtype="<i4")
            user_params[6] = int(self.sampling_rate[0] * 1000)
            f.write(user_params.tobytes())
            f.write(b" " * 256)
            f.write(self.data.tobytes())

    def _save_skio(self, filename: str, gzip: bool = False) -> None:
        """
        Uses :obj:`skimage.io.imsave` to write data to filename [1]_.

        Parameters
        ----------
        filename : str
            Path to write to with a format supported by :obj:`skimage.io.imsave`.
        gzip : bool, optional
            If True, the output will be gzip compressed.

        References
        ----------
        .. [1] https://scikit-image.org/docs/stable/api/skimage.io.html
        """
        swap, kwargs = filename, {}
        if gzip:
            swap = BytesIO()
            kwargs["format"] = splitext(basename(filename.replace(".gz", "")))[
                1
            ].replace(".", "")
        skio.imsave(fname=swap, arr=self.data.astype("float32"), **kwargs)
        if gzip:
            with gzip_open(filename, "wb") as outfile:
                outfile.write(swap.getvalue())

    def _save_hdf5(self, filename: str, gzip: bool = False) -> None:
        """
        Saves the Density instance data to an HDF5 file, with optional compression.

        Parameters
        ----------
        filename : str
            Path to write to.
        gzip : bool, optional
            If True, the output will be gzip compressed.

        See Also
        --------
        :py:meth:`Density._load_hdf5`
        """
        compression = "gzip" if gzip else None
        with h5py.File(filename, mode="w") as f:
            f.create_dataset(
                "data",
                data=self.data,
                shape=self.data.shape,
                dtype=self.data.dtype,
                compression=compression,
            )
            f.create_dataset("origin", data=self.origin)
            f.create_dataset("sampling_rate", data=self.sampling_rate)

            self.metadata["mean"] = self.metadata.get("mean", 0)
            self.metadata["std"] = self.metadata.get("std", 0)
            self.metadata["min"] = self.metadata.get("min", 0)
            self.metadata["max"] = self.metadata.get("max", 0)
            if not isinstance(self.data, np.memmap):
                self.metadata["mean"] = self.data.mean()
                self.metadata["std"] = self.data.std()
                self.metadata["min"] = self.data.min()
                self.metadata["max"] = self.data.max()

            for key, val in self.metadata.items():
                f.attrs[key] = val

    @property
    def empty(self) -> "Density":
        """
        Returns a copy of the class instance with all elements in
        :py:attr:`Density.data` set to zero. :py:attr:`Density.metadata` will be
        initialized accordingly. :py:attr:`Density.origin` and
        :py:attr:`Density.sampling_rate` are copied.

        Returns
        -------
        :py:class:`Density`
            Empty class instance.

        Examples
        --------
        >>> import numpy as np
        >>> from tme import Density
        >>> original_density = Density.from_file("/path/to/file.mrc")
        >>> empty_density = original_density.empty
        >>> np.all(empty_density.data == 0)
        True
        """
        return Density(
            data=np.zeros_like(self.data),
            origin=deepcopy(self.origin),
            sampling_rate=deepcopy(self.sampling_rate),
            metadata={"min": 0, "max": 0, "mean": 0, "std": 0},
        )

    def copy(self) -> "Density":
        """
        Create a copy of the class instance.

        Returns
        -------
        :py:class:`Density`
            A copy of the class instance.

        Examples
        --------
        >>> from tme import Density
        >>> original_density = Density.from_file("/path/to/file.mrc")
        >>> copied_density = original_density.copy
        >>> np.all(copied_density.data == original_density.data)
        True
        """
        return Density(
            data=self.data.copy(),
            origin=deepcopy(self.origin[:]),
            sampling_rate=self.sampling_rate,
            metadata=deepcopy(self.metadata),
        )

    def to_memmap(self) -> None:
        """
        Converts :py:attr:`Density.data` to a :obj:`numpy.memmap`.

        Examples
        --------
        The following outlines how to use the :py:meth:`Density.to_memmap` method.

        >>> from tme import Density
        >>> large_density = Density.from_file("/path/to/large_file.mrc")
        >>> large_density.to_memmap()

        A more efficient solution to achieve the result outlined above is to
        provide the ``use_memmap`` flag in :py:meth:`Density.from_file`.

        >>> Density.from_file("/path/to/large_file.mrc", use_memmap = True)

        In practice, the :py:meth:`Density.to_memmap` method finds application, if a
        large number of :py:class:`Density` instances need to be in memory at once,
        without occupying the full phyiscal memory required to store
        :py:attr:`Density.data`.


        See Also
        --------
        :py:meth:`Density.to_numpy`
        """
        if isinstance(self.data, np.memmap):
            return None

        filename = array_to_memmap(arr=self.data)

        self.data = np.memmap(
            filename, mode="r", dtype=self.data.dtype, shape=self.data.shape
        )

    def to_numpy(self) -> None:
        """
        Converts :py:attr:`Density.data` to an in-memory :obj:`numpy.ndarray`.

        Examples
        --------
        >>> from tme import Density
        >>> density = Density.from_file("/path/to/large_file.mrc")
        >>> density.to_memmap()  # Convert to memory-mapped array first
        >>> density.to_numpy()   # Now, convert back to an in-memory array

        See Also
        --------
        :py:meth:`Density.to_memmap`
        """
        self.data = memmap_to_array(self.data)

    @property
    def shape(self) -> Tuple[int]:
        """
        Returns the dimensions of :py:attr:`Density.data`.

        Returns
        -------
        tuple
            The dimensions of :py:attr:`Density.data`.

        Examples
        --------
        >>> import numpy as np
        >>> from tme import Density
        >>> dens = Density(np.array([0, 1, 1, 1, 0]))
        >>> dens.shape
        (5,)
        """
        return self.data.shape

    @property
    def data(self) -> NDArray:
        """
        Returns the value of :py:attr:`Density.data`.

        Returns
        -------
        NDArray
            Value of the instance's :py:attr:`Density.data` attribute.

        Examples
        --------
        The following outlines the usage of :py:attr:`Density.data`:

        >>> import numpy as np
        >>> from tme import Density
        >>> dens = Density(np.array([0, 1, 1, 1, 0]))
        >>> dens.data
        array([0, 1, 1, 1, 0])

        """
        return self._data

    @data.setter
    def data(self, data: NDArray) -> None:
        """
        Sets the value of the instance's :py:attr:`Density.data` attribute.
        """
        self._data = data

    @property
    def origin(self) -> NDArray:
        """
        Returns the value of the instance's :py:attr:`Density.origin`
        attribute.

        Returns
        -------
        NDArray
            Value of the instance's :py:attr:`Density.origin` attribute.

        Examples
        --------
        The following outlines the usage of :py:attr:`Density.origin`:

        >>> import numpy as np
        >>> from tme import Density
        >>> dens = Density(np.array([0, 1, 1, 1, 0]))
        >>> dens.origin
        array([0.])
        """
        return self._origin

    @origin.setter
    def origin(self, origin: NDArray) -> None:
        """
        Sets the origin of the class instance.
        """
        origin = np.asarray(origin)
        origin = np.repeat(origin, self.data.ndim // origin.size)
        self._origin = origin

    @property
    def sampling_rate(self) -> NDArray:
        """
        Returns the value of the instance's :py:attr:`Density.sampling_rate` attribute.

        Returns
        -------
        NDArray
            Sampling rate along axis.
        """
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate: NDArray) -> None:
        """
        Sets the sampling rate of the class instance.
        """
        sampling_rate = np.asarray(sampling_rate)
        sampling_rate = np.repeat(sampling_rate, self.data.ndim // sampling_rate.size)
        self._sampling_rate = sampling_rate

    @property
    def metadata(self) -> Dict:
        """
        Returns the instance's :py:attr:`Density.metadata` attribute.

        Returns
        -------
        Dict
            Metadata dictionary. Empty by default.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Dict) -> None:
        """
        Sets the metadata of the class instance.
        """
        self._metadata = metadata

    def to_pointcloud(self, threshold: float = 0) -> NDArray:
        """
        Returns data indices that are larger than the given threshold.

        Parameters
        ----------
        threshold : float, optional
            The cutoff value to determine the indices. Default is 0.

        Returns
        -------
        NDArray
            Data indices that are larger than the given threshold with shape
            (dimensions, indices).

        Examples
        --------
        >>> density.to_pointcloud(0)
        """
        return np.array(np.where(self.data > threshold))

    def _pad_slice(self, box: Tuple[slice], pad_kwargs: Dict = {}) -> NDArray:
        """
        Pads the internal data array according to box.

        Negative slices indices will result in a left-hand padding, while
        slice indices larger than the box_size property of the class
        instance will result in a right-hand padding.

        Parameters
        ----------
        box : tuple of slice
            Tuple of slice objects that define the box dimensions.
        pad_kwargs: dict, optional
            Parameter dictionary passed to numpy pad.

        Returns
        -------
        NDArray
            The padded internal data array.
        """
        box_start = np.array([b.start for b in box])
        box_stop = np.array([b.stop for b in box])
        left_pad = -np.minimum(box_start, np.zeros(len(box), dtype=int))

        right_pad = box_stop - box_start * (box_start > 0)
        right_pad -= np.array(self.shape, dtype=int)
        right_pad = np.maximum(right_pad, np.zeros_like(right_pad))
        padding = tuple((left, right) for left, right in zip(left_pad, right_pad))

        ret = np.pad(self.data, padding, **pad_kwargs)
        return ret

    def adjust_box(self, box: Tuple[slice], pad_kwargs: Dict = {}) -> None:
        """
        Adjusts :py:attr:`Density.data` and :py:attr:`Density.origin`
        according to the provided box.

        Parameters
        ----------
        box : tuple of slices
            Description of how each axis of :py:attr:`Density.data` should be sliced.
        pad_kwargs: dict, optional
            Parameter dictionary passed to :obj:`numpy.pad`.

        See Also
        --------
        :py:meth:`Density.trim_box`

        Examples
        --------
        The following demonstrates the ability of :py:meth:`Density.adjust_box`
        to extract a subdensity from the current :py:class:`Density` instance.
        :py:meth:`Density.adjust_box` not only operats on :py:attr:`Density.data`,
        but also modifies :py:attr:`Density.origin` according to ``box``.

        >>> import numpy as np
        >>> from tme import Density
        >>> dens = Density(np.ones((5, 5)))
        >>> box = (slice(1, 4), slice(2, 5))
        >>> dens.adjust_box(box)
        >>> dens
        Origin: (1.0, 2.0), sampling_rate: (1, 1), Shape: (3, 3)

        :py:meth:`Density.adjust_box` can also extend the box of the current
        :py:class:`Density` instance. This is achieved by negative start or
        stops that exceed the dimension of the current :py:attr:`Density.data` array.

        >>> box = (slice(-1, 10), slice(2, 10))
        >>> dens.adjust_box(box)
        >>> dens
        Origin: (0.0, 4.0), sampling_rate: (1, 1), Shape: (11, 8)

        However, do note that only the start coordinate of each slice in ``box``
        can be negative.

        >>> box = (slice(-1, 10), slice(2, -10))
        >>> dens.adjust_box(box)
        >>> dens
        Origin: (-1.0, 6.0), sampling_rate: (1, 1), Shape: (11, 0)
        """
        crop_box = tuple(
            slice(max(b.start, 0), min(b.stop, shape))
            for b, shape in zip(box, self.data.shape)
        )
        self.data = self.data[crop_box].copy()

        # In case the box is larger than the current map
        self.data = self._pad_slice(box, pad_kwargs=pad_kwargs)

        # Adjust the origin
        left_shift = np.array([-1 * box[i].start for i in range(len(box))])
        self.origin = self.origin - np.multiply(left_shift, self.sampling_rate)

    def trim_box(self, cutoff: float, margin: int = 0) -> Tuple[slice]:
        """
        Computes a rectangle with sufficient dimension that encloses all
        values of the internal data array larger than the specified cutoff,
        expanded by the specified margin.

        The output can be passed to :py:meth:`Density.adjust_box` to crop
        the internal data array.

        Parameters
        ----------
        cutoff : float
            The threshold value for determining the minimum enclosing box. Default is 0.
        margin : int, optional
            The margin to add to the box dimensions. Default is 0.

        Returns
        -------
        tuple
            A tuple containing slice objects representing the box.

        Raises
        ------
        ValueError
            If the cutoff is larger than or equal to the maximum density value.

        Examples
        --------
        The following will compute the bounding box that encloses all values
        in the example array that are larger than zero:

        >>> import numpy as np
        >>> from tme import Density
        >>> dens = Density(np.array([0,1,1,1,0]))
        >>> dens.trim_box(0)
        (slice(1, 4, None),)

        The resulting tuple can be passed to :py:meth:`Density.adjust_box` to trim the
        current :py:class:`Density` instance:

        >>> dens.adjust_box(dens.trim_box(0))
        >>> dens.data.shape
        (3,)

        See Also
        --------
        :py:meth:`Density.adjust_box`
        """
        if cutoff >= self.data.max():
            raise ValueError(
                f"Cutoff exceeds data range ({cutoff} >= {self.data.max()})."
            )
        starts, stops = [], []
        for axis in range(self.data.ndim):
            projected_max = np.max(
                self.data, axis=tuple(i for i in range(self.data.ndim) if i != axis)
            )
            valid = np.where(projected_max > cutoff)[0]
            starts.append(max(0, valid[0] - margin))
            stops.append(min(self.data.shape[axis], valid[-1] + margin + 1))
        slices = tuple(slice(*coord) for coord in zip(starts, stops))
        return slices

    def minimum_enclosing_box(
        self,
        cutoff: float,
        use_geometric_center: bool = False,
    ) -> Tuple[slice]:
        """
        Compute the enclosing box that holds all possible rotations of the internal
        data array.

        Parameters
        ----------
        cutoff : float
            Above this value arr elements are considered. Defaults to 0.
        use_geometric_center : bool, optional
            Whether the box should accommodate the geometric or the coordinate
            center. Defaults to False.

        Returns
        -------
        tuple
            Tuple of slices corresponding to the minimum enclosing box.

        See Also
        --------
        :py:meth:`Density.adjust_box`
        :py:meth:`tme.matching_utils.minimum_enclosing_box`
        """
        coordinates = self.to_pointcloud(threshold=cutoff)
        starts, stops = coordinates.min(axis=1), coordinates.max(axis=1)

        shape = minimum_enclosing_box(
            coordinates=coordinates,
            use_geometric_center=use_geometric_center,
        )
        difference = np.maximum(np.subtract(shape, np.subtract(stops, starts)), 0)

        shift_start = np.divide(difference, 2).astype(int)
        shift_stop = shift_start + np.mod(difference, 2)

        starts = (starts - shift_start).astype(int)
        stops = (stops + shift_stop).astype(int)

        enclosing_box = tuple(slice(start, stop) for start, stop in zip(starts, stops))

        return tuple(enclosing_box)

    def pad(
        self, new_shape: Tuple[int], center: bool = True, padding_value: float = 0
    ) -> None:
        """
        :py:meth:`Density.pad` extends the internal :py:attr:`Density.data`
        array of the current :py:class:`Density` instance to ``new_shape`` and
        adapts :py:attr:`Density.origin` accordingly.

        Parameters
        ----------
        new_shape : tuple of int
            The desired shape for the new volume.
        center : bool, optional
            Whether the data should be centered in the new box. Default is True.
        padding_value : float, optional
            Value to pad the data array with. Default is zero.

        Raises
        ------
        ValueError
            If the length of ``new_shape`` does not match the dimensionality of
            :py:attr:`Density.data`.

        Examples
        --------
        The following demonstrates the functionality of :py:meth:`Density.pad` on
        a one-dimensional array:

        >>> import numpy as np
        >>> from tme import Density
        >>> dens = Density(np.array([1,1,1]))
        >>> dens.pad(new_shape = (5,), center = True)
        >>> dens.data
        array([0, 1, 1, 1, 0])

        If ``center`` is set to False, the padding values will be appended:

        >>> dens = Density(np.array([1,1,1]))
        >>> dens.pad(new_shape = (5,), center = False)
        >>> dens.data
        array([1, 1, 1, 0, 0])

        It's also possible to pass a user-defined ``padding_value``:

        >>> dens = Density(np.array([1,1,1]))
        >>> dens.pad(new_shape = (5,), center = True, padding_value = -1)
        >>> dens.data
        array([-1, 1, 1, 1, -1])
        """
        if len(new_shape) != self.data.ndim:
            raise ValueError(
                f"new_shape has dimension {len(new_shape)}"
                f" but expected was {self.data.ndim}."
            )

        new_box = tuple(slice(0, stop) for stop in new_shape)
        if center:
            overhang = np.subtract(new_shape, self.shape).astype(int)
            padding = overhang // 2
            left = -padding
            right = np.add(self.shape, padding + overhang % 2)
            new_box = tuple(slice(*box) for box in zip(left, right))

        self.adjust_box(new_box, pad_kwargs={"constant_values": padding_value})

    def centered(self, cutoff: float = 0) -> Tuple["Density", NDArray]:
        """
        Shifts the data center of mass to the center of the data array using linear
        interpolation. The box size of the returned :py:class:`Density` object is at
        least equal to the box size of the class instance.

        Parameters
        ----------
        cutoff : float, optional
            Only elements in data larger than cutoff will be considered for
            computing the new box. By default considers only positive elements.

        Notes
        -----
        Should any axis of the class instance data array be smaller than the return
        value of :py:meth:`Density.minimum_enclosing_box`, the size of the internal
        data array is adapted to avoid array elements larger than cutoff to fall
        outside the data array.

        Returns
        -------
        :py:class:`Density`
            A centered copy of the class instance.
        NDArray
            The offset between array center and center of mass.

        See Also
        --------
        :py:meth:`Density.trim_box`
        :py:meth:`Density.minimum_enclosing_box`

        Examples
        --------
        :py:meth:`Density.centered` returns a tuple containing a centered version
        of the current :py:class:`Density` instance, as well as an array with
        translations. The translation corresponds to the shift between the original and
        current center of mass.

        >>> import numpy as np
        >>> from tme import Density
        >>> dens = Density(np.ones((5,5,5)))
        >>> centered_dens, translation = dens.centered(0)
        >>> translation
        array([0., 0., 0.])

        :py:meth:`Density.centered` extended the :py:attr:`Density.data` attribute
        of the current :py:class:`Density` instance and modified
        :py:attr:`Density.origin` accordingly.

        >>> centered_dens
        Origin: (-2.0, -2.0, -2.0), sampling_rate: (1, 1, 1), Shape: (9, 9, 9)

        :py:meth:`Density.centered` achieves centering via zero-padding and
        rigid-transform of the internal :py:attr:`Density.data` attribute.
        `centered_dens` is sufficiently large to represent all rotations of the
        :py:attr:`Density.data` attribute, such as ones obtained from
        :py:meth:`tme.matching_utils.get_rotation_matrices`.

        >>> from tme.matching_utils import get_rotation_matrices
        >>> rotation_matrix = get_rotation_matrices(dim = 3 ,angular_sampling = 10)[0]
        >>> rotated_centered_dens = centered_dens.rigid_transform(
        >>>     rotation_matrix = rotation_matrix,
        >>>     order = None
        >>> )
        >>> print(centered_dens.data.sum(), rotated_centered_dens.data.sum())
        125.0 125.0

        """
        ret = self.copy()

        box = ret.minimum_enclosing_box(cutoff=cutoff, use_geometric_center=False)
        ret.adjust_box(box)

        new_shape = np.maximum(ret.shape, self.shape)
        new_shape = np.add(new_shape, 1 - np.mod(new_shape, 2))
        ret.pad(new_shape)

        center = self.center_of_mass(ret.data, cutoff)
        shift = np.subtract(np.divide(np.subtract(ret.shape, 1), 2), center)

        ret = ret.rigid_transform(
            translation=shift,
            rotation_matrix=np.eye(ret.data.ndim),
            use_geometric_center=False,
            order=1,
        )

        shift = np.subtract(center, self.center_of_mass(ret.data, cutoff))
        return ret, shift

    def rigid_transform(
        self,
        rotation_matrix: NDArray,
        translation: NDArray = None,
        order: int = 3,
        use_geometric_center: bool = True,
    ) -> "Density":
        """
        Performs a rigid transform of the class instance.

        Parameters
        ----------
        rotation_matrix : NDArray
            Rotation matrix to apply.
        translation : NDArray
            Translation to apply.
        order : int, optional
            Interpolation order to use. Default is 3, has to be in range 0-5.
        use_geometric_center : bool, optional
            Use geometric or mass center as rotation center.

        Returns
        -------
        Density
            The transformed instance of :py:class:`Density`.

        Examples
        --------
        Define the :py:class:`Density` instance

        >>> import numpy as np
        >>> from tme import Density
        >>> dens = Density(np.arange(9).reshape(3,3).astype(np.float32))
        >>> dens, translation = dens.centered(0)

        and apply the rotation, in this case a mirror around the z-axis

        >>> rotation_matrix = np.eye(dens.data.ndim)
        >>> rotation_matrix[0, 0] = -1
        >>> dens_transform = dens.rigid_transform(rotation_matrix = rotation_matrix)
        >>> dens_transform.data
        array([[0.        , 0.        , 0.        , 0.        , 0.        ],
               [0.5       , 3.0833333 , 3.5833333 , 3.3333333 , 0.        ],
               [0.75      , 4.6666665 , 5.6666665 , 5.4166665 , 0.        ],
               [0.25      , 1.6666666 , 2.6666667 , 2.9166667 , 0.        ],
               [0.        , 0.08333334, 0.5833333 , 0.8333333 , 0.        ]],
              dtype=float32)

        Notes
        -----
        This function assumes the internal :py:attr:`Density.data` attribute is
        sufficiently sized to hold the transformation.

        See Also
        --------
        :py:meth:`Density.centered`, :py:meth:`Density.minimum_enclosing_box`
        """
        ret = self.empty
        NumpyFFTWBackend().rigid_transform(
            arr=self.data,
            rotation_matrix=rotation_matrix,
            translation=translation,
            use_geometric_center=use_geometric_center,
            out=ret.data,
            order=order,
        )

        eps = np.finfo(ret.data.dtype).eps
        ret.data[np.abs(ret.data) < eps] = 0
        return ret

    def resample(
        self, new_sampling_rate: Tuple[float], method: str = "spline", order: int = 1
    ) -> "Density":
        """
        Resamples :py:attr:`Density.data` to ``new_sampling_rate``.

        Parameters
        ----------
        new_sampling_rate : tuple of floats or float
            Sampling rate to resample to for a single or all axes.
        method: str, optional
            Resampling method to use, defaults to `spline`. Availabe options are:

            +---------+----------------------------------------------------------+
            | spline  | Smooth spline interpolation via :obj:`scipy.ndimage.zoom`|
            +---------+----------------------------------------------------------+
            | fourier | Frequency preserving Fourier cropping                    |
            +---------+----------------------------------------------------------+

        order : int, optional
            Order of spline used for interpolation, by default 1. Ignored when
            ``method`` is `fourier`.

        Raises
        ------
        ValueError
            If ``method`` is not supported.

        Returns
        -------
        :py:class:`Density`
            A resampled copy of the class instance.

        Examples
        --------
        The following makes use of :py:meth:`tme.matching_utils.create_mask`
        to define a :py:class:`Density` instance containing a 2D circle with
        a sampling rate of 2

        >>> from tme import Density
        >>> from tme.matching_utils import create_mask
        >>> mask = create_mask(
        >>>     mask_type="ellipse",
        >>>     shape=(11,11),
        >>>     center=(5,5),
        >>>     radius=3
        >>> )
        >>> dens = Density(mask, sampling_rate=2)
        >>> dens
        Origin: (0.0, 0.0), sampling_rate: (2, 2), Shape: (11, 11)

        Using :py:meth:`Density.resample` we can modulate the sampling rate
        using spline interpolation of desired order

        >>> dens.resample(new_sampling_rate= 4, method="spline", order=3)
        Origin: (0.0, 0.0), sampling_rate: (4, 4), Shape: (6, 6)

        Or Fourier cropping which results in a less smooth output, but more faithfully
        captures the contained frequency information

        >>> dens.resample(new_sampling_rate=4, method="fourier")
        Origin: (0.0, 0.0), sampling_rate: (4, 4), Shape: (6, 6)

        ``new_sampling_rate`` can also be specified per axis

        >>> dens.resample(new_sampling_rate=(4,1), method="spline", order=3)
        Origin: (0.0, 0.0), sampling_rate: (4, 1), Shape: (6, 22)

        """
        _supported_methods = ("spline", "fourier")
        if method not in _supported_methods:
            raise ValueError(
                f"Expected method to be one of {_supported_methods}, got '{method}'."
            )
        new_sampling_rate = np.array(new_sampling_rate)
        new_sampling_rate = np.repeat(
            new_sampling_rate, self.data.ndim // new_sampling_rate.size
        )

        ret = self.copy()
        scale_factor = np.divide(ret.sampling_rate, new_sampling_rate)
        if method == "spline":
            ret.data = zoom(ret.data, scale_factor, order=order)
        elif method == "fourier":
            ret_shape = np.round(np.multiply(scale_factor, ret.shape)).astype(int)

            axis = range(len(ret_shape))
            mask = np.zeros(self.shape, dtype=bool)
            mask[tuple(slice(0, x) for x in ret_shape)] = 1
            mask = np.roll(
                mask, shift=-np.floor(np.divide(ret_shape, 2)).astype(int), axis=axis
            )
            mask_ret = np.zeros(ret_shape, dtype=bool)
            mask_ret[tuple(slice(0, x) for x in self.shape)] = 1
            mask_ret = np.roll(
                mask_ret,
                shift=-np.floor(np.divide(self.shape, 2)).astype(int),
                axis=axis,
            )

            arr_ft = np.fft.fftn(self.data)
            arr_ft *= np.prod(ret_shape) / np.prod(self.shape)
            ret_ft = np.zeros(ret_shape, dtype=arr_ft.dtype)
            ret_ft[mask_ret] = arr_ft[mask]
            ret.data = np.real(np.fft.ifftn(ret_ft))

        ret.sampling_rate = new_sampling_rate
        return ret

    def density_boundary(
        self, weight: float, fraction_surface: float = 0.1, volume_factor: float = 1.21
    ) -> Tuple[float]:
        """
        Computes the density boundary of the class instance. The density
        boundary in this setting is defined as minimal and maximal density value
        enclosing a certain ``weight``.

        Parameters
        ----------
        weight : float
            Density weight to compute volume cutoff on. This could e.g. be the
            sum of contained atomic weights.
        fraction_surface : float, optional
            Approximate fraction of surface voxels on all voxels enclosing
            ``weight``, by default 0.1. Decreasing this value increases the
            upper volume boundary.
        volume_factor : float, optional
            Factor used to compute how many distinct density values
            can be used to represent ``weight``, by default 1.21.

        Returns
        -------
        tuple
            Tuple containing lower and upper bound on densities.

        References
        ----------
        .. [1] Cragnolini T, Sahota H, Joseph AP, Sweeney A, Malhotra S,
            Vasishtan D, Topf M (2021a) TEMPy2: A Python library with
            improved 3D electron microscopy density-fitting and validation
            workflows. Acta Crystallogr Sect D Struct Biol 77:41–47.
            https://doi.org/10.1107/S2059798320014928

        Raises
        ------
        ValueError
            If input any input parameter is <= 0.
        """
        if weight <= 0 or fraction_surface <= 0 or volume_factor <= 0:
            raise ValueError(
                "weight, fraction_surface and volume_factor need to be >= 0."
            )
        num_voxels = np.min(
            volume_factor * weight / np.power(self.sampling_rate, self.data.ndim)
        ).astype(int)
        surface_included_voxels = int(num_voxels * (1 + fraction_surface))

        map_partition = np.partition(
            self.data.flatten(), (-num_voxels, -surface_included_voxels)
        )
        upper_limit = map_partition[-num_voxels]
        lower_limit = map_partition[-surface_included_voxels]

        return (lower_limit, upper_limit)

    def surface_coordinates(
        self, density_boundaries: Tuple[float], method: str = "ConvexHull"
    ) -> NDArray:
        """
        Calculates the surface coordinates of the class instance using
        different boundary and surface detection methods. This method is relevant
        for determining coordinates used in non-exhaustive template matching,
        see :py:class:`tme.matching_optimization.optimize_match`.

        Parameters
        ----------
        density_boundaries : tuple
            Lower and upper bound of density values to be considered
            (can be obtained from :py:meth:`Density.density_boundary`).
        method : str, optional
            Method to use for surface coordinate computation

            +--------------+-----------------------------------------------------+
            | ConvexHull   | Use the lower bound density convex hull vertices.   |
            +--------------+-----------------------------------------------------+
            | Weight       | Use all coordinates within ``density_boundaries``.  |
            +--------------+-----------------------------------------------------+
            | Sobel        | Set densities below the lower bound density to zero |
            |              | apply a sobel filter and return density coordinates |
            |              | larger than 0.5 times the maximum filter value.     |
            +--------------+-----------------------------------------------------+
            | Laplace      | Like 'Sobel', but with a Laplace filter.            |
            +--------------+-----------------------------------------------------+
            | Minimum      | Like 'Sobel' and 'Laplace' but with a spherical     |
            |              | minimum filter on the lower density bound.          |
            +--------------+-----------------------------------------------------+

        Raises
        ------
        ValueError
            If the chosen method is not available.

        Returns
        -------
        NDArray
            An array of surface coordinates with shape (points, dimensions).

        References
        ----------
        .. [1] Cragnolini T, et al. (2021) Acta Crys Sect D Struct Biol

        See Also
        --------
        :py:class:`tme.matching_optimization.NormalVectorScore`
        :py:class:`tme.matching_optimization.PartialLeastSquareDifference`
        :py:class:`tme.matching_optimization.MutualInformation`
        :py:class:`tme.matching_optimization.Envelope`
        :py:class:`tme.matching_optimization.Chamfer`
        """
        _available_methods = ["ConvexHull", "Weight", "Sobel", "Laplace", "Minimum"]

        if method not in _available_methods:
            raise ValueError(
                "Argument method has to be one of the following: %s"
                % ", ".join(_available_methods)
            )

        lower_bound, upper_bound = density_boundaries
        if method == "ConvexHull":
            binary = np.transpose(np.where(self.data > lower_bound))
            hull = ConvexHull(binary)
            surface_points = binary[hull.vertices[:]]

        elif method == "Sobel":
            filtered_map = np.multiply(self.data, (self.data > lower_bound))
            magn = generic_gradient_magnitude(filtered_map, sobel)
            surface_points = np.argwhere(magn > 0.5 * magn.max())

        elif method == "Laplace":
            filtered_map = self.data > lower_bound
            magn = laplace(filtered_map)
            surface_points = np.argwhere(magn > 0.5 * magn.max())

        elif method == "Minimum":
            fp = np.zeros((self.data.ndim,) * self.data.ndim)
            center = np.ones(self.data.ndim, dtype=int)
            fp[tuple(center)] = 1
            for i in range(self.data.ndim):
                offset = np.zeros(self.data.ndim, dtype=int)
                offset[i] = 1
                fp[tuple(center + offset)] = 1
                fp[tuple(center - offset)] = 1

            filtered_map = (self.data > lower_bound).astype(int)
            filtered_map_surface = minimum_filter(
                filtered_map, footprint=fp, mode="constant", cval=0.8
            )
            filtered_map_surface = ((filtered_map - filtered_map_surface) == 1).astype(
                int
            )
            surface_points = np.argwhere(filtered_map_surface == 1)

        elif method == "Weight":
            surface_points = np.argwhere(
                np.logical_and(self.data < upper_bound, self.data > lower_bound)
            )

        return surface_points

    def normal_vectors(self, coordinates: NDArray) -> NDArray:
        """
        Calculates the normal vectors for the given coordinates on the densities
        of the class instance. If the normal vector to a given coordinate
        can not be computed, the zero vector is returned instead. The output of this
        function can e.g. be used in
        :py:class:`tme.matching_optimization.NormalVectorScore`.

        Parameters
        ----------
        coordinates : NDArray
            An array of integer coordinates with shape (dimensions, coordinates)
            of which to calculate the normal vectors.

        Returns
        -------
        NDArray
            An array with unit normal vectors with same shape as coordinates.

        References
        ----------
        .. [1] Cragnolini T, Sahota H, Joseph AP, Sweeney A, Malhotra S,
            Vasishtan D, Topf M (2021a) TEMPy2: A Python library with
            improved 3D electron microscopy density-fitting and validation
            workflows. Acta Crystallogr Sect D Struct Biol 77:41–47.
            https://doi.org/10.1107/S2059798320014928

        Raises
        ------
        ValueError
            If coordinates.shape[1] does not match self.data.ndim,
            coordinates.ndim != 2 or lies outside self.data.

        See Also
        --------
        :py:class:`tme.matching_optimization.NormalVectorScore`
        :py:class:`tme.matching_optimization.PartialLeastSquareDifference`
        :py:class:`tme.matching_optimization.MutualInformation`
        :py:class:`tme.matching_optimization.Envelope`
        :py:class:`tme.matching_optimization.Chamfer`
        """
        normal_vectors, coordinates = [], np.asarray(coordinates, dtype=int)

        if coordinates.ndim != 2:
            raise ValueError("Coordinates should have shape point x dimension.")
        if coordinates.shape[1] != self.data.ndim:
            raise ValueError(
                f"Expected coordinate dimension {self.data.ndim}, "
                f"got {coordinates.shape[1]}."
            )
        in_box = np.logical_and(
            coordinates < np.array(self.shape), coordinates >= 0
        ).min(axis=1)
        coordinates = coordinates[in_box, :]
        for index in range(coordinates.shape[0]):
            point = coordinates[index, :]
            start = np.maximum(point - 1, 0)
            stop = np.minimum(point + 2, self.data.shape)
            slc = tuple(slice(*coords) for coords in zip(start, stop))

            inner_facing = np.array(np.where(self.data[slc] > self.data[tuple(point)]))
            if inner_facing.size == 0:
                normal_vectors.append(np.zeros_like(point))
                continue
            inner_facing -= np.ones_like(point)[:, None]
            inner_facing = inner_facing.sum(axis=1)
            inner_facing = inner_facing / np.linalg.norm(inner_facing)
            normal_vectors.append(inner_facing)

        return np.array(normal_vectors)

    def core_mask(self) -> NDArray:
        """
        Calculates a weighted core mask by performing iterative binary erosion on
        :py:attr:`Density.data`. In each iteration, all mask elements corresponding
        to a non-zero data elemnt are incremented by one. Therefore, a mask element
        with value N corresponds to a data value that remained non-zero for N iterations.
        Mask elements with high values are likely part of the core density [1]_.

        Returns
        -------
        NDArray
            Core-weighted mask with shape of :py:attr:`Density.data`.

        References
        ----------
        .. [1]  Gydo Zundert and Alexandre Bonvin. Fast and sensitive rigid-body
                fitting into cryo-em density maps with powerfit. AIMS Biophysics,
                2:73–87, 04 2015. doi:10.3934/biophy.2015.2.73
        """
        core_indices = np.zeros(self.shape)
        eroded_mask = self.data > 0
        while eroded_mask.sum() > 0:
            core_indices += eroded_mask
            eroded_mask = binary_erosion(eroded_mask)
        return core_indices

    @staticmethod
    def center_of_mass(arr: NDArray, cutoff: float = None) -> NDArray:
        """
        Computes the center of mass of a numpy ndarray instance using all available
        elements. For template matching it typically makes sense to only input
        positive densities.

        Parameters
        ----------
        arr : NDArray
            Array to compute the center of mass of.
        cutoff : float, optional
            Densities less than or equal to cutoff are nullified for center
            of mass computation. By default considers all values.

        Returns
        -------
        NDArray
            Center of mass with shape (arr.ndim).
        """
        return NumpyFFTWBackend().center_of_mass(arr, cutoff)

    @classmethod
    def match_densities(
        cls,
        target: "Density",
        template: "Density",
        cutoff_target: float = 0,
        cutoff_template: float = 0,
        scoring_method: str = "NormalizedCrossCorrelation",
        **kwargs,
    ) -> Tuple["Density", NDArray, NDArray, NDArray]:
        """
        Aligns two :py:class:`Density` instances target and template and returns
        the aligned template.

        If voxel sizes of target and template dont match coordinates are scaled
        to the numerically smaller voxel size. Instances are prealigned based on their
        center of mass. Finally :py:meth:`tme.matching_optimization.optimize_match` is
        used to determine translation and rotation to map template to target.

        Parameters
        ----------
        target : Density
            The target map for alignment.
        template : Density
            The template that should be aligned to the target.
        cutoff_target : float, optional
            The cutoff value for the target map, by default 0.
        cutoff_template : float, optional
            The cutoff value for the template map, by default 0.
        scoring_method : str, optional
            The scoring method to use for alignment. See
            :py:class:`tme.matching_optimization.create_score_object` for available methods,
            by default "NormalizedCrossCorrelation".
        kwargs : dict, optional
            Optional keyword arguments passed to
            :py:meth:`tme.matching_optimization.optimize_match`.

        Returns
        -------
        Tuple
            Tuple containing template aligned to target as :py:class:`Density` object,
            translation in voxels and rotation matrix used for the transformation.

        Notes
        -----
        No densities below cutoff_template are present in the returned Density object.
        """
        from .matching_utils import normalize_template
        from .matching_optimization import optimize_match, create_score_object

        template_mask = template.empty
        template_mask.data.fill(1)

        normalize_template(
            template=template.data,
            mask=template_mask.data,
            n_observations=template_mask.data.sum(),
        )

        target_sampling_rate = np.array(target.sampling_rate)
        template_sampling_rate = np.array(template.sampling_rate)
        target_sampling_rate = np.repeat(
            target_sampling_rate, target.data.ndim // target_sampling_rate.size
        )
        template_sampling_rate = np.repeat(
            template_sampling_rate, template.data.ndim // template_sampling_rate.size
        )
        if not np.allclose(target_sampling_rate, template_sampling_rate):
            print(
                "Voxel size of target and template do not match. "
                "Using smaller voxel size for refinement."
            )

        target_coordinates = target.to_pointcloud(cutoff_target)

        template_coordinates = template.to_pointcloud(cutoff_template)
        template_weights = template.data[tuple(template_coordinates)]

        refinement_sampling_rate = np.minimum(
            target_sampling_rate, template_sampling_rate
        )
        target_scaling = np.divide(target_sampling_rate, refinement_sampling_rate)
        template_scaling = np.divide(template_sampling_rate, refinement_sampling_rate)
        target_coordinates = target_coordinates * target_scaling[:, None]
        template_coordinates = template_coordinates * template_scaling[:, None]

        mass_center_difference = np.subtract(
            cls.center_of_mass(target.data, cutoff_target),
            cls.center_of_mass(template.data, cutoff_template),
        ).astype(int)
        template_coordinates += mass_center_difference[:, None]

        coordinates_mask = template_mask.to_pointcloud()
        coordinates_mask = coordinates_mask * template_scaling[:, None]
        coordinates_mask += mass_center_difference[:, None]

        score_object = create_score_object(
            score=scoring_method,
            target=target.data,
            template_coordinates=template_coordinates,
            template_mask_coordinates=coordinates_mask,
            template_weights=template_weights,
            sampling_rate=np.ones(template.data.ndim),
        )

        translation, rotation_matrix, score = optimize_match(
            score_object=score_object, **kwargs
        )

        translation += mass_center_difference
        translation = np.divide(translation, template_scaling)

        template.sampling_rate = template_sampling_rate.copy()
        ret = template.rigid_transform(
            rotation_matrix=rotation_matrix, use_geometric_center=False
        )
        ret.origin = target.origin.copy()
        ret.origin = ret.origin + np.multiply(translation, target_sampling_rate)

        return ret, translation, rotation_matrix

    @classmethod
    def match_structure_to_density(
        cls,
        target: "Density",
        template: "Structure",
        cutoff_target: float = 0,
        scoring_method: str = "NormalizedCrossCorrelation",
        optimization_method: str = "basinhopping",
        maxiter: int = 500,
    ) -> Tuple["Structure", NDArray, NDArray]:
        """
        Aligns a :py:class:`tme.structure.Structure` template to :py:class:`Density`
        target and returns an aligned :py:class:`tme.structure.Structure` instance.

        If voxel sizes of target and template dont match coordinates are scaled
        to the numerically smaller voxel size. Prealignment is done by center's
        of mass. Finally :py:class:`tme.matching_optimization.optimize_match` is used to
        determine translation and rotation to match a template to target.

        Parameters
        ----------
        target : Density
            The target map for template matching.
        template : Structure
            The template that should be aligned to the target.
        cutoff_target : float, optional
            The cutoff value for the target map, by default 0.
        cutoff_template : float, optional
            The cutoff value for the template map, by default 0.
        scoring_method : str, optional
            The scoring method to use for template matching. See
            :py:class:`tme.matching_optimization.create_score_object` for available methods,
            by default "NormalizedCrossCorrelation".
        optimization_method : str, optional
            Optimizer that is used.
            See :py:meth:`tme.matching_optimization.optimize_match`.
        maxiter : int, optional
            Maximum number of iterations for the optimizer.
            See :py:meth:`tme.matching_optimization.optimize_match`.

        Returns
        -------
        Structure
            Tuple containing template aligned to target as
            :py:class:`tme.structure.Structure` object, translation and rotation
            matrix used for the transformation.

        Notes
        -----
        Translation and rotation are in xyz format, different from
        :py:meth:`match_densities`, which is zyx.
        """
        template_density = cls.from_structure(
            filename_or_structure=template, sampling_rate=target.sampling_rate
        )

        ret, translation, rotation_matrix = cls.match_densities(
            target=target,
            template=template_density,
            cutoff_target=cutoff_target,
            cutoff_template=0,
            scoring_method=scoring_method,
            optimization_method=optimization_method,
            maxiter=maxiter,
        )
        out = template.copy()
        final_translation = np.subtract(ret.origin, template_density.origin)

        # Atom coordinates are in xyz
        final_translation = final_translation[::-1]
        rotation_matrix = rotation_matrix[::-1, ::-1]

        out = out.rigid_transform(
            translation=final_translation, rotation_matrix=rotation_matrix
        )

        return out, final_translation, rotation_matrix

    @staticmethod
    def fourier_shell_correlation(density1: "Density", density2: "Density") -> NDArray:
        """
        Computes the Fourier Shell Correlation (FSC) between two instances of `Density`.

        The Fourier transforms of the input maps are divided into shells
        based on their spatial frequency. The correlation between corresponding shells
        in the two maps is computed to give the FSC.

        Parameters
        ----------
        density1 : Density
            An instance of `Density` class for the first map for comparison.
        density2 : Density
            An instance of `Density` class for the second map for comparison.

        Returns
        -------
        NDArray
            An array of shape (N, 2), where N is the number of shells,
            the first column represents the spatial frequency for each shell
            and the second column represents the corresponding FSC.

        References
        ----------
        .. [1] https://github.com/tdgrant1/denss/blob/master/saxstats/saxstats.py
        """
        side = density1.data.shape[0]
        df = 1.0 / side

        qx_ = np.fft.fftfreq(side) * side * df
        qx, qy, qz = np.meshgrid(qx_, qx_, qx_, indexing="ij")
        qr = np.sqrt(qx**2 + qy**2 + qz**2)

        qmax = np.max(qr)
        qstep = np.min(qr[qr > 0])
        nbins = int(qmax / qstep)
        qbins = np.linspace(0, nbins * qstep, nbins + 1)
        qbin_labels = np.searchsorted(qbins, qr, "right") - 1

        F1 = np.fft.fftn(density1.data)
        F2 = np.fft.fftn(density2.data)

        qbin_labels = qbin_labels.reshape(-1)
        numerator = np.bincount(
            qbin_labels, weights=np.real(F1 * np.conj(F2)).reshape(-1)
        )
        term1 = np.bincount(qbin_labels, weights=np.abs(F1).reshape(-1) ** 2)
        term2 = np.bincount(qbin_labels, weights=np.abs(F2).reshape(-1) ** 2)
        np.multiply(term1, term2, out=term1)
        denominator = np.sqrt(term1)
        FSC = np.divide(numerator, denominator)

        qidx = np.where(qbins < qx.max())

        return np.vstack((qbins[qidx], FSC[qidx])).T


def is_gzipped(filename: str) -> bool:
    """Check if a file is a gzip file by reading its magic number."""
    with open(filename, "rb") as f:
        return f.read(2) == b"\x1f\x8b"
