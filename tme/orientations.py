"""
Handle template matching orientations and conversion between formats.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import warnings
from typing import Dict, List, Tuple
from dataclasses import dataclass, field, InitVar
from string import ascii_lowercase, ascii_uppercase

import numpy as np

from .types import ArrayLike
from .parser import StarParser

# Exceeds available numpy dimensions for default installations
NAMES = ["x", "y", "z", *ascii_lowercase[:-3], *ascii_uppercase]


@dataclass
class Orientations:
    """
    Handle template matching orientations and conversion between formats.

    Examples
    --------
    The following achieves the minimal definition of an :py:class:`Orientations` instance

    >>> import numpy as np
    >>> from tme import Orientations
    >>> translations = np.random.randint(low = 0, high = 100, size = (100,3))
    >>> rotations = np.random.rand(100, 3)
    >>> orientations = Orientations(
    >>>     translations=translations,
    >>>     rotations=rotations,
    >>> )

    The created ``orientations`` object can be written to disk in a range of formats.
    See :py:meth:`Orientations.to_file` for available formats. The following creates
    a STAR file

    >>> orientations.to_file("test.star")

    :py:meth:`Orientations.from_file` can create :py:class:`Orientations` instances
    from a range of formats, to enable conversion between formats

    >>> orientations_star = Orientations.from_file("test.star")
    >>> np.all(orientations.translations == orientations_star.translations)
    True

    Parameters
    ----------
    translations: array_like
        Array with translations of each orientations (n, d).
    rotations: array_like
        Array with euler angles of each orientation in zxy convention (n, d).
    scores: array_like, optional
        Array with the score of each orientation (n, ). When provided, stored in
        ``metadata["_pytmeScore"]`` and overrides any preexisting value there.
    details: array_like, optional
        Array with additional orientation details (n, ). When provided, stored in
        ``metadata["_rlnClassNumber"]`` and overrides any preexisting value there.
    metadata: dict, optional
        Per-particle metadata. Array values whose first dimension matches the
        number of orientations are sliced by :py:meth:`__getitem__`.
    optics: dict, optional
        File-level optics-group metadata (Relion-style).
    """

    translations: ArrayLike
    rotations: ArrayLike
    scores: InitVar[ArrayLike] = None
    details: InitVar[ArrayLike] = None
    metadata: Dict = field(default_factory=dict)
    optics: Dict = field(default_factory=dict)

    _METADATA_ALIASES = {"scores": "_pytmeScore", "details": "_rlnClassNumber"}

    def __post_init__(self, scores, details):
        self.translations = np.asarray(self.translations).astype(np.float32)
        self.rotations = np.asarray(self.rotations).astype(np.float32)

        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dict.")
        if not isinstance(self.optics, dict):
            raise ValueError("optics must be a dict.")
        if self.translations.ndim != 2 or self.rotations.ndim != 2:
            raise ValueError("Expected stack of translations and rotations.")

        n = self.translations.shape[0]
        if scores is not None:
            warnings.warn(
                "The `scores=` argument is deprecated and will be removed in a "
                "future release; set metadata['_pytmeScore'] instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.metadata["_pytmeScore"] = np.asarray(scores, dtype=np.float32)
        elif "_pytmeScore" not in self.metadata:
            self.metadata["_pytmeScore"] = np.zeros(n, np.float32)

        if details is not None:
            warnings.warn(
                "The `details=` argument is deprecated and will be removed in a "
                "future release; set metadata['_rlnClassNumber'] instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.metadata["_rlnClassNumber"] = np.asarray(details)
        elif "_rlnClassNumber" not in self.metadata:
            self.metadata["_rlnClassNumber"] = np.full(n, -1)

        if self.rotations.shape[0] != n:
            raise ValueError(
                "The first dimension of all parameters needs to be of equal length."
            )
        for key, val in self.metadata.items():
            if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] != n:
                raise ValueError(
                    "The first dimension of all parameters needs to be of equal length."
                )

    def __getattr__(self, name):
        key = type(self)._METADATA_ALIASES.get(name)
        if key is None:
            raise AttributeError(name)
        return self.metadata.get(key)

    def __setstate__(self, state: Dict) -> None:
        # Pre v0.3.4 pickles lack metadata; pre-optics pickles lack optics
        state.setdefault("metadata", {})
        state.setdefault("optics", {})
        legacy_scores = state.pop("scores", None)
        legacy_details = state.pop("details", None)
        if legacy_scores is not None and "_pytmeScore" not in state["metadata"]:
            state["metadata"]["_pytmeScore"] = np.asarray(
                legacy_scores, dtype=np.float32
            )
        if legacy_details is not None and "_rlnClassNumber" not in state["metadata"]:
            state["metadata"]["_rlnClassNumber"] = np.asarray(legacy_details)
        self.__dict__.update(state)

    def __iter__(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Iterate over the current class instance. Each iteration returns a orientation
        defined by its translation, rotation, score and additional detail.

        Yields
        ------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple of arrays defining the given orientation.
        """
        yield from zip(self.translations, self.rotations, self.scores, self.details)

    def __len__(self) -> int:
        """Return the number of distinct particles"""
        return self.translations.shape[0]

    def __getitem__(self, indices: List[int]) -> "Orientations":
        """
        Retrieve a subset of orientations based on the provided indices.

        Parameters
        ----------
        indices : List[int]
            A list of indices specifying the orientations to be retrieved.

        Returns
        -------
        :py:class:`Orientations`
            A new :py:class:`Orientations`instance containing only the selected orientations.
        """
        indices = np.asarray(indices)
        new_metadata = {}
        for key, val in self.metadata.items():
            new_metadata[key] = val
            if isinstance(val, np.ndarray) and val.shape[0] == len(self):
                new_metadata[key] = val[indices].copy()

        return self.__class__(
            translations=self.translations[indices].copy(),
            rotations=self.rotations[indices].copy(),
            metadata=new_metadata,
            optics=dict(self.optics),
        )

    def copy(self) -> "Orientations":
        """
        Create a copy of the current class instance.

        Returns
        -------
        :py:class:`Orientations`
            Copy of the class instance.
        """
        return self[np.arange(self.scores.size)]

    def to_file(self, filename: str, file_format: type = None, **kwargs) -> None:
        """
        Save the current class instance to a file in the specified format.

        Parameters
        ----------
        filename : str
            The name of the file where the orientations will be saved.
        file_format : type, optional
            The format in which to save the orientations. Defaults to None and infers
            the file_format from the typical extension. Supported formats are

            +------------+----------------------------------------------------+
            | tsv        | pytme's standard tab-separated orientations file   |
            +------------+----------------------------------------------------+
            | star       | Creates a STAR file of orientations                |
            +------------+----------------------------------------------------+
            | dynamo     | Creates a dynamo table                             |
            +------------+----------------------------------------------------+

        **kwargs : dict
            Additional keyword arguments specific to the file format.

        Raises
        ------
        ValueError
            If an unsupported file format is specified.
        """
        mapping = {
            "tsv": self._to_text,
            "star": self._to_star,
            "dynamo": self._to_dynamo_tbl,
        }
        if file_format is None:
            if filename.lower().endswith(".star"):
                file_format = "star"
            elif filename.lower().endswith(".tbl"):
                file_format = "dynamo"
            elif filename.lower().endswith(".tsv"):
                file_format = "tsv"

        func = mapping.get(file_format, None)
        if func is None:
            raise ValueError(
                f"{file_format} not implemented. Supported are {','.join(mapping.keys())}."
            )

        return func(filename=filename, **kwargs)

    def _to_text(self, filename: str, **kwargs) -> None:
        """
        Save orientations in a text file format.

        Parameters
        ----------
        filename : str
            The name of the file to save the orientations.

        Notes
        -----
        The file is saved with a header specifying each column: x, y, z, euler_x,
        euler_y, euler_z, score, detail. Each row in the file corresponds to an orientation.
        """
        header = "\t".join(
            [
                *list(NAMES[: self.translations.shape[1]]),
                *[f"euler_{x}" for x in NAMES[: self.rotations.shape[1]]],
                "score",
                "detail",
            ]
        )
        with open(filename, mode="w", encoding="utf-8") as ofile:
            _ = ofile.write(f"{header}\n")
            for translation, angles, score, detail in self:
                out_string = (
                    "\t".join([str(x) for x in (*translation, *angles, score, detail)])
                    + "\n"
                )
                _ = ofile.write(out_string)
        return None

    def _to_dynamo_tbl(
        self,
        filename: str,
        name_prefix: str = None,
        sampling_rate: float = 1.0,
        subtomogram_size: int = 0,
        **kwargs,
    ) -> None:
        """
        Save orientations in Dynamo's tbl file format.

        Parameters
        ----------
        filename : str
            The name of the file to save the orientations.
        sampling_rate : float, optional
            Subtomogram sampling rate in angstrom per voxel

        Notes
        -----
        The file is saved with a standard header used in Dynamo tbl files
        outlined in [1]_. Each row corresponds to a particular partice.

        References
        ----------
        .. [1]  https://wiki.dynamo.biozentrum.unibas.ch/w/index.php/Table
        """
        with open(filename, mode="w", encoding="utf-8") as ofile:
            for index, (translation, rotation, score, detail) in enumerate(self):
                out = [
                    index,
                    1,
                    0,
                    0,
                    0,
                    0,
                    *rotation,
                    self.scores[index],
                    self.scores[index],
                    0,
                    0,
                    # Wedge parameters
                    -90,
                    90,
                    -60,
                    60,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    # Coordinate in original volume
                    *translation,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    sampling_rate,
                    3,
                    0,
                    0,
                ]
                _ = ofile.write(" ".join([str(x) for x in out]) + "\n")

        return None

    def _to_star(
        self, filename: str, source_path: str = None, version: str = None, **kwargs
    ) -> None:
        """
        Save orientations in STAR file format.

        Parameters
        ----------
        filename : str
            The name of the file to save the orientations.
        source_path : str
            Path to image file the orientation is in reference to.
            Per-particle sources from ``metadata["source"]`` take precedence.
        version : str
            Version indicator.
        """
        from .__version__ import __version__

        header = [
            "data_particles",
            "",
            "loop_",
            "_rlnCoordinateX",
            "_rlnCoordinateY",
            "_rlnCoordinateZ",
            "_rlnAngleRot",
            "_rlnAngleTilt",
            "_rlnAnglePsi",
        ]

        target_identifer = "_rlnMicrographName"
        if version == "# version 50001":
            header[3] = "_rlnCenteredCoordinateXAngst"
            header[4] = "_rlnCenteredCoordinateYAngst"
            header[5] = "_rlnCenteredCoordinateZAngst"
            target_identifer = "_rlnTomoName"

        if source_path is not None:
            header.append(target_identifer)

        n = self.translations.shape[0]
        extra_meta_keys = [
            key
            for key, val in self.metadata.items()
            if isinstance(val, np.ndarray) and val.shape[0] == n and key not in header
        ]
        header.extend(extra_meta_keys)

        optics_block = ""
        if self.optics:
            optics_block = ["data_optics", "", "loop_"]
            optics_block.extend(self.optics.keys())
            optics_block.append(" ".join(str(v) for v in self.optics.values()))
            optics_block.append("")
            optics_block = "\n".join(optics_block) + "\n"

        header = "\n".join(header)
        with open(filename, mode="w", encoding="utf-8") as ofile:
            _ = ofile.write(f"# Created using pytme (version {__version__}).\n\n")

            if version is not None:
                _ = ofile.write(f"{version.strip()}\n\n")

            _ = ofile.write(optics_block)

            _ = ofile.write(f"{header}\n")
            for index, (translation, rotation, _, _) in enumerate(self):
                line = [str(x) for x in translation]
                line.extend([str(x) for x in rotation])

                if source_path is not None:
                    line.append(source_path)

                for key in extra_meta_keys:
                    line.append(str(self.metadata[key][index]))
                _ = ofile.write("\t".join([str(x) for x in line]) + "\n")

        return None

    @classmethod
    def from_file(
        cls, filename: str, file_format: type = None, **kwargs
    ) -> "Orientations":
        """
        Create an instance of :py:class:`Orientations` from a file.

        Parameters
        ----------
        filename : str
            The name of the file from which to read the orientations.
        file_format : type, optional
            The format of the file. Defaults to None and infers
            the file_format from the typical extension. Supported formats are

            +------------+----------------------------------------------------+
            | tsv        | Read pytme's tab-separated orientations file       |
            +------------+----------------------------------------------------+
            | star       | Read a STAR file of orientations                   |
            +------------+----------------------------------------------------+
            | dynamo     | Read a dynamo table                                |
            +------------+----------------------------------------------------+

        **kwargs
            Additional keyword arguments specific to the file format.

        Returns
        -------
        :py:class:`Orientations`
            An instance of :py:class:`Orientations` populated with data from the file.

        Raises
        ------
        ValueError
            If an unsupported file format is specified.
        """
        mapping = {
            "tsv": cls._from_text,
            "star": cls._from_star,
            "tbl": cls._from_tbl,
        }
        if file_format is None:
            if filename.lower().endswith(".star"):
                file_format = "star"
            elif filename.lower().endswith(".tbl"):
                file_format = "tbl"
            elif filename.lower().endswith(".tsv"):
                file_format = "tsv"

        func = mapping.get(file_format, None)
        if func is None:
            raise ValueError(
                f"{file_format} not implemented. Supported are {','.join(mapping.keys())}."
            )

        translation, rotation, meta, optics = func(filename=filename, **kwargs)
        return cls(
            translations=translation,
            rotations=rotation,
            metadata=meta,
            optics=optics,
        )

    @staticmethod
    def _from_text(
        filename: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        Read orientations from a text file.

        Parameters
        ----------
        filename : str
            The name of the file from which to read the orientations.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Dict, Dict]
            Translation, rotation, meta dict with scores/class and empty optics dict.

        Notes
        -----
        The text file is expected to have a header and data in columns. Colums containing
        the name euler are considered to specify rotations. The second last and last
        column correspond to score and detail. Its possible to only specify translations,
        in this case the remaining columns will be filled with trivial values.
        """
        with open(filename, mode="r", encoding="utf-8") as infile:
            data = [x.strip().split("\t") for x in infile.read().split("\n")]

        header = data.pop(0)
        translation, rotation, score, detail = [], [], [], []
        for candidate in data:
            if len(candidate) <= 1:
                continue

            translation.append(
                tuple(candidate[i] for i, x in enumerate(header) if x in NAMES)
            )
            rotation.append(
                tuple(candidate[i] for i, x in enumerate(header) if "euler" in x)
            )
            score.append(candidate[-2])
            detail.append(candidate[-1])

        translation, rotation = np.vstack(translation), np.vstack(rotation)
        if translation.shape[1] == len(header):
            rotation = np.zeros(translation.shape, dtype=np.float32)
            score = np.zeros(translation.shape[0], dtype=np.float32)
            detail = np.zeros(translation.shape[0], dtype=np.float32) - 1

        if rotation.size == 0 and translation.shape[0] != 0:
            rotation = np.zeros(translation.shape, dtype=np.float32)

        header_order = tuple(x for x in header if x in NAMES)
        sort_order = tuple(NAMES.index(x) for x in header_order)
        translation = translation[..., sort_order]

        header_order = tuple(
            x for x in header if "euler" in x and x.replace("euler_", "") in NAMES
        )
        header_order = zip(header_order, range(len(header_order)))
        sort_order = tuple(
            x[1] for x in sorted(header_order, key=lambda x: x[0], reverse=False)
        )
        rotation = rotation[..., sort_order]

        metadata = {"_pytmeScore": score, "_rlnClassNumber": detail}
        return translation, rotation, metadata, {}

    @classmethod
    def _from_star(
        cls, filename: str, delimiter: str = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        parser = StarParser(filename, delimiter=delimiter)

        keyword_order = ("data_particles", "particles", "data")
        for keyword in keyword_order:
            ret = parser.get(keyword, None)
            if ret is None:
                ret = parser.get(f"{keyword}_", None)
            if ret is not None:
                break

        if ret is None:
            raise ValueError(
                f"Could not find either {keyword_order} section found in {filename}."
            )

        keys_v4 = ("_rlnCoordinateX", "_rlnCoordinateY", "_rlnCoordinateZ")
        keys_v5 = (
            "_rlnCenteredCoordinateXAngst",
            "_rlnCenteredCoordinateYAngst",
            "_rlnCenteredCoordinateZAngst",
        )
        if all(key in ret for key in keys_v4):
            keys = keys_v4
        elif all(key in ret for key in keys_v5):
            keys = keys_v5
        else:
            raise ValueError(
                f"File format not recognized. Need either {keys_v4} or {keys_v5}."
            )
        translation = np.vstack(tuple(ret[x] for x in keys)).astype(np.float32).T

        default_angle = np.zeros(translation.shape[0], dtype=np.float32)
        for x in ("_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"):
            if x not in ret:
                ret[x] = default_angle

        rotation = np.vstack(
            (ret["_rlnAngleRot"], ret["_rlnAngleTilt"], ret["_rlnAnglePsi"])
        )
        rotation = rotation.astype(np.float32).T

        metadata = {}
        consumed_keys = set(keys) | {"_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi"}
        for key, val in ret.items():
            if key in consumed_keys:
                continue
            try:
                metadata[key] = np.array(val, dtype=float)
            except (ValueError, TypeError):
                metadata[key] = np.array(val)

        optics, section = {}, parser.get("data_optics", {})
        for key, values in section.items():
            if not values:
                continue
            try:
                optics[key] = float(values[0])
            except (TypeError, ValueError):
                optics[key] = values[0]
        return translation, rotation, metadata, optics

    @staticmethod
    def _from_tbl(
        filename: str, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with open(filename, mode="r", encoding="utf-8") as infile:
            data = infile.read().split("\n")
        data = [x.strip().split(" ") for x in data if len(x.strip())]

        if len(data[0]) != 38:
            raise ValueError(
                "Expected tbl file to have 38 columns generated by _to_tbl."
            )

        translations, rotations, scores, details = [], [], [], []
        for peak in data:
            rotations.append((peak[6], peak[7], peak[8]))
            scores.append(peak[9])
            details.append(-1)
            translations.append((peak[23], peak[24], peak[25]))

        metadata = {"_pytmeScore": scores, "_rlnClassNumber": details}
        return translations, rotations, metadata, {}

    def get_extraction_slices(
        self,
        target_shape: Tuple[int],
        extraction_shape: Tuple[int],
        drop_out_of_box: bool = False,
        return_orientations: bool = False,
    ) -> "Orientations":
        """
        Calculate slices for extracting regions of interest within a larger array.

        Parameters
        ----------
        target_shape : Tuple[int]
            The shape of the target array within which regions are to be extracted.
        extraction_shape : Tuple[int]
            The shape of the regions to be extracted.
        drop_out_of_box : bool, optional
            If True, drop regions that extend beyond the target array boundary, by default False.
        return_orientations : bool, optional
            If True, return orientations along with slices, by default False.

        Returns
        -------
        Union[Tuple[List[slice]], Tuple["Orientations", List[slice], List[slice]]]
            If return_orientations is False, returns a tuple containing slices for candidate
            regions and observation regions.
            If return_orientations is True, returns a tuple containing orientations along
            with slices for candidate regions and observation regions.

        Raises
        ------
        SystemExit
            If no peak remains after filtering, indicating an error.
        """
        from .matching_utils import compute_extraction_box

        obs_beg, obs_end, cand_beg, cand_end, keep = compute_extraction_box(
            self.translations.astype(int),
            extraction_shape=extraction_shape,
            original_shape=target_shape,
        )

        subset = self
        if drop_out_of_box:
            n_remaining = keep.sum()
            if n_remaining == 0:
                print("No peak remaining after filtering")
            subset = self[keep]
            cand_beg, cand_end = cand_beg[keep,], cand_end[keep,]
            obs_beg, obs_end = obs_beg[keep,], obs_end[keep,]

        cand_beg, cand_end = cand_beg.astype(int), cand_end.astype(int)
        obs_beg, obs_end = obs_beg.astype(int), obs_end.astype(int)
        candidate_slices = [
            tuple(slice(s, e) for s, e in zip(start_row, stop_row))
            for start_row, stop_row in zip(cand_beg, cand_end)
        ]

        observation_slices = [
            tuple(slice(s, e) for s, e in zip(start_row, stop_row))
            for start_row, stop_row in zip(obs_beg, obs_end)
        ]

        if return_orientations:
            return subset, candidate_slices, observation_slices
        return candidate_slices, observation_slices


# The `scores`/`details` InitVar defaults register as class attributes that
# would shadow __getattr__; remove them so reads fall through to metadata.
del Orientations.scores
del Orientations.details
