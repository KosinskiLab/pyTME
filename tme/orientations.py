"""
Handle template matching orientations and conversion between formats.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import List, Tuple
from dataclasses import dataclass
from string import ascii_lowercase, ascii_uppercase

import numpy as np

from .parser import StarParser
from .__version__ import __version__
from .matching_utils import compute_extraction_box

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
    >>> scores = np.random.rand(100)
    >>> details = np.full((100,), fill_value = -1)
    >>> orientations = Orientations(
    >>>     translations = translations,
    >>>     rotations = rotations,
    >>>     scores = scores,
    >>>     details = details,
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
    translations: np.ndarray
        Array with translations of each orientations (n, d).
    rotations: np.ndarray
        Array with euler angles of each orientation in zxy convention (n, d).
    scores: np.ndarray
        Array with the score of each orientation (n, ).
    details: np.ndarray
        Array with additional orientation details (n, ).
    """

    #: Array with translations of each orientation (n, d).
    translations: np.ndarray

    #: Array with zyz euler angles of each orientation (n, d).
    rotations: np.ndarray

    #: Array with scores of each orientation (n, ).
    scores: np.ndarray

    #: Array with additional details of each orientation(n, ).
    details: np.ndarray

    def __post_init__(self):
        self.translations = np.array(self.translations).astype(np.float32)
        self.rotations = np.array(self.rotations).astype(np.float32)
        self.scores = np.array(self.scores).astype(np.float32)
        self.details = np.array(self.details).astype(np.float32)
        n_orientations = set(
            [
                self.translations.shape[0],
                self.rotations.shape[0],
                self.scores.shape[0],
                self.details.shape[0],
            ]
        )
        if len(n_orientations) != 1:
            raise ValueError(
                "The first dimension of all parameters needs to be of equal length."
            )
        if self.translations.ndim != 2:
            raise ValueError("Expected two dimensional translations parameter.")

        if self.rotations.ndim != 2:
            raise ValueError("Expected two dimensional rotations parameter.")

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
        attributes = (
            "translations",
            "rotations",
            "scores",
            "details",
        )
        kwargs = {attr: getattr(self, attr)[indices].copy() for attr in attributes}
        return self.__class__(**kwargs)

    def copy(self) -> "Orientations":
        """
        Create a copy of the current class instance.

        Returns
        -------
        :py:class:`Orientations`
            Copy of the class instance.
        """
        indices = np.arange(self.scores.size)
        return self[indices]

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

            +---------------+----------------------------------------------------+
            | text          | pytme's standard tab-separated orientations file   |
            +---------------+----------------------------------------------------+
            | star          | Creates a STAR file of orientations                |
            +---------------+----------------------------------------------------+
            | dynamo        | Creates a dynamo table                             |
            +---------------+----------------------------------------------------+

        **kwargs : dict
            Additional keyword arguments specific to the file format.

        Raises
        ------
        ValueError
            If an unsupported file format is specified.
        """
        mapping = {
            "text": self._to_text,
            "star": self._to_star,
            "dynamo": self._to_dynamo_tbl,
        }
        if file_format is None:
            file_format = "text"
            if filename.lower().endswith(".star"):
                file_format = "star"
            elif filename.lower().endswith(".tbl"):
                file_format = "dynamo"

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
        version : str
            Version indicator.
        """
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
            "_rlnClassNumber",
        ]

        target_identifer = "_rlnMicrographName"
        if version == "# version 50001":
            header[3] = "_rlnCenteredCoordinateXAngst"
            header[4] = "_rlnCenteredCoordinateYAngst"
            header[5] = "_rlnCenteredCoordinateZAngst"
            target_identifer = "_rlnTomoName"

        if source_path is not None:
            header.append(target_identifer)

        header.append("_pytmeScore")
        header = "\n".join(header)
        with open(filename, mode="w", encoding="utf-8") as ofile:
            _ = ofile.write(f"# Created using pytme (version {__version__}).\n\n")

            if version is not None:
                _ = ofile.write(f"{version.strip()}\n\n")

            _ = ofile.write(f"{header}\n")
            for index, (translation, rotation, score, detail) in enumerate(self):
                line = [str(x) for x in translation]
                line.extend([str(x) for x in rotation])
                line.extend([str(detail)])

                if source_path is not None:
                    line.append(source_path)
                line.append(score)

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

            +---------------+----------------------------------------------------+
            | text          | pyTME's standard tab-separated orientations file   |
            +---------------+----------------------------------------------------+
            | star          | Creates a STAR file of orientations                |
            +---------------+----------------------------------------------------+
            | dynamo        | Creates a dynamo table                             |
            +---------------+----------------------------------------------------+

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
            "text": cls._from_text,
            "star": cls._from_star,
            "tbl": cls._from_tbl,
        }
        if file_format is None:
            file_format = "text"

            if filename.lower().endswith(".star"):
                file_format = "star"
            elif filename.lower().endswith(".tbl"):
                file_format = "tbl"

        func = mapping.get(file_format, None)
        if func is None:
            raise ValueError(
                f"{file_format} not implemented. Supported are {','.join(mapping.keys())}."
            )

        translations, rotations, scores, details, *_ = func(filename=filename, **kwargs)
        return cls(
            translations=translations,
            rotations=rotations,
            scores=scores,
            details=details,
        )

    @staticmethod
    def _from_text(
        filename: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read orientations from a text file.

        Parameters
        ----------
        filename : str
            The name of the file from which to read the orientations.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing numpy arrays for translations, rotations, scores,
            and details.

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

        translation = np.vstack(translation)
        rotation = np.vstack(rotation)
        score = np.array(score)
        detail = np.array(detail)

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

        return translation, rotation, score, detail

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

        default = np.zeros(translation.shape[0])

        details = ret.get("_rlnClassNumber", default)
        scores = ret.get("_pytmeScore", default)
        return translation, rotation, scores, details

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

        translations, rotations = np.array(translations), np.array(rotations)
        scores, details = np.array(scores), np.array(details)
        return translations, rotations, scores, details

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
