#!python3
""" Handle template matching orientations and conversion between formats.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import re
from collections import deque
from dataclasses import dataclass
from string import ascii_lowercase
from typing import List, Tuple, Dict

import numpy as np
from scipy.spatial.transform import Rotation


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

    """

    #: Return a numpy array with translations of each orientation (n x d).
    translations: np.ndarray

    #: Return a numpy array with euler angles of each orientation in zxy format (n x d).
    rotations: np.ndarray

    #: Return a numpy array with the score of each orientation (n, ).
    scores: np.ndarray

    #: Return a numpy array with additional orientation details (n, ).
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
        kwargs = {attr: getattr(self, attr)[indices] for attr in attributes}
        return self.__class__(**kwargs)

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
            | text          | pyTME's standard tab-separated orientations file   |
            +---------------+----------------------------------------------------+
            | relion        | Creates a STAR file of orientations                |
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
            "relion": self._to_relion_star,
            "dynamo": self._to_dynamo_tbl,
        }
        if file_format is None:
            file_format = "text"
            if filename.lower().endswith(".star"):
                file_format = "relion"
            elif filename.lower().endswith(".tbl"):
                file_format = "dynamo"

        func = mapping.get(file_format, None)
        if func is None:
            raise ValueError(
                f"{file_format} not implemented. Supported are {','.join(mapping.keys())}."
            )

        return func(filename=filename, **kwargs)

    def _to_text(self, filename: str) -> None:
        """
        Save orientations in a text file format.

        Parameters
        ----------
        filename : str
            The name of the file to save the orientations.

        Notes
        -----
        The file is saved with a header specifying each column: z, y, x, euler_z,
        euler_y, euler_x, score, detail. Each row in the file corresponds to an orientation.
        """
        naming = ascii_lowercase[::-1]
        header = "\t".join(
            [
                *list(naming[: self.translations.shape[1]]),
                *[f"euler_{x}" for x in naming[: self.rotations.shape[1]]],
                "score",
                "detail",
            ]
        )
        with open(filename, mode="w", encoding="utf-8") as ofile:
            _ = ofile.write(f"{header}\n")
            for translation, angles, score, detail in self:
                translation_string = "\t".join([str(x) for x in translation])
                angle_string = "\t".join([str(x) for x in angles])
                _ = ofile.write(
                    f"{translation_string}\t{angle_string}\t{score}\t{detail}\n"
                )
        return None

    def _to_dynamo_tbl(
        self,
        filename: str,
        name_prefix: str = None,
        sampling_rate: float = 1.0,
        subtomogram_size: int = 0,
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
                rotation = Rotation.from_euler("zyx", rotation, degrees=True)
                rotation = rotation.as_euler(seq="xyx", degrees=True)
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
                    *translation[::-1],
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

    def _to_relion_star(
        self,
        filename: str,
        name_prefix: str = None,
        ctf_image: str = None,
        sampling_rate: float = 1.0,
        subtomogram_size: int = 0,
    ) -> None:
        """
        Save orientations in RELION's STAR file format.

        Parameters
        ----------
        filename : str
            The name of the file to save the orientations.
        name_prefix : str, optional
            A prefix to add to the image names in the STAR file.
        ctf_image : str, optional
            Path to CTF or wedge mask RELION.
        sampling_rate : float, optional
            Subtomogram sampling rate in angstrom per voxel
        subtomogram_size : int, optional
            Size of the square shaped subtomogram.

        Notes
        -----
        The file is saved with a standard header used in RELION STAR files.
        Each row in the file corresponds to an orientation.
        """
        optics_header = [
            "# version 30001",
            "data_optics",
            "",
            "loop_",
            "_rlnOpticsGroup",
            "_rlnOpticsGroupName",
            "_rlnSphericalAberration",
            "_rlnVoltage",
            "_rlnImageSize",
            "_rlnImageDimensionality",
            "_rlnImagePixelSize",
        ]
        optics_data = [
            "1",
            "opticsGroup1",
            "2.700000",
            "300.000000",
            str(int(subtomogram_size)),
            "3",
            str(float(sampling_rate)),
        ]
        optics_header = "\n".join(optics_header)
        optics_data = "\t".join(optics_data)

        header = [
            "data_particles",
            "",
            "loop_",
            "_rlnCoordinateX",
            "_rlnCoordinateY",
            "_rlnCoordinateZ",
            "_rlnImageName",
            "_rlnAngleRot",
            "_rlnAngleTilt",
            "_rlnAnglePsi",
            "_rlnOpticsGroup",
        ]
        if ctf_image is not None:
            header.append("_rlnCtfImage")

        ctf_image = "" if ctf_image is None else f"\t{ctf_image}"

        header = "\n".join(header)
        name_prefix = "" if name_prefix is None else name_prefix

        with open(filename, mode="w", encoding="utf-8") as ofile:
            _ = ofile.write(f"{optics_header}\n")
            _ = ofile.write(f"{optics_data}\n")

            _ = ofile.write("\n# version 30001\n")
            _ = ofile.write(f"{header}\n")

            # pyTME uses a zyx data layout
            for index, (translation, rotation, score, detail) in enumerate(self):
                rotation = Rotation.from_euler("zyx", rotation, degrees=True)
                rotation = rotation.as_euler(seq="xyx", degrees=True)

                translation_string = "\t".join([str(x) for x in translation][::-1])
                angle_string = "\t".join([str(x) for x in rotation])
                name = f"{name_prefix}_{index}.mrc"
                _ = ofile.write(
                    f"{translation_string}\t{name}\t{angle_string}\t1{ctf_image}\n"
                )

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
            | relion        | Creates a STAR file of orientations                |
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
            "relion": cls._from_relion_star,
            "tbl": cls._from_tbl,
        }
        if file_format is None:
            file_format = "text"

            if filename.lower().endswith(".star"):
                file_format = "relion"
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
        The text file is expected to have a header and data in columns corresponding to
        z, y, x, euler_z, euler_y, euler_x, score, detail.
        """
        with open(filename, mode="r", encoding="utf-8") as infile:
            data = [x.strip().split("\t") for x in infile.read().split("\n")]

        header = data.pop(0)
        translation, rotation, score, detail = [], [], [], []
        for candidate in data:
            if len(candidate) <= 1:
                continue

            translation.append(
                tuple(
                    candidate[i] for i, x in enumerate(header) if x in ascii_lowercase
                )
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

        return translation, rotation, score, detail

    @staticmethod
    def _parse_star(filename: str, delimiter: str = None) -> Dict:
        pattern = re.compile(r"\s*#.*")
        with open(filename, mode="r", encoding="utf-8") as infile:
            data = infile.read()

        data = deque(filter(lambda line: line and line[0] != "#", data.split("\n")))

        ret, category, block = {}, None, []
        while data:
            line = data.popleft()

            if line.startswith("data") and not line.startswith("_"):
                if category != line and category is not None:
                    headers = list(ret[category].keys())
                    headers = [pattern.sub("", x) for x in headers]
                    ret[category] = {
                        header: list(column)
                        for header, column in zip(headers, zip(*block))
                    }
                    block.clear()
                category = line
                if category not in ret:
                    ret[category] = {}
                continue

            if line.startswith("_"):
                ret[category][line] = []
                continue

            if line.startswith("loop"):
                continue

            line_split = line.split(delimiter)
            if len(line_split):
                block.append(line_split)

        headers = list(ret[category].keys())
        headers = [pattern.sub("", x) for x in headers]
        ret[category] = {
            header: list(column) for header, column in zip(headers, zip(*block))
        }
        return ret

    @classmethod
    def _from_relion_star(
        cls, filename: str, delimiter: str = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ret = cls._parse_star(filename=filename, delimiter=delimiter)
        ret = ret["data_particles"]

        translation = np.vstack(
            (ret["_rlnCoordinateZ"], ret["_rlnCoordinateY"], ret["_rlnCoordinateX"])
        )
        translation = translation.astype(np.float32).T

        rotation = np.vstack(
            (ret["_rlnAngleRot"], ret["_rlnAngleTilt"], ret["_rlnAnglePsi"])
        )
        rotation = rotation.astype(np.float32).T

        rotation = Rotation.from_euler("xyx", rotation, degrees=True)
        rotation = rotation.as_euler(seq="zyx", degrees=True)
        score = np.ones(translation.shape[0])
        detail = np.ones(translation.shape[0]) * 1

        return translation, rotation, score, detail

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
            rotation = Rotation.from_euler(
                "xyx", (peak[6], peak[7], peak[8]), degrees=True
            )
            rotations.append(rotation.as_euler(seq="zyx", degrees=True))
            scores.append(peak[9])
            details.append(-1)
            translations.append((peak[25], peak[24], peak[23]))

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
        right_pad = np.divide(extraction_shape, 2).astype(int)
        left_pad = np.add(right_pad, np.mod(extraction_shape, 2)).astype(int)

        peaks = self.translations.astype(int)
        obs_beg = np.subtract(peaks, left_pad)
        obs_end = np.add(peaks, right_pad)

        obs_beg = np.maximum(obs_beg, 0)
        obs_end = np.minimum(obs_end, target_shape)

        cand_beg = left_pad - np.subtract(peaks, obs_beg)
        cand_end = left_pad + np.subtract(obs_end, peaks)

        subset = self
        if drop_out_of_box:
            stops = np.subtract(cand_end, extraction_shape)
            keep_peaks = (
                np.sum(
                    np.multiply(cand_beg == 0, stops == 0),
                    axis=1,
                )
                == peaks.shape[1]
            )
            n_remaining = keep_peaks.sum()
            if n_remaining == 0:
                print(
                    "No peak remaining after filtering. Started with"
                    f" {peaks.shape[0]} filtered to {n_remaining}."
                    " Consider reducing min_distance, increase num_peaks or use"
                    " a different peak caller."
                )

            cand_beg = cand_beg[keep_peaks,]
            cand_end = cand_end[keep_peaks,]
            obs_beg = obs_beg[keep_peaks,]
            obs_end = obs_end[keep_peaks,]
            subset = self[keep_peaks]

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
