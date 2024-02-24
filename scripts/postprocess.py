#!python3
""" CLI to simplify analysing the output of match_template.py.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
from os import getcwd
from os.path import join
import argparse
from sys import exit
from typing import List, Tuple
from os.path import splitext
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray

from tme import Density, Structure
from tme.analyzer import (
    PeakCallerSort,
    PeakCallerMaximumFilter,
    PeakCallerFast,
    PeakCallerRecursiveMasking,
    PeakCallerScipy,
)
from tme.matching_utils import (
    load_pickle,
    euler_to_rotationmatrix,
    euler_from_rotationmatrix,
    centered_mask,
)

PEAK_CALLERS = {
    "PeakCallerSort": PeakCallerSort,
    "PeakCallerMaximumFilter": PeakCallerMaximumFilter,
    "PeakCallerFast": PeakCallerFast,
    "PeakCallerRecursiveMasking": PeakCallerRecursiveMasking,
    "PeakCallerScipy": PeakCallerScipy,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Peak Calling for Template Matching Outputs"
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the output of match_template.py.",
    )
    parser.add_argument(
        "--output_prefix",
        required=True,
        help="Prefix for the output file name. Extension depends on output_format.",
    )
    parser.add_argument(
        "--number_of_peaks",
        type=int,
        default=1000,
        help="Number of peaks to consider. Note, this is the number of called peaks "
        ", subject to min_distance and min_boundary_distance filtering. Therefore, the "
        "returned number of peaks will be at most equal to number_of_peaks. "
        "Ignored when --orientations is provided.",
    )
    parser.add_argument(
        "--min_distance",
        type=int,
        default=5,
        help="Minimum distance between peaks. Ignored when --orientations is provided.",
    )
    parser.add_argument(
        "--min_boundary_distance",
        type=int,
        default=0,
        help="Minimum distance from target boundaries. Ignored when --orientations "
        "is provided.",
    )
    parser.add_argument(
        "--mask_edges",
        action="store_true",
        default=False,
        help="Whether to mask edges of the input score array according to the template shape."
        "Uses twice the value of --min_boundary_distance if boht are provided.",
    )
    parser.add_argument(
        "--wedge_mask",
        type=str,
        default=None,
        help="Path to Fourier space mask. Only considered if output_format is relion.",
    )
    parser.add_argument(
        "--peak_caller",
        choices=list(PEAK_CALLERS.keys()),
        default="PeakCallerScipy",
        help="Peak caller to use for analysis. Ignored if input_file contains peaks or when "
        "--orientations is provided.",
    )
    parser.add_argument(
        "--orientations",
        default=None,
        help="Path to orientations file to overwrite orientations computed from"
        " match_template.py output.",
    )
    parser.add_argument(
        "--output_format",
        choices=["orientations", "alignment", "extraction", "relion"],
        default="orientations",
        help="Choose the output format. Available formats are: "
        "orientations (translation, rotation, and score), "
        "alignment (aligned template to target based on orientations), "
        "extraction (extract regions around peaks from targets, i.e. subtomograms). "
        "relion (perform extraction step and generate corresponding star files).",
    )
    args = parser.parse_args()

    return args


@dataclass
class Orientations:
    #: Return a numpy array with translations of each orientation (n x d).
    translations: np.ndarray

    #: Return a numpy array with euler angles of each orientation in zxy format (n x d).
    rotations: np.ndarray

    #: Return a numpy array with the score of each orientation (n, ).
    scores: np.ndarray

    #: Return a numpy array with additional orientation details (n, ).
    details: np.ndarray

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

    def to_file(self, filename: str, file_format: type, **kwargs) -> None:
        """
        Save the current class instance to a file in the specified format.

        Parameters
        ----------
        filename : str
            The name of the file where the orientations will be saved.
        file_format : type
            The format in which to save the orientations. Supported formats are 'text' and 'relion'.
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
        }

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
        header = "\t".join(
            ["z", "y", "x", "euler_z", "euler_y", "euler_x", "score", "detail"]
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
    def from_file(cls, filename: str, file_format: type, **kwargs) -> "Orientations":
        """
        Create an instance of :py:class:`Orientations` from a file.

        Parameters
        ----------
        filename : str
            The name of the file from which to read the orientations.
        file_format : type
            The format of the file. Currently, only 'text' format is supported.
        **kwargs : dict
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
        }

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
            _ = data.pop(0)

        translation, rotation, score, detail = [], [], [], []
        for candidate in data:
            if len(candidate) <= 1:
                continue
            if len(candidate) != 8:
                candidate.append(-1)

            candidate = [float(x) for x in candidate]
            translation.append((candidate[0], candidate[1], candidate[2]))
            rotation.append((candidate[3], candidate[4], candidate[5]))
            score.append(candidate[6])
            detail.append(candidate[7])

        translation = np.vstack(translation).astype(int)
        rotation = np.vstack(rotation).astype(float)
        score = np.array(score).astype(float)
        detail = np.array(detail).astype(float)

        return translation, rotation, score, detail


def load_template(filepath: str, sampling_rate: NDArray) -> "Density":
    try:
        template = Density.from_file(filepath)
        template, _ = template.centered(0)
        center_of_mass = template.center_of_mass(template.data)
    except ValueError:
        template = Structure.from_file(filepath)
        center_of_mass = template.center_of_mass()[::-1]
        template = Density.from_structure(template, sampling_rate=sampling_rate)

    return template, center_of_mass


def main():
    args = parse_args()
    data = load_pickle(args.input_file)

    meta = data[-1]
    target_origin, _, sampling_rate, cli_args = meta

    if args.orientations is not None:
        orientations = Orientations.from_file(
            filename=args.orientations, file_format="text"
        )

    else:
        translations, rotations, scores, details = [], [], [], []
        # Output is MaxScoreOverRotations
        if data[0].ndim == data[2].ndim:
            scores, offset, rotation_array, rotation_mapping, meta = data
            if args.mask_edges:
                template, center_of_mass = load_template(
                    cli_args.template, sampling_rate=sampling_rate
                )
                if not cli_args.no_centering:
                    template, *_ = template.centered(0)
                mask_size = template.shape
                if args.min_boundary_distance > 0:
                    mask_size = 2 * args.min_boundary_distance
                scores = centered_mask(
                    scores, np.subtract(scores.shape, mask_size) + 1
                )

            peak_caller = PEAK_CALLERS[args.peak_caller](
                number_of_peaks=args.number_of_peaks,
                min_distance=args.min_distance,
                min_boundary_distance=args.min_boundary_distance,
            )
            peak_caller(scores, rotation_matrix=np.eye(3))
            candidates = peak_caller.merge(
                candidates=[tuple(peak_caller)],
                number_of_peaks=args.number_of_peaks,
                min_distance=args.min_distance,
                min_boundary_distance=args.min_boundary_distance,
            )
            if len(candidates) == 0:
                exit(
                    "Found no peaks. Try reducing min_distance or min_boundary_distance."
                )

            for translation, _, score, detail in zip(*candidates):
                rotations.append(rotation_mapping[rotation_array[tuple(translation)]])

        else:
            candidates = data
            translation, rotation, score, detail, *_ = data
            for i in range(translation.shape[0]):
                rotations.append(euler_from_rotationmatrix(rotation[i]))

        rotations = np.vstack(rotations).astype(float)
        translations, scores, details = candidates[0], candidates[2], candidates[3]
        orientations = Orientations(
            translations=translations,
            rotations=rotations,
            scores=scores,
            details=details,
        )

    if args.output_format == "orientations":
        orientations.to_file(filename=f"{args.output_prefix}.tsv", file_format="text")
        exit(0)

    _, template_extension = splitext(cli_args.template)
    template, center_of_mass = load_template(
        filepath=cli_args.template, sampling_rate=sampling_rate
    )
    template_is_density, index = isinstance(template, Density), 0

    if args.output_format == "relion":
        new_shape = np.add(template.shape, np.mod(template.shape, 2))
        new_shape = np.repeat(new_shape.max(), new_shape.size).astype(int)
        print(f"Padding template from {template.shape} to {new_shape} for RELION.")
        template.pad(new_shape)

    if args.output_format in ("extraction", "relion"):
        target = Density.from_file(cli_args.target)

        if not np.all(np.divide(target.shape, template.shape) > 2):
            print(
                "Target might be too small relative to template to extract"
                " meaningful particles."
                f" Target : {target.shape}, template : {template.shape}."
            )

        peaks = orientations.translations.astype(int)
        max_shape = np.max(template.shape).astype(int)
        half_shape = max_shape // 2

        left_pad = half_shape
        right_pad = np.add(half_shape, max_shape % 2)
        starts = np.subtract(peaks, left_pad)
        stops = np.add(peaks, right_pad)

        candidate_starts = np.maximum(starts, 0).astype(int)
        candidate_stops = np.minimum(stops, target.shape).astype(int)
        keep_peaks = (
            np.sum(
                np.multiply(starts == candidate_starts, stops == candidate_stops),
                axis=1,
            )
            == peaks.shape[1]
        )

        orientations = orientations[keep_peaks]
        working_directory = getcwd()
        if args.output_format == "relion":
            orientations.to_file(
                filename=f"{args.output_prefix}.star",
                file_format="relion",
                name_prefix=join(working_directory, args.output_prefix),
                ctf_image=args.wedge_mask,
                sampling_rate=target.sampling_rate.max(),
                subtomogram_size=template.shape[0],
            )

        peaks = peaks[keep_peaks,]
        starts = starts[keep_peaks,]
        stops = stops[keep_peaks,]
        candidate_starts = candidate_starts[keep_peaks,]
        candidate_stops = candidate_stops[keep_peaks,]

        if not len(peaks):
            print(
                "No peak remaining after filtering. Started with"
                f" {orientations.translations.shape[0]} filtered to {peaks.shape[0]}."
                " Consider reducing min_distance, increase num_peaks or use"
                " a different peak caller."
            )
            exit(-1)

        observation_starts = np.subtract(candidate_starts, starts).astype(int)
        observation_stops = np.subtract(np.add(max_shape, candidate_stops), stops)
        observation_stops = observation_stops.astype(int)

        candidate_slices = [
            tuple(slice(s, e) for s, e in zip(start_row, stop_row))
            for start_row, stop_row in zip(candidate_starts, candidate_stops)
        ]

        observation_slices = [
            tuple(slice(s, e) for s, e in zip(start_row, stop_row))
            for start_row, stop_row in zip(observation_starts, observation_stops)
        ]
        observations = np.zeros(
            (len(candidate_slices), max_shape, max_shape, max_shape)
        )

        slices = zip(candidate_slices, observation_slices)
        for idx, (cand_slice, obs_slice) in enumerate(slices):
            observations[idx][:] = np.mean(target.data[cand_slice])
            observations[idx][obs_slice] = target.data[cand_slice]

        for index in range(observations.shape[0]):
            out_density = Density(
                data=observations[index],
                sampling_rate=sampling_rate,
                origin=candidate_starts[index] * sampling_rate,
            )
            # out_density.data = out_density.data * template_mask.data
            out_density.to_file(
                join(working_directory, f"{args.output_prefix}_{index}.mrc")
            )

        exit(0)

    for translation, angles, *_ in orientations:
        rotation_matrix = euler_to_rotationmatrix(angles)

        if template_is_density:
            translation = np.subtract(translation, center_of_mass)
            transformed_template = template.rigid_transform(
                rotation_matrix=rotation_matrix
            )
            new_origin = np.add(target_origin / sampling_rate, translation)
            transformed_template.origin = np.multiply(new_origin, sampling_rate)
        else:
            new_center_of_mass = np.add(
                np.multiply(translation, sampling_rate), target_origin
            )
            translation = np.subtract(new_center_of_mass, center_of_mass)
            transformed_template = template.rigid_transform(
                translation=translation[::-1],
                rotation_matrix=rotation_matrix[::-1, ::-1],
            )
        # template_extension should contain the extension '.'
        transformed_template.to_file(
            f"{args.output_prefix}_{index}{template_extension}"
        )
        index += 1


if __name__ == "__main__":
    main()
