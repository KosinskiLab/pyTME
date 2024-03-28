#!python3
""" CLI to simplify analysing the output of match_template.py.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
from os import getcwd
from os.path import join, abspath
import argparse
from sys import exit
from typing import List, Tuple
from os.path import splitext
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfcinv
from scipy.spatial.transform import Rotation

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

    input_group = parser.add_argument_group("Input")
    output_group = parser.add_argument_group("Output")
    peak_group = parser.add_argument_group("Peak Calling")
    additional_group = parser.add_argument_group("Additional Parameters")

    input_group.add_argument(
        "--input_file",
        required=True,
        nargs="+",
        help="Path to the output of match_template.py.",
    )
    input_group.add_argument(
        "--target_mask",
        required=False,
        type=str,
        help="Path to an optional mask applied to template matching scores.",
    )
    input_group.add_argument(
        "--orientations",
        required=False,
        type=str,
        help="Path to file generated using output_format orientations. Can be filtered "
        "to exclude false-positive peaks. If this file is provided, peak calling "
        "is skipped and corresponding parameters ignored.",
    )

    output_group.add_argument(
        "--output_prefix",
        required=True,
        help="Output filename, extension will be added based on output_format.",
    )
    output_group.add_argument(
        "--output_format",
        choices=["orientations", "alignment", "extraction", "relion", "backmapping"],
        default="orientations",
        help="Available output formats:"
        "orientations (translation, rotation, and score), "
        "alignment (aligned template to target based on orientations), "
        "extraction (extract regions around peaks from targets, i.e. subtomograms), "
        "relion (perform extraction step and generate corresponding star files), "
        "backmapping (map template to target using identified peaks).",
    )

    peak_group.add_argument(
        "--peak_caller",
        choices=list(PEAK_CALLERS.keys()),
        default="PeakCallerScipy",
        help="Peak caller for local maxima identification.",
    )
    peak_group.add_argument(
        "--minimum_score",
        type=float,
        default=None,
        help="Minimum score from which peaks will be considered.",
    )
    peak_group.add_argument(
        "--maximum_score",
        type=float,
        default=None,
        help="Maximum score until which peaks will be considered.",
    )
    peak_group.add_argument(
        "--min_distance",
        type=int,
        default=5,
        help="Minimum distance between peaks.",
    )
    peak_group.add_argument(
        "--min_boundary_distance",
        type=int,
        default=0,
        help="Minimum distance of peaks to target edges.",
    )
    peak_group.add_argument(
        "--mask_edges",
        action="store_true",
        default=False,
        help="Whether candidates should not be identified from scores that were "
        "computed from padded densities. Superseded by min_boundary_distance.",
    )
    peak_group.add_argument(
        "--number_of_peaks",
        type=int,
        default=None,
        required=False,
        help="Upper limit of peaks to call, subject to filtering parameters. Default 1000. "
        "If minimum_score is provided all peaks scoring higher will be reported.",
    )
    peak_group.add_argument(
        "--peak_oversampling",
        type=int,
        default=1,
        help="1 / factor equals voxel precision, e.g. 2 detects half voxel "
        "translations. Useful for matching structures to electron density maps.",
    )

    additional_group.add_argument(
        "--subtomogram_box_size",
        type=int,
        default=None,
        help="Subtomogram box size, by default equal to the centered template. Will be "
        "padded to even values if output_format is relion.",
    )
    additional_group.add_argument(
        "--mask_subtomograms",
        action="store_true",
        default=False,
        help="Whether to mask subtomograms using the template mask. The mask will be "
        "rotated according to determined angles.",
    )
    additional_group.add_argument(
        "--invert_target_contrast",
        action="store_true",
        default=False,
        help="Whether to invert the target contrast.",
    )
    additional_group.add_argument(
        "--wedge_mask",
        type=str,
        default=None,
        help="Path to file used as ctf_mask for output_format relion.",
    )
    additional_group.add_argument(
        "--n_false_positives",
        type=int,
        default=None,
        required=False,
        help="Number of accepted false-positives picks to determine minimum score.",
    )

    args = parser.parse_args()

    if args.wedge_mask is not None:
        args.wedge_mask = abspath(args.wedge_mask)

    if args.output_format == "relion" and args.subtomogram_box_size is not None:
        args.subtomogram_box_size += args.subtomogram_box_size % 2

    if args.orientations is not None:
        args.orientations = Orientations.from_file(
            filename=args.orientations, file_format="text"
        )

    if args.minimum_score is not None or args.n_false_positives is not None:
        args.number_of_peaks = np.iinfo(np.int64).max
    else:
        args.number_of_peaks = 1000

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

    def get_extraction_slices(
        self,
        target_shape,
        extraction_shape,
        drop_out_of_box: bool = False,
        return_orientations: bool = False,
    ) -> "Orientations":
        left_pad = np.divide(extraction_shape, 2).astype(int)
        right_pad = np.add(left_pad, np.mod(extraction_shape, 2)).astype(int)

        obs_start = np.subtract(self.translations, left_pad)
        obs_stop = np.add(self.translations, right_pad)

        cand_start = np.subtract(np.maximum(obs_start, 0), obs_start)
        cand_stop = np.subtract(obs_stop, np.minimum(obs_stop, target_shape))
        cand_stop = np.subtract(extraction_shape, cand_stop)
        obs_start = np.maximum(obs_start, 0)
        obs_stop = np.minimum(obs_stop, target_shape)

        subset = self
        if drop_out_of_box:
            stops = np.subtract(cand_stop, extraction_shape)
            keep_peaks = (
                np.sum(
                    np.multiply(cand_start == 0, stops == 0),
                    axis=1,
                )
                == self.translations.shape[1]
            )
            n_remaining = keep_peaks.sum()
            if n_remaining == 0:
                print(
                    "No peak remaining after filtering. Started with"
                    f" {self.translations.shape[0]} filtered to {n_remaining}."
                    " Consider reducing min_distance, increase num_peaks or use"
                    " a different peak caller."
                )
                exit(-1)

            cand_start = cand_start[keep_peaks,]
            cand_stop = cand_stop[keep_peaks,]
            obs_start = obs_start[keep_peaks,]
            obs_stop = obs_stop[keep_peaks,]
            subset = self[keep_peaks]

        cand_start, cand_stop = cand_start.astype(int), cand_stop.astype(int)
        obs_start, obs_stop = obs_start.astype(int), obs_stop.astype(int)

        candidate_slices = [
            tuple(slice(s, e) for s, e in zip(start_row, stop_row))
            for start_row, stop_row in zip(cand_start, cand_stop)
        ]

        observation_slices = [
            tuple(slice(s, e) for s, e in zip(start_row, stop_row))
            for start_row, stop_row in zip(obs_start, obs_stop)
        ]

        if return_orientations:
            return subset, candidate_slices, observation_slices

        return candidate_slices, observation_slices


def load_template(filepath: str, sampling_rate: NDArray, center: bool = True):
    try:
        template = Density.from_file(filepath)
        center_of_mass = template.center_of_mass(template.data)
        template_is_density = True
    except ValueError:
        template = Structure.from_file(filepath)
        center_of_mass = template.center_of_mass()[::-1]
        template = Density.from_structure(template, sampling_rate=sampling_rate)
        template_is_density = False

    translation = np.zeros_like(center_of_mass)
    if center:
        template, translation = template.centered(0)

    return template, center_of_mass, translation, template_is_density


def merge_outputs(data, filepaths: List[str], args):
    if len(filepaths) == 0:
        return data, 1

    if data[0].ndim != data[2].ndim:
        return data, 1

    from tme.matching_exhaustive import _normalize_under_mask

    def _norm_scores(data, args):
        target_origin, _, sampling_rate, cli_args = data[-1]

        _, template_extension = splitext(cli_args.template)
        ret = load_template(
            filepath=cli_args.template,
            sampling_rate=sampling_rate,
            center=not cli_args.no_centering,
        )
        template, center_of_mass, translation, template_is_density = ret

        if args.mask_edges and args.min_boundary_distance == 0:
            max_shape = np.max(template.shape)
            args.min_boundary_distance = np.ceil(np.divide(max_shape, 2))

        target_mask = 1
        if args.target_mask is not None:
            target_mask = Density.from_file(args.target_mask).data
        elif cli_args.target_mask is not None:
            target_mask = Density.from_file(args.target_mask).data

        mask = np.ones_like(data[0])
        np.multiply(mask, target_mask, out=mask)

        cropped_shape = np.subtract(
            mask.shape, np.multiply(args.min_boundary_distance, 2)
        ).astype(int)
        mask[cropped_shape] = 0
        _normalize_under_mask(template=data[0], mask=mask, mask_intensity=mask.sum())
        return data[0]

    entities = np.zeros_like(data[0])
    data[0] = _norm_scores(data=data, args=args)
    for index, filepath in enumerate(filepaths):
        new_scores = _norm_scores(data=load_pickle(filepath), args=args)
        indices = new_scores > data[0]
        entities[indices] = index + 1
        data[0][indices] = new_scores[indices]

    return data, entities


def main():
    args = parse_args()
    data = load_pickle(args.input_file[0])

    target_origin, _, sampling_rate, cli_args = data[-1]

    _, template_extension = splitext(cli_args.template)
    ret = load_template(
        filepath=cli_args.template,
        sampling_rate=sampling_rate,
        center=not cli_args.no_centering,
    )
    template, center_of_mass, translation, template_is_density = ret

    if args.output_format == "relion" and args.subtomogram_box_size is None:
        new_shape = np.add(template.shape, np.mod(template.shape, 2))
        new_shape = np.repeat(new_shape.max(), new_shape.size).astype(int)
        print(f"Padding template from {template.shape} to {new_shape} for RELION.")
        template.pad(new_shape)

    template_mask = template.empty
    template_mask.data[:] = 1
    if cli_args.template_mask is not None:
        template_mask = Density.from_file(cli_args.template_mask)
        template_mask.pad(template.shape, center=False)
        origin_translation = np.divide(
            np.subtract(template.origin, template_mask.origin), template.sampling_rate
        )
        translation = np.add(translation, origin_translation)

        template_mask = template_mask.rigid_transform(
            rotation_matrix=np.eye(template_mask.data.ndim),
            translation=-translation,
            order=1,
        )

    if args.mask_edges and args.min_boundary_distance == 0:
        max_shape = np.max(template.shape)
        args.min_boundary_distance = np.ceil(np.divide(max_shape, 2))

    # data, entities = merge_outputs(data=data, filepaths=args.input_file[1:], args=args)

    orientations = args.orientations
    if orientations is None:
        translations, rotations, scores, details = [], [], [], []
        # Output is MaxScoreOverRotations
        if data[0].ndim == data[2].ndim:
            scores, offset, rotation_array, rotation_mapping, meta = data

            if args.target_mask is not None:
                target_mask = Density.from_file(args.target_mask)
                scores = scores * target_mask.data

            if args.n_false_positives is not None:
                args.n_false_positives = max(args.n_false_positives, 1)
                cropped_shape = np.subtract(
                    scores.shape, np.multiply(args.min_boundary_distance, 2)
                ).astype(int)

                cropped_shape = tuple(
                    slice(
                        int(args.min_boundary_distance),
                        int(x - args.min_boundary_distance),
                    )
                    for x in scores.shape
                )
                # Rickgauer et al. 2017
                n_correlations = np.size(scores[cropped_shape]) * len(rotation_mapping)
                minimum_score = np.multiply(
                    erfcinv(2 * args.n_false_positives / n_correlations),
                    np.sqrt(2) * np.std(scores[cropped_shape]),
                )
                print(f"Determined minimum score cutoff: {minimum_score}.")
                minimum_score = max(minimum_score, 0)
                args.minimum_score = minimum_score

            peak_caller = PEAK_CALLERS[args.peak_caller](
                number_of_peaks=args.number_of_peaks,
                min_distance=args.min_distance,
                min_boundary_distance=args.min_boundary_distance,
            )
            if args.minimum_score is not None:
                args.number_of_peaks = np.inf

            peak_caller(
                scores,
                rotation_matrix=np.eye(3),
                mask=template.data,
                rotation_mapping=rotation_mapping,
                rotation_array=rotation_array,
                minimum_score=args.minimum_score,
            )
            candidates = peak_caller.merge(
                candidates=[tuple(peak_caller)],
                number_of_peaks=args.number_of_peaks,
                min_distance=args.min_distance,
                min_boundary_distance=args.min_boundary_distance,
            )
            if len(candidates) == 0:
                print("Found no peaks. Consider changing peak calling parameters.")
                exit(-1)

            for translation, _, score, detail in zip(*candidates):
                rotations.append(rotation_mapping[rotation_array[tuple(translation)]])

        else:
            candidates = data
            translation, rotation, *_ = data
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

    if args.minimum_score is not None:
        keep = orientations.scores >= args.minimum_score
        orientations = orientations[keep]

    if args.maximum_score is not None:
        keep = orientations.scores <= args.maximum_score
        orientations = orientations[keep]

    if args.output_format == "orientations":
        orientations.to_file(filename=f"{args.output_prefix}.tsv", file_format="text")
        exit(0)

    target = Density.from_file(cli_args.target)
    if args.invert_target_contrast:
        target.data = np.divide(
            np.subtract(-target.data, target.data.min()),
            np.subtract(target.data.max(), target.data.min()),
        )

    if args.output_format in ("extraction", "relion"):
        if not np.all(np.divide(target.shape, template.shape) > 2):
            print(
                "Target might be too small relative to template to extract"
                " meaningful particles."
                f" Target : {target.shape}, template : {template.shape}."
            )

        extraction_shape = template.shape
        if args.subtomogram_box_size is not None:
            extraction_shape = np.repeat(
                args.subtomogram_box_size, len(extraction_shape)
            )

        orientations, cand_slices, obs_slices = orientations.get_extraction_slices(
            target_shape=target.shape,
            extraction_shape=extraction_shape,
            drop_out_of_box=True,
            return_orientations=True,
        )

        working_directory = getcwd()
        if args.output_format == "relion":
            orientations.to_file(
                filename=f"{args.output_prefix}.star",
                file_format="relion",
                name_prefix=join(working_directory, args.output_prefix),
                ctf_image=args.wedge_mask,
                sampling_rate=target.sampling_rate.max(),
                subtomogram_size=extraction_shape[0],
            )

        observations = np.zeros((len(cand_slices), *extraction_shape))
        slices = zip(cand_slices, obs_slices)
        for idx, (cand_slice, obs_slice) in enumerate(slices):
            observations[idx][:] = np.mean(target.data[obs_slice])
            observations[idx][cand_slice] = target.data[obs_slice]

        for index in range(observations.shape[0]):
            cand_start = [x.start for x in cand_slices[index]]
            out_density = Density(
                data=observations[index],
                sampling_rate=sampling_rate,
                origin=np.multiply(cand_start, sampling_rate),
            )
            if args.mask_subtomograms:
                rotation_matrix = euler_to_rotationmatrix(orientations.rotations[index])
                mask_transfomed = template_mask.rigid_transform(
                    rotation_matrix=rotation_matrix, order=1
                )
                out_density.data = out_density.data * mask_transfomed.data
            out_density.to_file(
                join(working_directory, f"{args.output_prefix}_{index}.mrc")
            )

        exit(0)

    if args.output_format == "backmapping":
        orientations, cand_slices, obs_slices = orientations.get_extraction_slices(
            target_shape=target.shape,
            extraction_shape=template.shape,
            drop_out_of_box=True,
            return_orientations=True,
        )
        ret, template_sum = target.empty, template.data.sum()
        for index in range(len(cand_slices)):
            rotation_matrix = euler_to_rotationmatrix(orientations.rotations[index])

            transformed_template = template.rigid_transform(
                rotation_matrix=rotation_matrix
            )
            transformed_template.data = np.multiply(
                transformed_template.data,
                np.divide(template_sum, transformed_template.data.sum()),
            )
            cand_slice, obs_slice = cand_slices[index], obs_slices[index]
            ret.data[obs_slice] += transformed_template.data[cand_slice]
        ret.to_file(f"{args.output_prefix}_backmapped.mrc")
        exit(0)

    if args.peak_oversampling > 1:
        peak_caller = peak_caller = PEAK_CALLERS[args.peak_caller]()
        if data[0].ndim != data[2].ndim:
            print(
                "Input pickle does not contain template matching scores."
                " Cannot oversample peaks."
            )
            exit(-1)
        orientations.translations = peak_caller.oversample_peaks(
            score_space=data[0],
            translations=orientations.translations,
            oversampling_factor=args.oversampling_factor,
        )

    for index, (translation, angles, *_) in enumerate(orientations):
        rotation_matrix = euler_to_rotationmatrix(angles)
        if template_is_density:
            translation = np.subtract(translation, center_of_mass)
            transformed_template = template.rigid_transform(
                rotation_matrix=rotation_matrix
            )
            new_origin = np.add(target_origin / sampling_rate, translation)
            transformed_template.origin = np.multiply(new_origin, sampling_rate)
        else:
            template = Structure.from_file(cli_args.template)
            new_center_of_mass = np.add(
                np.multiply(translation, sampling_rate), target_origin
            )
            translation = np.subtract(new_center_of_mass, center_of_mass)
            transformed_template = template.rigid_transform(
                translation=translation[::-1],
                rotation_matrix=rotation_matrix[::-1, ::-1],
            )
        # template_extension should contain '.'
        transformed_template.to_file(
            f"{args.output_prefix}_{index}{template_extension}"
        )
        index += 1


if __name__ == "__main__":
    main()
