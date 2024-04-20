""" Implements class Structure to represent atomic structures.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import warnings
from copy import deepcopy
from collections import namedtuple
from typing import List, Dict, Tuple
from itertools import groupby
from dataclasses import dataclass
from os.path import splitext, basename

import numpy as np

from .parser import PDBParser, MMCIFParser
from .matching_utils import (
    rigid_transform,
    _format_mmcif_colunns,
    minimum_enclosing_box,
)
from .helpers import atom_profile
from .types import NDArray


@dataclass(repr=False)
class Structure:
    """
    Represents atomic structures in accordance with the Protein Data Bank (PDB)
    format specification.

    References
    ----------
    .. [1]  https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
    """

    #: Return a numpy array with record types, e.g. ATOM, HETATM.
    record_type: NDArray

    #: Return a numpy array with serial number of each atom.
    atom_serial_number: NDArray

    #: Return a numpy array with name of each atom.
    atom_name: NDArray

    #: Return a numpy array with coordinates of each atom in x, y, z.
    atom_coordinate: NDArray

    #: Return a numpy array with alternate location indicates of each atom.
    alternate_location_indicator: NDArray

    #: Return a numpy array with originating residue names of each atom.
    residue_name: NDArray

    #: Return a numpy array with originating structure chain of each atom.
    chain_identifier: NDArray

    #: Return a numpy array with originating residue id of each atom.
    residue_sequence_number: NDArray

    #: Return a numpy array with insertion information d of each atom.
    code_for_residue_insertion: NDArray

    #: Return a numpy array with occupancy factors of each atom.
    occupancy: NDArray

    #: Return a numpy array with B-factors for each atom.
    temperature_factor: NDArray

    #: Return a numpy array with segment identifier for each atom.
    segment_identifier: NDArray

    #: Return a numpy array with element symbols of each atom.
    element_symbol: NDArray

    #: Return a numpy array with charges of each atom.
    charge: NDArray

    #: Returns a dictionary with class instance metadata.
    details: dict

    def __post_init__(self, *args, **kwargs):
        """
        Initialize the structure and populate header details.

        Raises
        ------
        ValueError
            If other NDArray attributes to not match the number of atoms.
            If the shape of atom_coordinates and chain_identifier doesn't match.
        """
        self._elements = Elements()
        self.details = self._populate_details(self.details)

        n_atoms = self.atom_coordinate.shape[0]
        for attribute in self.__dict__:
            value = getattr(self, attribute)
            if type(value) != np.ndarray:
                continue
            if value.shape[0] != n_atoms:
                raise ValueError(
                    f"Expected shape of {attribute}: {n_atoms}, got {value.shape[0]}."
                )

    def __getitem__(self, indices: List[int]) -> "Structure":
        """
        Get a Structure instance for specified indices.

        Parameters
        ----------
        indices : Union[int, bool, NDArray]
            The indices to get.

        Returns
        -------
        Structure
            The Structure instance for the given indices.
        """
        if type(indices) in (int, bool):
            indices = (indices,)

        indices = np.asarray(indices)
        attributes = (
            "record_type",
            "atom_serial_number",
            "atom_name",
            "atom_coordinate",
            "alternate_location_indicator",
            "residue_name",
            "chain_identifier",
            "residue_sequence_number",
            "code_for_residue_insertion",
            "occupancy",
            "temperature_factor",
            "segment_identifier",
            "element_symbol",
            "charge",
        )
        kwargs = {attr: getattr(self, attr)[indices] for attr in attributes}
        ret = self.__class__(**kwargs, details={})
        return ret

    def __repr__(self):
        """
        Return a string representation of the Structure.

        Returns
        -------
        str
            The string representation.
        """
        unique_chains = "-".join(
            [
                ",".join([str(x) for x in entity])
                for entity in self.details["unique_chains"]
            ]
        )
        min_atom = np.min(self.atom_serial_number)
        max_atom = np.max(self.atom_serial_number)
        n_atom = self.atom_serial_number.size

        min_residue = np.min(self.residue_sequence_number)
        max_residue = np.max(self.residue_sequence_number)
        n_residue = self.residue_sequence_number.size

        repr_str = (
            f"Structure object at {id(self)}\n"
            f"Unique Chains: {unique_chains}, "
            f"Atom Range: {min_atom}-{max_atom} [N = {n_atom}], "
            f"Residue Range: {min_residue}-{max_residue} [N = {n_residue}]"
        )
        return repr_str

    def get_chains(self) -> List[str]:
        """
        Returns a list of available chains.

        Returns
        -------
        list
            The list of available chains.
        """
        return list(self.details["chain_weight"].keys())

    def copy(self) -> "Structure":
        """
        Returns a copy of the Structure instance.

        Returns
        -------
        Structure
            The copied Structure instance.
        """
        return deepcopy(self)

    def _populate_details(self, details: Dict = {}) -> Dict:
        """
        Populate the details dictionary with the data from the Structure instance.

        Parameters
        ----------
        details : dict, optional
            The initial details dictionary, by default {}.

        Returns
        -------
        dict
            The populated details dictionary.
        """
        details["weight"] = np.sum(
            [self._elements[atype].atomic_weight for atype in self.element_symbol]
        )

        label, idx, chain = np.unique(
            self.chain_identifier, return_inverse=True, return_index=True
        )
        chain_weight = np.bincount(
            chain,
            [self._elements[atype].atomic_weight for atype in self.element_symbol],
        )
        labels = self.chain_identifier[idx]
        details["chain_weight"] = {key: val for key, val in zip(labels, chain_weight)}

        # Group non-unique chains in separate lists in details["unique_chains"]
        details["unique_chains"], temp = [], {}
        for chain_label in label:
            index = len(details["unique_chains"])
            chain_sequence = "".join(
                [
                    str(y)
                    for y in self.element_symbol[
                        np.where(self.chain_identifier == chain_label)
                    ]
                ]
            )
            if chain_sequence not in temp:
                temp[chain_sequence] = index
                details["unique_chains"].append([chain_label])
                continue
            idx = temp.get(chain_sequence)
            details["unique_chains"][idx].append(chain_label)

        filtered_data = [
            (label, integer)
            for label, integer in zip(
                self.chain_identifier, self.residue_sequence_number
            )
        ]
        filtered_data = sorted(filtered_data, key=lambda x: x[0])
        details["chain_range"] = {}
        for label, values in groupby(filtered_data, key=lambda x: x[0]):
            values = [int(x[1]) for x in values]
            details["chain_range"][label] = (min(values), max(values))

        return details

    @classmethod
    def from_file(
        cls,
        filename: str,
        keep_non_atom_records: bool = False,
        filter_by_elements: set = None,
        filter_by_residues: set = None,
    ) -> "Structure":
        """
        Reads in an mmcif or pdb file and converts it into class instance.

        Parameters
        ----------
        filename : str
            Path to the mmcif or pdb file.
        keep_non_atom_records : bool, optional
            Wheter to keep residues that are not labelled ATOM.
        filter_by_elements: set, optional
            Which elements to keep. Default corresponds to all elements.
        filter_by_residues: set, optional
            Which residues to keep. Default corresponds to all residues.

        Raises
        ------
        ValueError
            If the extension is not '.pdb' or '.cif'.

        Returns
        -------
        Structure
            Read in structure file.
        """
        _, file_extension = splitext(basename(filename.upper()))
        if file_extension == ".PDB":
            func = cls._load_pdb
        elif file_extension == ".CIF":
            func = cls._load_mmcif
        else:
            raise NotImplementedError(
                "Could not determine structure filetype from extension."
                " Supported filetypes are mmcif (.cif) and pdb (.pdb)."
            )
        data = func(filename)

        keep = np.ones(data["element_symbol"].size, dtype=bool)
        if filter_by_elements:
            keep = np.logical_and(
                keep,
                np.in1d(data["element_symbol"], np.array(list(filter_by_elements))),
            )
        if filter_by_residues:
            keep = np.logical_and(
                keep, np.in1d(data["residue_name"], np.array(list(filter_by_residues)))
            )
        if not keep_non_atom_records:
            keep = np.logical_and(keep, data["record_type"] == "ATOM")

        for key in data:
            if key == "details":
                continue
            if type(data[key]) == np.ndarray:
                data[key] = data[key][keep]
            else:
                data[key] = [x for x, flag in zip(data[key], keep) if flag]

        data["details"]["filepath"] = filename

        return cls(**data)

    @staticmethod
    def _load_mmcif(filename: str) -> Dict:
        """
        Parses a macromolecular Crystallographic Information File (mmCIF)
        and returns the data in a dictionary format.

        Parameters
        ----------
        filename : str
            The filename of the mmCIF to load.

        Returns
        -------
        dict
            A dictionary of numpy arrays. Keys are the names of the PDB
            coordinate section. In addition, some details about the parsed
            structure are included. In case of conversion failure, the failing
            attribute is set to 0 if its supposed to be an integer value.
        """
        result = MMCIFParser(filename)

        atom_site_mapping = {
            "record_type": ("group_PDB", str),
            "atom_serial_number": ("id", int),
            "atom_name": ("label_atom_id", str),
            "alternate_location_indicator": ("label_alt_id", str),
            "residue_name": ("label_comp_id", str),
            # "chain_identifier": ("auth_asym_id", str),
            "chain_identifier": ("label_asym_id", str),
            "residue_sequence_number": ("label_seq_id", int),
            "code_for_residue_insertion": ("pdbx_PDB_ins_code", str),
            "occupancy": ("occupancy", float),
            "temperature_factor": ("B_iso_or_equiv", float),
            "segment_identifier": ("pdbx_PDB_model_num", str),
            "element_symbol": ("type_symbol", str),
            "charge": ("pdbx_formal_charge", str),
        }

        out = {}
        for out_key, (atom_site_key, dtype) in atom_site_mapping.items():
            out_data = [
                x.strip() for x in result["atom_site"].get(atom_site_key, ["."])
            ]
            if dtype == int:
                out_data = [0 if x == "." else int(x) for x in out_data]
            try:
                out[out_key] = np.asarray(out_data).astype(dtype)
            except ValueError:
                default = ["."] if dtype == str else 0
                print(f"Converting {out_key} to {dtype} failed, set to {default}.")
                out[out_key] = np.repeat(default, len(out_data)).astype(dtype)

        number_entries = len(max(out.values(), key=len))
        for key, value in out.items():
            if value.size != 1:
                continue
            out[key] = np.repeat(value, number_entries // value.size)

        out["details"] = {}
        out["atom_coordinate"] = np.transpose(
            np.array(
                [
                    result["atom_site"]["Cartn_x"],
                    result["atom_site"]["Cartn_y"],
                    result["atom_site"]["Cartn_z"],
                ],
                dtype=np.float32,
            )
        )

        detail_mapping = {
            "resolution": ("em_3d_reconstruction", "resolution", np.nan),
            "resolution_method": ("em_3d_reconstruction", "resolution_method", np.nan),
            "method": ("exptl", "method", np.nan),
            "electron_source": ("em_imaging", "electron_source", np.nan),
            "illumination_mode": ("em_imaging", "illumination_mode", np.nan),
            "microscope_model": ("em_imaging", "microscope_model", np.nan),
        }
        for out_key, (base_key, inner_key, default) in detail_mapping.items():
            if base_key not in result:
                continue
            out["details"][out_key] = result[base_key].get(inner_key, default)

        return out

    @staticmethod
    def _load_pdb(filename: str) -> Dict:
        """
        Parses a Protein Data Bank (PDB) file and returns the data
        in a dictionary format.

        Parameters
        ----------
        filename : str
            The filename of the PDB file to load.

        Returns
        -------
        dict
            A dictionary of numpy arrays. Keys are the names of the PDB
            coordinate section. In addition, some details about the parsed
            structure are included. In case of conversion failure, the failing
            attribute is set to 0 if its supposed to be an integer value.
        """
        result = PDBParser(filename)

        atom_site_mapping = {
            "record_type": ("record_type", str),
            "atom_serial_number": ("atom_serial_number", int),
            "atom_name": ("atom_name", str),
            "alternate_location_indicator": ("alternate_location_indicator", str),
            "residue_name": ("residue_name", str),
            "chain_identifier": ("chain_identifier", str),
            "residue_sequence_number": ("residue_sequence_number", int),
            "code_for_residue_insertion": ("code_for_residue_insertion", str),
            "occupancy": ("occupancy", float),
            "temperature_factor": ("temperature_factor", float),
            "segment_identifier": ("segment_identifier", str),
            "element_symbol": ("element_symbol", str),
            "charge": ("charge", str),
        }

        out = {"details": result["details"]}
        for out_key, (inner_key, dtype) in atom_site_mapping.items():
            out_data = [x.strip() for x in result[inner_key]]
            if dtype == int:
                out_data = [0 if x == "." else int(x) for x in out_data]
            try:
                out[out_key] = np.asarray(out_data).astype(dtype)
            except ValueError:
                default = "." if dtype == str else 0
                print(
                    f"Converting {out_key} to {dtype} failed. Setting {out_key} to {default}."
                )
                out[out_key] = np.repeat(default, len(out_data)).astype(dtype)

        out["atom_coordinate"] = np.array(result["atom_coordinate"], dtype=np.float32)

        return out

    def to_file(self, filename: str) -> None:
        """
        Writes the Structure instance data to a Protein Data Bank (PDB) or
        macromolecular Crystallographic Information File (mmCIF) file depending
        one whether filename ends with '.pdb' or '.cif'.

        Raises
        ------
        ValueError
            If the extension is not '.pdb' or '.cif'.

        Parameters
        ----------
        filename : str
            The filename of the file to write.
        """
        data_out = []
        if np.any(np.vectorize(len)(self.chain_identifier) > 2):
            warnings.warn("Chain identifiers longer than one will be shortened.")

        _, file_extension = splitext(basename(filename.upper()))
        if file_extension == ".PDB":
            func = self._write_pdb
        elif file_extension == ".CIF":
            func = self._write_mmcif
        else:
            raise NotImplementedError(
                "Could not determine structure filetype."
                " Supported filetypes are mmcif (.cif) and pdb (.pdb)."
            )

        if self.atom_coordinate.shape[0] > 10**5 and func == self._write_pdb:
            warnings.warn(
                "The structure contains more than 100,000 atoms. Consider using mmcif."
            )

        with open(filename, mode="w", encoding="utf-8") as ofile:
            ofile.writelines(func())

    def _write_pdb(self) -> List[str]:
        """
        Returns a PDB string representation of the structure instance.

        Returns
        -------
        list
            List containing PDB file coordine lines.
        """
        data_out = []
        for index in range(self.atom_coordinate.shape[0]):
            x, y, z = self.atom_coordinate[index, :]
            line = list(" " * 80)
            line[0:6] = f"{self.record_type[index]:<6}"
            line[6:11] = f"{self.atom_serial_number[index]:>5}"
            line[12:16] = f"{self.atom_name[index]:<4}"
            line[16] = f"{self.alternate_location_indicator[index]:<1}"
            line[17:20] = f"{self.residue_name[index]:<3}"
            line[21] = f"{self.chain_identifier[index][0]:<1}"
            line[22:26] = f"{self.residue_sequence_number[index]:>4}"
            line[26] = f"{self.code_for_residue_insertion[index]:<1}"
            line[30:38] = f"{x:>8.3f}"
            line[38:46] = f"{y:>8.3f}"
            line[46:54] = f"{z:>8.3f}"
            line[54:60] = f"{self.occupancy[index]:>6.2f}"
            line[60:66] = f"{self.temperature_factor[index]:>6.2f}"
            line[72:76] = f"{self.segment_identifier[index]:>4}"
            line[76:78] = f"{self.element_symbol[index]:<2}"
            line[78:80] = f"{self.charge[index]:>2}"
            data_out.append("".join(line))
        data_out.append("END")
        data_out = "\n".join(data_out)
        return data_out

    def _write_mmcif(self) -> List[str]:
        """
        Returns a MMCIF string representation of the structure instance.

        Returns
        -------
        list
            List containing MMCIF file coordinate lines.
        """
        model_num, entity_id = 1, 1
        data = {
            "group_PDB": [],
            "id": [],
            "type_symbol": [],
            "label_atom_id": [],
            "label_alt_id": [],
            "label_comp_id": [],
            "label_asym_id": [],
            "label_entity_id": [],
            "label_seq_id": [],
            "pdbx_PDB_ins_code": [],
            "Cartn_x": [],
            "Cartn_y": [],
            "Cartn_z": [],
            "occupancy": [],
            "B_iso_or_equiv": [],
            "pdbx_formal_charge": [],
            "auth_seq_id": [],
            "auth_comp_id": [],
            "auth_asym_id": [],
            "auth_atom_id": [],
            "pdbx_PDB_model_num": [],
        }

        for index in range(self.atom_coordinate.shape[0]):
            x, y, z = self.atom_coordinate[index, :]
            data["group_PDB"].append(self.record_type[index])
            data["id"].append(str(self.atom_serial_number[index]))
            data["type_symbol"].append(self.element_symbol[index])
            data["label_atom_id"].append(self.atom_name[index])
            data["label_alt_id"].append(self.alternate_location_indicator[index])
            data["label_comp_id"].append(self.residue_name[index])
            data["label_asym_id"].append(self.chain_identifier[index][0])
            data["label_entity_id"].append(str(entity_id))
            data["label_seq_id"].append(str(self.residue_sequence_number[index]))
            data["pdbx_PDB_ins_code"].append(self.code_for_residue_insertion[index])
            data["Cartn_x"].append(f"{x:.3f}")
            data["Cartn_y"].append(f"{y:.3f}")
            data["Cartn_z"].append(f"{z:.3f}")
            data["occupancy"].append(f"{self.occupancy[index]:.2f}")
            data["B_iso_or_equiv"].append(f"{self.temperature_factor[index]:.2f}")
            data["pdbx_formal_charge"].append(self.charge[index])
            data["auth_seq_id"].append(str(self.residue_sequence_number[index]))
            data["auth_comp_id"].append(self.residue_name[index])
            data["auth_asym_id"].append(self.chain_identifier[index][0])
            data["auth_atom_id"].append(self.atom_name[index])
            data["pdbx_PDB_model_num"].append(str(model_num))

        output_data = {"atom_site": data}
        original_file = self.details.get("filepath", "")
        try:
            new_data = {k: v for k, v in MMCIFParser(original_file).items()}
            index = self.atom_serial_number - 1
            new_data["atom_site"] = {
                k: [v[i] for i in index] for k, v in new_data["atom_site"].items()
            }
            new_data["atom_site"]["Cartn_x"] = data["Cartn_x"]
            new_data["atom_site"]["Cartn_y"] = data["Cartn_y"]
            new_data["atom_site"]["Cartn_z"] = data["Cartn_z"]
            output_data = new_data
        except Exception:
            pass

        ret = ""
        for category, subdict in output_data.items():
            ret += "#\n"
            is_loop = isinstance(subdict[list(subdict.keys())[0]], list)
            if not is_loop:
                for k in subdict:
                    ret += f"_{category}.{k}\t{subdict[k]}\n"
            else:
                ret += "loop_\n"
                ret += "".join([f"_{category}.{k}\n" for k in subdict])
                padded_subdict = _format_mmcif_colunns(subdict)

                data = [
                    "".join([str(x) for x in content])
                    for content in zip(*padded_subdict.values())
                ]
                ret += "\n".join([entry for entry in data]) + "\n"

        return ret

    def subset_by_chain(self, chain: str = None) -> "Structure":
        """
        Return a subset of the structure that contains only atoms belonging to
        a specific chain. If no chain is specified, all chains are returned.

        Parameters
        ----------
        chain : str, optional
            The chain identifier. If multiple chains should be selected they need
            to be a comma separated string, e.g. 'A,B,CE'. If chain None,
            all chains are returned. Default is None.

        Returns
        -------
        Structure
            A subset of the original structure containing only the specified chain.
        """
        chain = np.unique(self.chain_identifier) if chain is None else chain.split(",")
        keep = np.in1d(self.chain_identifier, chain)
        return self[keep]

    def subset_by_range(
        self,
        start: int,
        stop: int,
        chain: str = None,
    ) -> "Structure":
        """
        Return a subset of the structure within a specific range of residues.

        Parameters
        ----------
        start : int
            The starting residue sequence number.

        stop : int
            The ending residue sequence number.

        chain : str, optional
            The chain identifier. If multiple chains should be selected they need
            to be a comma separated string, e.g. 'A,B,CE'. If chain None,
            all chains are returned. Default is None.

        Returns
        -------
        Structure
            A subset of the original structure within the specified residue range.
        """
        ret = self.subset_by_chain(chain=chain)
        keep = np.logical_and(
            ret.residue_sequence_number >= start, ret.residue_sequence_number <= stop
        )
        return ret[keep]

    def center_of_mass(self) -> NDArray:
        """
        Calculate the center of mass of the structure.

        Returns
        -------
        NDArray
            The center of mass of the structure.
        """
        weights = [self._elements[atype].atomic_weight for atype in self.element_symbol]
        return np.dot(self.atom_coordinate.T, weights) / np.sum(weights)

    def rigid_transform(
        self,
        rotation_matrix: NDArray,
        translation: NDArray,
        use_geometric_center: bool = False,
    ) -> "Structure":
        """
        Performs a rigid transform of internal structure coordinates.

        Parameters
        ----------
        rotation_matrix : NDArray
            The rotation matrix to apply to the coordinates.
        translation : NDArray
            The vector to translate the coordinates by.
        use_geometric_center : bool, optional
            Whether to use geometric or coordinate center.

        Returns
        -------
        Structure
            The transformed instance of :py:class:`tme.structure.Structure`.
        """
        out = np.empty_like(self.atom_coordinate.T)
        rigid_transform(
            coordinates=self.atom_coordinate.T,
            rotation_matrix=rotation_matrix,
            translation=translation,
            out=out,
            use_geometric_center=use_geometric_center,
        )
        ret = self.copy()
        ret.atom_coordinate = out.T.copy()
        return ret

    def centered(self) -> Tuple["Structure", NDArray]:
        """
        Shifts the structure analogous to :py:meth:`tme.density.Density.centered`.

        Returns
        -------
        Structure
            A copy of the class instance whose data center of mass is in the
            center of the data array.
        NDArray
            The coordinate translation.

        See Also
        --------
        :py:meth:`tme.Density.centered`
        """
        center_of_mass = self.center_of_mass()
        enclosing_box = minimum_enclosing_box(coordinates=self.atom_coordinate.T)
        shift = np.subtract(np.divide(enclosing_box, 2), center_of_mass)

        transformed_structure = self.rigid_transform(
            translation=shift, rotation_matrix=np.eye(shift.size)
        )

        return transformed_structure, shift

    def _coordinate_to_position(
        self,
        shape: Tuple[int],
        sampling_rate: Tuple[float],
        origin: Tuple[float],
    ) -> (NDArray, Tuple[str], Tuple[int], float, Tuple[float]):
        """
        Converts coordinates to positions.

        Parameters
        ----------
        shape : Tuple[int,]
            The desired shape of the output array.

        sampling_rate : float
            The sampling rate of the output array in unit of self.atom_coordinate.

        origin : Tuple[float,]
            The origin of the coordinate system.
        Returns
        -------
        Tuple[NDArray, List[str], Tuple[int, ], float, Tuple[float,]]
            Returns positions, atom_types, shape, sampling_rate, and origin.
        """
        coordinates = self.atom_coordinate.copy()
        atom_types = self.element_symbol.copy()

        # positions are in x, y, z map is z, y, x
        coordinates = coordinates[:, ::-1]

        sampling_rate = 1 if sampling_rate is None else sampling_rate
        adjust_origin = origin is not None and shape is None
        origin = coordinates.min(axis=0) if origin is None else origin
        positions = (coordinates - origin) / sampling_rate
        positions = np.rint(positions).astype(int)

        if adjust_origin:
            left_shift = positions.min(axis=0)
            positions -= left_shift
            shape = positions.max(axis=0) + 1
            origin = origin + np.multiply(left_shift, sampling_rate)

        if shape is None:
            shape = positions.max(axis=0) + 1

        valid_positions = np.sum(
            np.logical_and(positions < shape, positions >= 0), axis=1
        )

        positions = positions[valid_positions == positions.shape[1], :]
        atom_types = atom_types[valid_positions == positions.shape[1]]

        self.details["nAtoms_outOfBound"] = 0
        if positions.shape[0] != coordinates.shape[0]:
            out_of_bounds = coordinates.shape[0] - positions.shape[0]
            print(f"{out_of_bounds}/{coordinates.shape[0]} atoms were out of bounds.")
            self.details["nAtoms_outOfBound"] = out_of_bounds

        return positions, atom_types, shape, sampling_rate, origin

    def _position_to_vdw_sphere(
        self,
        positions: Tuple[float],
        atoms: Tuple[str],
        sampling_rate: Tuple[float],
        volume: NDArray,
    ) -> None:
        """
        Updates a volume with van der Waals spheres.

        Parameters
        ----------
        positions : Tuple[float, float, float]
            The positions of the atoms.

        atoms : Tuple[str]
            The types of the atoms.

        sampling_rate : float
            The desired sampling rate in unit of self.atom_coordinate of the
            output array.

        volume : NDArray
            The volume to update.
        """
        index_dict, vdw_rad, shape = {}, {}, volume.shape
        for atom_index, atom_position in enumerate(positions):
            atom_type = atoms[atom_index]
            if atom_type not in index_dict.keys():
                atom_vdwr = np.ceil(
                    np.divide(self._elements[atom_type].vdwr, (sampling_rate * 100))
                ).astype(int)

                vdw_rad[atom_type] = atom_vdwr
                atom_slice = tuple(slice(-k, k + 1) for k in atom_vdwr)
                distances = np.linalg.norm(
                    np.divide(
                        np.mgrid[atom_slice],
                        atom_vdwr.reshape((-1,) + (1,) * volume.ndim),
                    ),
                    axis=0,
                )
                index_dict[atom_type] = (distances <= 1).astype(volume.dtype)

            footprint = index_dict[atom_type]
            start = np.maximum(np.subtract(atom_position, vdw_rad[atom_type]), 0)
            stop = np.minimum(np.add(atom_position, vdw_rad[atom_type]) + 1, shape)
            volume_slice = tuple(slice(*coord) for coord in zip(start, stop))

            start_index = np.maximum(-np.subtract(atom_position, vdw_rad[atom_type]), 0)
            stop_index = np.add(
                footprint.shape,
                np.minimum(
                    np.subtract(shape, np.add(atom_position, vdw_rad[atom_type]) + 1), 0
                ),
            )
            index_slice = tuple(slice(*coord) for coord in zip(start_index, stop_index))
            volume[volume_slice] += footprint[index_slice]

    def _position_to_scattering_factors(
        self,
        positions: NDArray,
        atoms: NDArray,
        sampling_rate: NDArray,
        volume: NDArray,
        lowpass_filter: bool = True,
        downsampling_factor: float = 1.35,
        source: str = "peng1995",
    ) -> None:
        """
        Updates a volume with scattering factors.

        Parameters
        ----------
        positions : NDArray
            The positions of the atoms.
        atoms : NDArray
            Element symbols.
        sampling_rate : float
            Sampling rate that was used to convert coordinates to positions.
        volume : NDArray
            The volume to update.
        lowpass_filter : NDArray
            Whether the scattering factors hsould be lowpass filtered.
        downsampling_factor : NDArray
            Downsampling factor for scattering factor computation.
        source : str
            Which scattering factors to use

        Reference
        ---------
        https://github.com/I2PC/xmipp.
        """
        scattering_profiles, shape = dict(), volume.shape
        for atom_index, point in enumerate(positions):
            if atoms[atom_index] not in scattering_profiles:
                spline = atom_profile(
                    atom=atoms[atom_index],
                    M=downsampling_factor,
                    method=source,
                    lfilter=lowpass_filter,
                )
                scattering_profiles.update({atoms[atom_index]: spline})

            atomic_radius = np.divide(
                self._elements[atoms[atom_index]].vdwr, sampling_rate * 100
            )
            starts = np.maximum(np.ceil(point - atomic_radius), 0).astype(int)
            stops = np.minimum(np.floor(point + atomic_radius), shape).astype(int)

            grid_index = np.meshgrid(
                *[range(start, stop) for start, stop in zip(starts, stops)]
            )
            distances = np.einsum(
                "aijk->ijk",
                np.array([(grid_index[i] - point[i]) ** 2 for i in range(len(point))]),
                dtype=np.float64,
            )
            distances = np.sqrt(distances)
            if not len(distances):
                grid_index, distances = point, 0
            np.add.at(
                volume,
                tuple(grid_index),
                scattering_profiles[atoms[atom_index]](distances),
            )

    def _get_atom_weights(
        self, atoms: Tuple[str] = None, weight_type: str = "atomic_weight"
    ) -> Tuple[float]:
        """
        Returns weights of individual atoms according to a specified weight type.

        Parameters
        ----------
        atoms : Tuple of strings, optional
            The atoms to get the weights for. If None, weights for all atoms
            are used. Default is None.

        weight_type : str, optional
            The type of weights to return. This can either be 'atomic_weight',
            'atomic_number', or 'van_der_waals_radius'. Default is 'atomic_weight'.

        Returns
        -------
        List[float]
            A list containing the weights of the atoms.
        """
        atoms = self.element_symbol if atoms is None else atoms
        match weight_type:
            case "atomic_weight":
                weight = [self._elements[atom].atomic_weight for atom in atoms]
            case "atomic_number":
                weight = [self._elements[atom].atomic_number for atom in atoms]
            case _:
                raise NotImplementedError(
                    "weight_type can either be 'atomic_weight' or 'atomic_number'"
                )
        return weight

    def to_volume(
        self,
        shape: Tuple[int] = None,
        sampling_rate: NDArray = None,
        origin: Tuple[float] = None,
        chain: str = None,
        weight_type: str = "atomic_weight",
        scattering_args: Dict = dict(),
    ) -> Tuple[NDArray, Tuple[int], NDArray]:
        """
        Converts atom coordinates of shape [n x 3] x, y, z to a volume with
        index z, y, x.

        Parameters
        ----------
        shape : Tuple[int, ...], optional
            Desired shape of the output array. If shape is given its expected to be
            in z, y, x form.
        sampling_rate : float, optional
            Sampling rate of the output array in the unit of self.atom_coordinate
        origin : Tuple[float, ...], optional
            Origin of the coordinate system. If origin is given its expected to be
            in z, y, x form.
        chain : str, optional
            The chain identifier. If multiple chains should be selected they need
            to be a comma separated string, e.g. 'A,B,CE'. If chain None,
            all chains are returned. Default is None.
        weight_type : str, optional
            Which weight should be given to individual atoms.
        scattering_args : dict, optional
            Additional arguments for scattering factor computation.

        Returns
        -------
        Tuple[NDArray, Tuple[int], NDArray]
            The volume, its origin and the voxel size in Ångstrom.
        """
        _weight_types = {
            "atomic_weight",
            "atomic_number",
            "van_der_waals_radius",
            "scattering_factors",
            "lowpass_scattering_factors",
        }
        _weight_string = ",".join([f"'{x}'" for x in _weight_types])
        if weight_type not in _weight_types:
            raise NotImplementedError(f"weight_type needs to be in {_weight_string}")

        if sampling_rate is None:
            sampling_rate = np.ones(self.atom_coordinate.shape[1])
        sampling_rate = np.array(sampling_rate)
        if sampling_rate.size == 1:
            sampling_rate = np.repeat(sampling_rate, self.atom_coordinate.shape[1])
        elif sampling_rate.size != self.atom_coordinate.shape[1]:
            raise ValueError(
                "sampling_rate should either be single value of array with"
                f"size {self.atom_coordinate.shape[1]}."
            )
        if "source" not in scattering_args:
            scattering_args["source"] = "peng1995"

        temp = self.subset_by_chain(chain=chain)

        positions, atoms, shape, sampling_rate, origin = temp._coordinate_to_position(
            shape=shape, sampling_rate=sampling_rate, origin=origin
        )
        volume = np.zeros(shape, dtype=np.float32)
        if weight_type in ("atomic_weight", "atomic_number"):
            weights = temp._get_atom_weights(atoms=atoms, weight_type=weight_type)
            np.add.at(volume, tuple(positions.T), weights)
        elif weight_type == "van_der_waals_radius":
            self._position_to_vdw_sphere(positions, atoms, sampling_rate, volume)
        elif weight_type == "scattering_factors":
            self._position_to_scattering_factors(
                positions,
                atoms,
                sampling_rate,
                volume,
                lowpass_filter=False,
                **scattering_args,
            )
        elif weight_type == "lowpass_scattering_factors":
            self._position_to_scattering_factors(
                positions,
                atoms,
                sampling_rate,
                volume,
                lowpass_filter=True,
                **scattering_args,
            )

        self.details.update(temp.details)
        return volume, origin, sampling_rate

    @classmethod
    def compare_structures(
        cls,
        structure1: "Structure",
        structure2: "Structure",
        origin: NDArray = None,
        sampling_rate: float = None,
        weighted: bool = False,
    ) -> float:
        """
        Compute root mean square deviation (RMSD) between two structures.

        Both structures need to have the same number of atoms. In practice, this means
        that *structure2* is a transformed version of *structure1*

        Parameters
        ----------
        structure1 : Structure
            Structure 1.

        structure2 : Structure
            Structure 2.

        origin : NDArray, optional
            Origin of the structure coordinate system.

        sampling_rate : float, optional
            Sampling rate if discretized on a grid in the unit of self.atom_coordinate.

        weighted : bool, optional
            Whether atoms should be weighted by their atomic weight.

        Returns
        -------
        float
            Root Mean Square Deviation (RMSD)
        """
        if origin is None:
            origin = np.zeros(structure1.atom_coordinate.shape[1])

        coordinates1 = structure1.atom_coordinate
        coordinates2 = structure2.atom_coordinate
        atoms1, atoms2 = structure1.element_symbol, structure2.element_symbol
        if sampling_rate is not None:
            coordinates1 = np.rint((coordinates1 - origin) / sampling_rate).astype(int)
            coordinates2 = np.rint((coordinates2 - origin) / sampling_rate).astype(int)

        weights1 = np.array(structure1._get_atom_weights(atoms=atoms1))
        weights2 = np.array(structure2._get_atom_weights(atoms=atoms2))
        if not weighted:
            weights1 = np.ones_like(weights1)
            weights2 = np.ones_like(weights2)

        if not np.allclose(coordinates1.shape, coordinates2.shape):
            raise ValueError(
                "Input structures need to have the same number of coordinates."
            )
        if not np.allclose(weights1.shape, weights2.shape):
            raise ValueError("Input structures need to have the same number of atoms.")

        squared_diff = np.sum(np.square(coordinates1 - coordinates2), axis=1)
        weighted_quared_diff = squared_diff * ((weights1 + weights2) / 2)
        rmsd = np.sqrt(np.mean(weighted_quared_diff))

        return rmsd

    @classmethod
    def align_structures(
        cls,
        structure1: "Structure",
        structure2: "Structure",
        origin: NDArray = None,
        sampling_rate: float = None,
        weighted: bool = False,
    ) -> Tuple["Structure", float]:
        """
        Align the atom coordinates of structure2 to structure1 using
        the Kabsch algorithm.

        Both structures need to have the same number of atoms. In practice, this means
        that *structure2* is a subset of *structure1*

        Parameters
        ----------
        structure1 : Structure
            Structure 1.

        structure2 : Structure
            Structure 2.

        origin : NDArray, optional
            Origin of the structure coordinate system.

        sampling_rate : float, optional
            Voxel size if discretized on a grid.

        weighted : bool, optional
            Whether atoms should be weighted by their atomic weight.

        Returns
        -------
        Structure
            *structure2* aligned to *structure1*.
        float
            Root Mean Square Error (RMSE)
        """
        if origin is None:
            origin = np.minimum(
                structure1.atom_coordinate.min(axis=0),
                structure2.atom_coordinate.min(axis=0),
            ).astype(int)

        initial_rmsd = cls.compare_structures(
            structure1=structure1,
            structure2=structure2,
            origin=origin,
            sampling_rate=sampling_rate,
            weighted=weighted,
        )

        reference = structure1.atom_coordinate.copy()
        query = structure2.atom_coordinate.copy()
        if sampling_rate is not None:
            reference, atoms1, shape, _, _ = structure1._coordinate_to_position(
                shape=None, sampling_rate=sampling_rate, origin=origin
            )
            query, atoms2, shape, _, _ = structure2._coordinate_to_position(
                shape=None, sampling_rate=sampling_rate, origin=origin
            )

        reference_mean = reference.mean(axis=0)
        query_mean = query.mean(axis=0)

        reference = reference - reference_mean
        query = query - query_mean

        corr = np.dot(query.T, reference)
        U, S, Vh = np.linalg.svd(corr)

        rotation = np.dot(Vh.T, U.T).T
        if np.linalg.det(rotation) < 0:
            Vh[2, :] *= -1
            rotation = np.dot(Vh.T, U.T).T

        translation = reference_mean - np.dot(query_mean, rotation)

        temp = structure1.copy()
        temp.atom_coordinate = reference + reference_mean
        ret = structure2.copy()
        ret.atom_coordinate = np.dot(query + query_mean, rotation) + translation

        final_rmsd = cls.compare_structures(
            structure1=temp,
            structure2=ret,
            origin=origin,
            sampling_rate=None,
            weighted=weighted,
        )

        print(f"Initial RMSD: {initial_rmsd:.5f} - Final RMSD: {final_rmsd:.5f}")

        return ret, final_rmsd


@dataclass(frozen=True, repr=True)
class Elements:
    """
    Lookup table containing information on chemical elements.
    """

    Atom = namedtuple(
        "Atom",
        [
            "atomic_number",
            "atomic_radius",
            "lattice_constant",
            "lattice_structure",
            "vdwr",
            "covalent_radius_bragg",
            "atomic_weight",
        ],
    )
    _default = Atom(0, 0, 0, "Atom does not exist in ressource.", 0, 0, 0)
    _elements = {
        "H": Atom(1, 25, 3.75, "HEX", 110, np.nan, 1.008),
        "HE": Atom(2, 120, 3.57, "HEX", 140, np.nan, 4.002602),
        "LI": Atom(3, 145, 3.49, "BCC", 182, 150, 6.94),
        "BE": Atom(4, 105, 2.29, "HEX", 153, 115, 9.0121831),
        "B": Atom(5, 85, 8.73, "TET", 192, np.nan, 10.81),
        "C": Atom(6, 70, 3.57, "DIA", 170, 77, 12.011),
        "N": Atom(7, 65, 4.039, "HEX", 155, 65, 14.007),
        "O": Atom(8, 60, 6.83, "CUB", 152, 65, 15.999),
        "F": Atom(9, 50, np.nan, "MCL", 147, 67, 18.998403163),
        "NE": Atom(10, 160, 4.43, "FCC", 154, np.nan, 20.1797),
        "NA": Atom(11, 180, 4.23, "BCC", 227, 177, 22.98976928),
        "MG": Atom(12, 150, 3.21, "HEX", 173, 142, 24.305),
        "AL": Atom(13, 125, 4.05, "FCC", 184, 135, 26.9815385),
        "SI": Atom(14, 110, 5.43, "DIA", 210, 117, 28.085),
        "P": Atom(15, 100, 7.17, "CUB", 180, np.nan, 30.973761998),
        "S": Atom(16, 100, 10.47, "ORC", 180, 102, 32.06),
        "CL": Atom(17, 100, 6.24, "ORC", 175, 105, 35.45),
        "AR": Atom(18, 71, 5.26, "FCC", 188, np.nan, 39.948),
        "K": Atom(19, 220, 5.23, "BCC", 275, 207, 39.0983),
        "CA": Atom(20, 180, 5.58, "FCC", 231, 170, 40.078),
        "SC": Atom(21, 160, 3.31, "HEX", 215, np.nan, 44.955908),
        "TI": Atom(22, 140, 2.95, "HEX", 211, 140, 47.867),
        "V": Atom(23, 135, 3.02, "BCC", 207, np.nan, 50.9415),
        "CR": Atom(24, 140, 2.88, "BCC", 206, 140, 51.9961),
        "MN": Atom(25, 140, 8.89, "CUB", 205, 147, 54.938044),
        "FE": Atom(26, 140, 2.87, "BCC", 204, 140, 55.845),
        "CO": Atom(27, 135, 2.51, "HEX", 200, 137, 58.933194),
        "NI": Atom(28, 135, 3.52, "FCC", 197, 135, 58.6934),
        "CU": Atom(29, 135, 3.61, "FCC", 196, 137, 63.546),
        "ZN": Atom(30, 135, 2.66, "HEX", 201, 132, 65.38),
        "GA": Atom(31, 130, 4.51, "ORC", 187, np.nan, 69.723),
        "GE": Atom(32, 125, 5.66, "DIA", 211, np.nan, 72.63),
        "AS": Atom(33, 115, 4.13, "RHL", 185, 126, 74.921595),
        "SE": Atom(34, 115, 4.36, "HEX", 190, 117, 78.971),
        "BR": Atom(35, 115, 6.67, "ORC", 185, 119, 79.904),
        "KR": Atom(36, np.nan, 5.72, "FCC", 202, np.nan, 83.798),
        "RB": Atom(37, 235, 5.59, "BCC", 303, 225, 85.4678),
        "SR": Atom(38, 200, 6.08, "FCC", 249, 195, 87.62),
        "Y": Atom(39, 180, 3.65, "HEX", 232, np.nan, 88.90584),
        "ZR": Atom(40, 155, 3.23, "HEX", 223, np.nan, 91.224),
        "NB": Atom(41, 145, 3.3, "BCC", 218, np.nan, 92.90637),
        "MO": Atom(42, 145, 3.15, "BCC", 217, np.nan, 95.95),
        "TC": Atom(43, 135, 2.74, "HEX", 216, np.nan, 97.90721),
        "RU": Atom(44, 130, 2.7, "HEX", 213, np.nan, 101.07),
        "RH": Atom(45, 135, 3.8, "FCC", 210, np.nan, 102.9055),
        "PD": Atom(46, 140, 3.89, "FCC", 210, np.nan, 106.42),
        "AG": Atom(47, 160, 4.09, "FCC", 211, 177, 107.8682),
        "CD": Atom(48, 155, 2.98, "HEX", 218, 160, 112.414),
        "IN": Atom(49, 155, 4.59, "TET", 193, np.nan, 114.818),
        "SN": Atom(50, 145, 5.82, "TET", 217, 140, 118.71),
        "SB": Atom(51, 145, 4.51, "RHL", 206, 140, 121.76),
        "TE": Atom(52, 140, 4.45, "HEX", 206, 133, 127.6),
        "I": Atom(53, 140, 7.72, "ORC", 198, 140, 126.90447),
        "XE": Atom(54, np.nan, 6.2, "FCC", 216, np.nan, 131.293),
        "CS": Atom(55, 260, 6.05, "BCC", 343, 237, 132.90545196),
        "BA": Atom(56, 215, 5.02, "BCC", 268, 210, 137.327),
        "LA": Atom(57, 195, 3.75, "HEX", 243, np.nan, 138.90547),
        "CE": Atom(58, 185, 5.16, "FCC", 242, np.nan, 140.116),
        "PR": Atom(59, 185, 3.67, "HEX", 240, np.nan, 140.90766),
        "ND": Atom(60, 185, 3.66, "HEX", 239, np.nan, 144.242),
        "PM": Atom(61, 185, np.nan, "", 238, np.nan, 144.91276),
        "SM": Atom(62, 185, 9, "RHL", 236, np.nan, 150.36),
        "EU": Atom(63, 185, 4.61, "BCC", 235, np.nan, 151.964),
        "GD": Atom(64, 180, 3.64, "HEX", 234, np.nan, 157.25),
        "TB": Atom(65, 175, 3.6, "HEX", 233, np.nan, 158.92535),
        "DY": Atom(66, 175, 3.59, "HEX", 231, np.nan, 162.5),
        "HO": Atom(67, 175, 3.58, "HEX", 230, np.nan, 164.93033),
        "ER": Atom(68, 175, 3.56, "HEX", 229, np.nan, 167.259),
        "TM": Atom(69, 175, 3.54, "HEX", 227, np.nan, 168.93422),
        "YB": Atom(70, 175, 5.49, "FCC", 226, np.nan, 173.045),
        "LU": Atom(71, 175, 3.51, "HEX", 224, np.nan, 174.9668),
        "HF": Atom(72, 155, 3.2, "HEX", 223, np.nan, 178.49),
        "TA": Atom(73, 145, 3.31, "BCC", 222, np.nan, 180.94788),
        "W": Atom(74, 135, 3.16, "BCC", 218, np.nan, 183.84),
        "RE": Atom(75, 135, 2.76, "HEX", 216, np.nan, 186.207),
        "OS": Atom(76, 130, 2.74, "HEX", 216, np.nan, 190.23),
        "IR": Atom(77, 135, 3.84, "FCC", 213, np.nan, 192.217),
        "PT": Atom(78, 135, 3.92, "FCC", 213, np.nan, 195.084),
        "AU": Atom(79, 135, 4.08, "FCC", 214, np.nan, 196.966569),
        "HG": Atom(80, 150, 2.99, "RHL", 223, np.nan, 200.592),
        "TL": Atom(81, 190, 3.46, "HEX", 196, 190, 204.38),
        "PB": Atom(82, 180, 4.95, "FCC", 202, np.nan, 207.2),
        "BI": Atom(83, 160, 4.75, "RHL", 207, 148, 208.9804),
        "PO": Atom(84, 190, 3.35, "SC", 197, np.nan, 209),
        "AT": Atom(85, np.nan, np.nan, "", 202, np.nan, 210),
        "RN": Atom(86, np.nan, np.nan, "FCC", 220, np.nan, 222),
        "FR": Atom(87, np.nan, np.nan, "BCC", 348, np.nan, 223),
        "RA": Atom(88, 215, np.nan, "", 283, np.nan, 226),
        "AC": Atom(89, 195, 5.31, "FCC", 247, np.nan, 227),
        "TH": Atom(90, 180, 5.08, "FCC", 245, np.nan, 232.0377),
        "PA": Atom(91, 180, 3.92, "TET", 243, np.nan, 231.03588),
        "U": Atom(92, 175, 2.85, "ORC", 241, np.nan, 238.02891),
        "NP": Atom(93, 175, 4.72, "ORC", 239, np.nan, 237),
        "PU": Atom(94, 175, np.nan, "MCL", 243, np.nan, 244),
        "AM": Atom(95, 175, np.nan, "", 244, np.nan, 243),
        "CM": Atom(96, np.nan, np.nan, "", 245, np.nan, 247),
        "BK": Atom(97, np.nan, np.nan, "", 244, np.nan, 247),
        "CF": Atom(98, np.nan, np.nan, "", 245, np.nan, 251),
        "ES": Atom(99, np.nan, np.nan, "", 245, np.nan, 252),
        "FM": Atom(100, np.nan, np.nan, "", 245, np.nan, 257),
        "MD": Atom(101, np.nan, np.nan, "", 246, np.nan, 258),
        "NO": Atom(102, np.nan, np.nan, "", 246, np.nan, 259),
        "LR": Atom(103, np.nan, np.nan, "", 246, np.nan, 262),
        "RF": Atom(104, np.nan, np.nan, "", np.nan, np.nan, 267),
        "DB": Atom(105, np.nan, np.nan, "", np.nan, np.nan, 268),
        "SG": Atom(106, np.nan, np.nan, "", np.nan, np.nan, 271),
        "BH": Atom(107, np.nan, np.nan, "", np.nan, np.nan, 274),
        "HS": Atom(108, np.nan, np.nan, "", np.nan, np.nan, 269),
        "MT": Atom(109, np.nan, np.nan, "", np.nan, np.nan, 276),
        "DS": Atom(110, np.nan, np.nan, "", np.nan, np.nan, 281),
        "RG": Atom(111, np.nan, np.nan, "", np.nan, np.nan, 281),
        "CN": Atom(112, np.nan, np.nan, "", np.nan, np.nan, 285),
        "NH": Atom(113, np.nan, np.nan, "", np.nan, np.nan, 286),
        "FL": Atom(114, np.nan, np.nan, "", np.nan, np.nan, 289),
        "MC": Atom(115, np.nan, np.nan, "", np.nan, np.nan, 288),
        "LV": Atom(116, np.nan, np.nan, "", np.nan, np.nan, 293),
        "TS": Atom(117, np.nan, np.nan, "", np.nan, np.nan, 294),
        "OG": Atom(118, np.nan, np.nan, "", np.nan, np.nan, 294),
    }

    def __getitem__(self, key: str):
        """
        Retrieve a value from the internal data using a given key.

        Parameters
        ----------
        key : str
            The key to use for retrieving the corresponding value from
            the internal data.

        Returns
        -------
        value
            The value associated with the provided key in the internal data.
        """
        return self._elements.get(key, self._default)
