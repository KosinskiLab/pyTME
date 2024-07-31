""" Implements parsers for atomic structure file formats.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import re
from collections import deque
from typing import List, Dict
from abc import ABC, abstractmethod

import numpy as np


class Parser(ABC):
    """
    Base class for structure file parsers.

    Classes inheriting from :py:class:`Parser` need to define
    a ``parse_input`` method that accepts a list of lines and returns a
    dictionary representation of the data.
    """

    def __init__(self, filename: str, mode: str = "r") -> None:
        """
        Initialize a Parser object.

        Parameters
        ----------
        filename : str
            File name to parse data from.

        mode : str, optional
            Mode to open the file. Default is 'r' for read.
        """
        with open(filename, "r") as infile:
            data = infile.read()

        data = deque(filter(lambda line: line and line[0] != "#", data.split("\n")))
        self._data = self.parse_input(data)

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
        return self._data[key]

    def __contains__(self, key) -> bool:
        """
        Check if a given key exists in the internal data.

        Parameters
        ----------
        key : str
            The key to check for in the internal data.

        Returns
        -------
        bool
            True if the key exists in the internal data, False otherwise.
        """
        return key in self._data

    def get(self, key, default):
        """
        Retrieve a value from the internal data using a given key. If the
        key does not exist, return a default value.

        Parameters
        ----------
        key : str
            The key to use for retrieving the corresponding value from
             the internal data.

        default : Any
            The value to return if the key does not exist in the internal data.

        Returns
        -------
        value
            The value associated with the provided key in the internal data,
            or the default value if the key does not exist.
        """
        if key in self._data:
            return self[key]
        return default

    def keys(self):
        """
        List keys available in internal dictionary.
        """
        return self._data.keys()

    def values(self):
        """
        List values available in internal dictionary.
        """
        return self._data.values()

    def items(self):
        """
        List items available in internal dictionary.
        """
        return self._data.items()

    @abstractmethod
    def parse_input(self, lines: List[str]) -> Dict:
        """
        Parse a list of lines from a file and convert the data into a dictionary.

        This function is not intended to be called directly, but should rather be
        defined by classes inheriting from :py:class:`Parser` to parse a given
        file format.

        Parameters
        ----------
        lines : list of str
            The lines of a structure file to parse.

        Returns
        -------
        dict
            A dictionary containing the parsed data.
        """


class PDBParser(Parser):
    """
    Convert PDB file data into a dictionary representation [1]_.

    References
    ----------
    .. [1]  https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
    """

    def parse_input(self, lines: List[str]) -> Dict:
        """
        Parse a list of lines from a PDB file and convert the data into a dictionary.

        Parameters
        ----------
        lines : list of str
            The lines of a PDB file to parse.

        Returns
        -------
        dict
            A dictionary containing the parsed data from the PDB file.
        """
        metadata = {
            "resolution": re.compile(
                r"(.)+?(EFFECTIVE RESOLUTION\s+\(ANGSTROMS\)){1}(.)+?(\d+\.\d+)(\s)*$"
            ),
            "reconstruction_method": re.compile(
                r"(.)+?(RECONSTRUCTION METHOD)+(.)+?(\w+\s*\w+)(\s)*$"
            ),
            "electron_source": re.compile(r"(.)+?(SOURCE)+(.)+?(\w+\s*\w+)(\s)*$"),
            "illumination_mode": re.compile(
                r"(.)+?(ILLUMINATION MODE)+(.)+?(\w+\s*\w+)(\s)*$"
            ),
            "microscope_mode": re.compile(
                r"(.)+?(IMAGING MODE)+(.)+?(\w+\s*\w+)(\s)*$"
            ),
            "microscope_model": re.compile(
                r"(.)+?(MICROSCOPE MODEL)+(.+?:\s+)+?(.+)(\s)*$"
            ),
        }

        data = {
            "record_type": [],
            "atom_serial_number": [],
            "atom_name": [],
            "alternate_location_indicator": [],
            "residue_name": [],
            "chain_identifier": [],
            "residue_sequence_number": [],
            "code_for_residue_insertion": [],
            "atom_coordinate": [],
            "occupancy": [],
            "temperature_factor": [],
            "segment_identifier": [],
            "element_symbol": [],
            "charge": [],
            "details": {},
        }
        data["details"]["resolution"] = np.nan

        for line in lines:
            if line.startswith("REMARK"):
                matches = [(key, metadata[key].match(line)) for key in metadata]
                matches = [match for match in matches if match[1]]
                for key, match in matches:
                    data["details"][key] = match.group(4)
                    _ = metadata.pop(key)
            elif line.startswith("ATOM") or line.startswith("HETATM"):
                data["record_type"].append(line[0:6])
                data["atom_serial_number"].append(line[6:11])
                data["atom_name"].append(line[12:16])
                data["alternate_location_indicator"].append(line[16])
                data["residue_name"].append(line[17:20])

                data["chain_identifier"].append(line[21])
                data["residue_sequence_number"].append(line[22:26])
                data["code_for_residue_insertion"].append(line[26])
                data["atom_coordinate"].append((line[30:38], line[38:46], line[46:54]))
                data["occupancy"].append(line[54:60])
                data["temperature_factor"].append(line[60:66])
                data["segment_identifier"].append(line[74:76])
                data["element_symbol"].append(line[76:78])
                data["charge"].append(line[78:80])

        data["details"]["resolution"] = float(data["details"]["resolution"])

        return data


class MMCIFParser(Parser):
    """
    Convert MMCIF file data into a dictionary representation. This implementation
    heavily relies on the atomium library [1]_.

    References
    ----------
    .. [1]  Ireland, S. M., & Martin, A. C. R. (2020). atomium (Version 1.0.0)
            [Computer software]. https://doi.org/10.1093/bioinformatics/btaa072
    """

    def parse_input(self, lines: List[str]) -> Dict:
        """
        Parse a list of lines from an MMCIF file and convert the data into a dictionary.

        Parameters
        ----------
        lines : list of str
            The lines of an MMCIF file to parse.

        Returns
        -------
        dict
            A dictionary containing the parsed data from the MMCIF file.
        """
        lines = self._consolidate_strings(lines)
        blocks = self._split_in_blocks(lines)
        mmcif_dict = {}
        for block in blocks:
            if block["lines"][0] == "loop_":
                mmcif_dict[block["category"]] = self._loop_block_to_dict(block)
            else:
                mmcif_dict[block["category"]] = self._non_loop_block_to_dict(block)
        return mmcif_dict

    @staticmethod
    def _consolidate_strings(lines: List[str]) -> List[str]:
        """
        Consolidate multi-line strings that have been separated by semicolons in a
        list of strings.

        Parameters
        ----------
        lines : deque of str
            Deque of strings where each string is a line from an MMCIF file.

        Returns
        -------
        deque of str
            A deque of consolidated strings from the given input.
        """
        new_lines = deque()
        while lines:
            line = lines.popleft()
            if line.startswith(";"):
                string = [line[1:].strip()]
                while not lines[0].startswith(";"):
                    string.append(lines.popleft())
                lines.popleft()
                new_lines[-1] += ' "{}"'.format(
                    " ".join(string).replace('"', "").replace("'", "'")
                )
            else:
                new_lines.append(line.replace('"', "").replace("'", "'"))
        return new_lines

    @staticmethod
    def _split_in_blocks(lines: List[str]) -> List[Dict]:
        """
        Split a deque of consolidated strings into a list of dictionaries,
        each representing a block of data.

        Parameters
        ----------
        lines : deque of str
            Deque of consolidated strings where each string is a line from
            an MMCIF file.

        Returns
        -------
        list of dict
            A list of dictionaries where each dictionary represents a block
            of data from the MMCIF file.
        """
        category = None
        block, blocks = [], []
        while lines:
            line = lines.popleft()
            if line.startswith("data_"):
                continue
            if line.startswith("_"):
                line_category = line.split(".")[0]
                if line_category != category:
                    if category:
                        blocks.append({"category": category[1:], "lines": block})
                    category = line_category
                    block = []
            if line.startswith("loop_"):
                if category:
                    blocks.append({"category": category[1:], "lines": block})
                category = lines[0].split(".")[0]
                block = []
            block.append(line)
        if block:
            blocks.append({"category": category[1:], "lines": block})
        return blocks

    @staticmethod
    def _non_loop_block_to_dict(block: Dict) -> Dict:
        """
        Convert a non-loop block of data into a dictionary.

        Parameters
        ----------
        block : dict
            A dictionary representing a non-loop block of data from an MMCIF file.

        Returns
        -------
        dict
            A dictionary representing the parsed data from the given non-loop block.
        """
        d = {}
        # category = block["lines"][0].split(".")[0]
        for index in range(len(block["lines"]) - 1):
            if block["lines"][index + 1][0] != "_":
                block["lines"][index] += " " + block["lines"][index + 1]
        block["lines"] = [line for line in block["lines"] if line[0] == "_"]
        for line in block["lines"]:
            name = line.split(".")[1].split()[0]
            value = " ".join(line.split()[1:])
            d[name] = value
        return d

    def _loop_block_to_dict(self, block: Dict) -> Dict:
        """
        Convert a loop block of data into a dictionary.

        Parameters
        ----------
        block : dict
            A dictionary representing a loop block of data from an MMCIF file.

        Returns
        -------
        dict
            A dictionary representing the parsed data from the given loop block.
        """
        names, lines = [], []
        body_start = 0
        for index, line in enumerate(block["lines"][1:], start=1):
            if not line.startswith("_" + block["category"]):
                body_start = index
                break
        names = [line.split(".")[1].rstrip() for line in block["lines"][1:body_start]]
        lines = [self._split_line(line) for line in block["lines"][body_start:]]
        # reunites broken lines
        for n in range(len(lines) - 1):
            while n < len(lines) - 1 and len(lines[n]) + len(lines[n + 1]) <= len(
                names
            ):
                lines[n] += lines.pop(n + 1)
        res = {name: [] for name in names}
        for line in lines:
            for name, value in zip(names, line):
                res[name].append(value)
        return res

    @staticmethod
    def _split_line(line: str) -> List[str]:
        """
        Split a string into substrings, ignoring quotation marks within the string.

        Parameters
        ----------
        line : str
            The string to be split.

        Returns
        -------
        list of str
            A list of substrings resulting from the split operation on the given string.
        """
        if not re.search("['\"]", line):
            return line.split()

        chars = deque(line.strip())
        values, value, in_string = [], [], False
        while chars:
            char = chars.popleft()
            if char == " " and not in_string:
                values.append("".join(value))
                value = []
            elif char == '"':
                in_string = not in_string
                value.append(char)
            else:
                value.append(char)

        values.append(value)
        return ["".join(v) for v in values if v]
