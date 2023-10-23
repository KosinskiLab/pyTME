from tempfile import mkstemp
from os import remove

import pytest
import numpy as np

from tme import Structure
from tme.matching_utils import euler_to_rotationmatrix, minimum_enclosing_box


STRUCTURE_ATTRIBUTES = [
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
    "details",
]


class TestStructure:
    def setup_method(self):
        self.structure = Structure.from_file("./tme/tests/data/Structures/5khe.cif")
        _, self.path = mkstemp()

    def teardown_method(self):
        del self.structure
        remove(self.path)

    def compare_structures(self, structure1, structure2, exclude_attributes=[]):
        for attribute in STRUCTURE_ATTRIBUTES:
            if attribute in exclude_attributes:
                continue
            value = getattr(structure1, attribute)
            value_comparison = getattr(structure2, attribute)
            if type(value) == np.ndarray:
                assert np.all(value_comparison == value)
            else:
                assert value == value_comparison

    def test_initialization(self):
        structure = Structure(
            record_type=self.structure.record_type,
            atom_serial_number=self.structure.atom_serial_number,
            atom_name=self.structure.atom_name,
            atom_coordinate=self.structure.atom_coordinate,
            alternate_location_indicator=self.structure.alternate_location_indicator,
            residue_name=self.structure.residue_name,
            chain_identifier=self.structure.chain_identifier,
            residue_sequence_number=self.structure.residue_sequence_number,
            code_for_residue_insertion=self.structure.code_for_residue_insertion,
            occupancy=self.structure.occupancy,
            temperature_factor=self.structure.temperature_factor,
            segment_identifier=self.structure.segment_identifier,
            element_symbol=self.structure.element_symbol,
            charge=self.structure.charge,
            details=self.structure.details,
        )

        for attribute in STRUCTURE_ATTRIBUTES:
            value = getattr(self.structure, attribute)
            value_comparison = getattr(structure, attribute)
            if type(value) == np.ndarray:
                assert np.all(value_comparison == value)
            else:
                assert value == value_comparison

    @pytest.mark.parametrize(
        "modified_attribute",
        [
            ("record_type"),
            ("atom_serial_number"),
            ("atom_name"),
            ("atom_coordinate"),
            ("alternate_location_indicator"),
            ("residue_name"),
            ("chain_identifier"),
            ("residue_sequence_number"),
            ("code_for_residue_insertion"),
            ("occupancy"),
            ("temperature_factor"),
            ("segment_identifier"),
            ("element_symbol"),
        ],
    )
    def test_initialization_errors(self, modified_attribute):
        kwargs = {
            attribute: getattr(self.structure, attribute)
            for attribute in STRUCTURE_ATTRIBUTES
            if attribute != modified_attribute
        }
        kwargs[modified_attribute] = getattr(self.structure, modified_attribute)[:1]

        with pytest.raises(ValueError):
            Structure(**kwargs)

    def test__getitem__(self):
        ret_single_index = self.structure[1]
        ret = self.structure[[1]]
        self.compare_structures(ret_single_index, ret)

        ret = self.structure[self.structure.record_type == "ATOM"]
        assert np.all(ret.record_type == "ATOM")

        ret = self.structure[self.structure.element_symbol == "C"]
        assert np.all(ret.element_symbol == "C")

    def test__repr__(self):
        unique_chains = ",".join(np.unique(self.structure.chain_identifier))

        min_atom = np.min(self.structure.atom_serial_number)
        max_atom = np.max(self.structure.atom_serial_number)
        n_atom = max_atom - min_atom + 1

        min_residue = np.min(self.structure.residue_sequence_number)
        max_residue = np.max(self.structure.residue_sequence_number)
        n_residue = max_residue - min_residue + 1

        repr_str = (
            f"Structure object at {id(self.structure)}\n"
            f"Unique Chains: {unique_chains}, "
            f"Atom Range: {min_atom}-{max_atom} [N = {n_atom}], "
            f"Residue Range: {min_residue}-{max_residue} [N = {n_residue}]"
        )
        assert repr_str == self.structure.__repr__()

    @pytest.mark.parametrize(
        "path",
        [
            ("./tme/tests/data/Structures/5khe.cif"),
            ("./tme/tests/data/Structures/5khe.pdb"),
        ],
    )
    def test_fromfile(self, path):
        _ = Structure.from_file(path)

    def test_fromfile_error(self):
        with pytest.raises(NotImplementedError):
            _ = Structure.from_file("madeup.extension")

    @pytest.mark.parametrize("file_format", [("cif"), ("pdb")])
    def test_to_file(self, file_format):
        _, path = mkstemp()
        path = f"{path}.{file_format}"
        self.structure.to_file(path)
        read = self.structure.from_file(path)
        comparison = self.structure.copy()

        self.compare_structures(comparison, read, exclude_attributes=["details"])

    def test_to_file_error(self):
        _, path = mkstemp()
        path = f"{path}.RAISERROR"
        with pytest.raises(NotImplementedError):
            self.structure.to_file(path)

    def test_subset_by_chain(self):
        chain = "A"
        ret = self.structure.subset_by_chain(chain=chain)
        assert np.all(ret.chain_identifier == chain)

    def test_subset_by_chain_range(self):
        chain, start, stop = "A", 0, 20
        ret = self.structure.subset_by_range(chain=chain, start=start, stop=stop)
        assert np.all(ret.chain_identifier == chain)
        assert np.all(
            np.logical_and(
                ret.residue_sequence_number >= start,
                ret.residue_sequence_number <= stop,
            )
        )

    def test_center_of_mass(self):
        center_of_mass = self.structure.center_of_mass()
        assert center_of_mass.shape[0] == self.structure.atom_coordinate.shape[1]
        assert np.allclose(center_of_mass, [-0.89391639, 29.94908928, -2.64736741])

    def test_centered(self):
        ret, translation = self.structure.centered()
        box = minimum_enclosing_box(coordinates=self.structure.atom_coordinate.T)
        assert np.allclose(ret.center_of_mass(), np.divide(box, 2), atol=1)

    def test__get_atom_weights_error(self):
        with pytest.raises(NotImplementedError):
            self.structure._get_atom_weights(
                self.structure.atom_name, weight_type="RAISEERROR"
            )

    def test_compare_structures(self):
        rmsd = Structure.compare_structures(self.structure, self.structure)
        assert rmsd == 0

        rmsd = Structure.compare_structures(
            self.structure, self.structure, weighted=True
        )
        assert rmsd == 0

        translation = (3, 0, 0)
        structure_transform = self.structure.rigid_transform(
            translation=translation,
            rotation_matrix=np.eye(self.structure.atom_coordinate.shape[1]),
        )
        rmsd = Structure.compare_structures(self.structure, structure_transform)
        assert np.allclose(rmsd, np.linalg.norm(translation))

    def test_comopare_structures_error(self):
        ret = self.structure[[1, 2, 3, 4, 5]]
        with pytest.raises(ValueError):
            Structure.compare_structures(self.structure, ret)

    def test_align_structures(self):
        rotation_matrix = euler_to_rotationmatrix((20, -10, 45))
        translation = (10, 0, -15)

        structure_transform = self.structure.rigid_transform(
            rotation_matrix=rotation_matrix, translation=translation
        )
        aligned, final_rmsd = Structure.align_structures(
            self.structure, structure_transform
        )
        assert final_rmsd <= 0.1

        aligned, final_rmsd = Structure.align_structures(
            self.structure, structure_transform, sampling_rate=1
        )
        assert final_rmsd <= 1
