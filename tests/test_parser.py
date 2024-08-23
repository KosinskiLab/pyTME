import pytest
from importlib_resources import files

from tme.parser import Parser, PDBParser


class TestParser:
    def setup_method(self):
        self.pdb_file = str(files("tests.data").joinpath("Structures/5khe.pdb"))

    def teardown_method(self):
        self.pdb_file = None

    def test_initialize_parser_error(self):
        with pytest.raises(TypeError):
            _ = Parser(self.pdb_file)

    def test_parser_get(self):
        parser = PDBParser(self.pdb_file)
        assert parser.get("NOTPRESENT", None) is None
        assert parser.get("record_type", None) is not None

    def test_parser_keys(self):
        parser = PDBParser(self.pdb_file)
        assert parser.keys() == parser._data.keys()

    def test_parser_values(self):
        parser = PDBParser(self.pdb_file)
        assert str(parser.values()) == str(parser._data.values())

    def test_parser_items(self):
        parser = PDBParser(self.pdb_file)
        assert parser.items() == parser._data.items()
