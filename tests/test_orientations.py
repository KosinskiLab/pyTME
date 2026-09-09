import pytest
import numpy as np

from tme import Orientations
from tme.matching_utils import generate_tempfile_name


class TestDensity:
    def setup_method(self):
        self.translations = np.random.rand(100, 3).astype(np.float32)
        self.rotations = np.random.rand(100, 3).astype(np.float32)
        self.scores = np.random.rand(100).astype(np.float32)
        self.details = np.random.rand(100).astype(np.float32)
        self.metadata = {
            "_pytmeScore": self.scores,
            "_rlnClassNumber": self.details,
        }

        self.orientations = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            metadata=dict(self.metadata),
        )

    def teardown_method(self):
        self.translations = None
        self.rotations = None
        self.scores = None
        self.details = None
        self.metadata = None
        self.orientations = None

    def test_initialization(self):
        orientations = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            metadata=dict(self.metadata),
        )

        assert np.array_equal(self.translations, orientations.translations)
        assert np.array_equal(self.rotations, orientations.rotations)
        assert np.array_equal(self.scores, orientations.scores)
        assert np.array_equal(self.details, orientations.details)

    def test_initialization_type(self):
        orientations = Orientations(
            translations=self.translations.astype(int),
            rotations=self.rotations.astype(int),
            metadata=dict(self.metadata),
        )
        assert np.issubdtype(orientations.translations.dtype, np.floating)
        assert np.issubdtype(orientations.rotations.dtype, np.floating)

    def test_initialization_error(self):
        with pytest.raises(ValueError):
            _ = Orientations(
                translations=self.translations,
                rotations=np.random.rand(self.translations.shape[0] + 1),
                metadata=dict(self.metadata),
            )

        with pytest.raises(ValueError):
            _ = Orientations(
                translations=np.random.rand(self.translations.shape[0]),
                rotations=np.random.rand(self.translations.shape[0] + 1),
                metadata=dict(self.metadata),
            )
            _ = Orientations(
                translations=self.translations,
                rotations=np.random.rand(self.translations.shape[0]),
                metadata=dict(self.metadata),
            )

        assert True

    @pytest.mark.parametrize("file_format", ("tsv", "star", "tbl"))
    def test_to_file(self, file_format: str):
        output_file = generate_tempfile_name(suffix=f".{file_format}")
        self.orientations.to_file(output_file)
        assert True

    @pytest.mark.parametrize("file_format", ("tsv", "star", "tbl"))
    def test_from_file(self, file_format: str):
        output_file = generate_tempfile_name(suffix=f".{file_format}")
        self.orientations.to_file(output_file)
        orientations_new = Orientations.from_file(output_file)

        assert np.allclose(
            self.orientations.translations, orientations_new.translations
        )
        assert np.allclose(
            self.orientations.rotations, orientations_new.rotations, atol=1e-3
        )

    @pytest.mark.parametrize("input_format", ("tsv", "star", "tbl"))
    @pytest.mark.parametrize("output_format", ("tsv", "star", "tbl"))
    def test_file_format_io(self, input_format: str, output_format: str):
        output_file = generate_tempfile_name(suffix=f".{input_format}")
        output_file2 = generate_tempfile_name(suffix=f".{output_format}")

        self.orientations.to_file(output_file)
        orientations_new = Orientations.from_file(output_file)
        orientations_new.to_file(output_file2)

        assert True

    @pytest.mark.parametrize("drop_oob", (True, False))
    @pytest.mark.parametrize("shape", (10, 40, 80))
    @pytest.mark.parametrize("odd", (True, False))
    def test_extraction(self, shape: int, drop_oob: bool, odd: bool):
        if odd:
            shape = shape + (1 - shape % 2)

        data = np.random.rand(50, 50, 50)
        translations = np.array([[25, 25, 25], [15, 25, 35], [35, 25, 15], [0, 15, 49]])
        n = translations.shape[0]
        orientations = Orientations(
            translations=translations,
            rotations=np.random.rand(*translations.shape),
            metadata={
                "_pytmeScore": np.random.rand(n).astype(np.float32),
                "_rlnClassNumber": np.random.rand(n),
            },
        )
        extraction_shape = np.repeat(np.array(shape), data.ndim)
        orientations, cand_slices, obs_slices = orientations.get_extraction_slices(
            target_shape=data.shape,
            extraction_shape=extraction_shape,
            drop_out_of_box=drop_oob,
            return_orientations=True,
        )
        assert orientations.translations.shape[0] == len(cand_slices)
        assert len(cand_slices) == len(obs_slices)

        cand_slices2, obs_slices2 = orientations.get_extraction_slices(
            target_shape=data.shape,
            extraction_shape=extraction_shape,
            drop_out_of_box=drop_oob,
            return_orientations=False,
        )
        assert cand_slices == cand_slices2
        assert obs_slices == obs_slices2

        # Check whether extraction slices are pasted in center
        out = np.zeros(extraction_shape, dtype=data.dtype)
        center = np.divide(extraction_shape, 2).astype(int)
        for index, (cand_slice, obs_slice) in enumerate(zip(cand_slices, obs_slices)):
            out[cand_slice] = data[obs_slice]
            assert np.allclose(
                out[tuple(center)],
                data[tuple(orientations.translations[index].astype(int))],
            )

    def test_optics_default_empty(self):
        orientations = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            metadata=dict(self.metadata),
        )
        assert orientations.optics == {}

    def test_optics_round_trip_on_instance(self):
        orientations = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            metadata=dict(self.metadata),
            optics={"_rlnImagePixelSize": 2.62},
        )
        assert orientations.optics == {"_rlnImagePixelSize": 2.62}

    def test_optics_invalid_type(self):
        with pytest.raises(ValueError):
            _ = Orientations(
                translations=self.translations,
                rotations=self.rotations,
                metadata=dict(self.metadata),
                optics="not a dict",
            )

    def test_setstate_supplies_default_optics(self):
        orientations = object.__new__(Orientations)
        legacy_state = {
            "translations": self.translations,
            "rotations": self.rotations,
            "scores": self.scores,
            "details": self.details,
            "metadata": {},
        }
        orientations.__setstate__(legacy_state)
        assert orientations.optics == {}

    @pytest.mark.parametrize(
        "order", (("x", "y", "z"), ("z", "y", "x"), ("y", "x", "z"))
    )
    def test_txt_sort(self, order: str):
        output_file = generate_tempfile_name(suffix=".tsv")
        translations = ((50, 30, 20), (10, 5, 30))

        with open(output_file, mode="w", encoding="utf-8") as ofile:
            ofile.write("\t".join([str(x) for x in order]) + "\n")
            for translation in translations:
                ofile.write("\t".join([str(x) for x in translation]) + "\n")

        translations = np.array(translations).astype(np.float32)
        orientations = Orientations.from_file(output_file)

        out_order = zip(order, range(len(order)))
        out_order = tuple(
            x[1] for x in sorted(out_order, key=lambda x: x[0], reverse=False)
        )

        assert np.array_equal(translations[..., out_order], orientations.translations)

    def test_optics_preserved_by_getitem(self):
        optics = {"_rlnImagePixelSize": 2.62, "_rlnVoltage": 300.0}
        orientations = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            metadata=dict(self.metadata),
            optics=optics,
        )
        subset = orientations[np.arange(5)]
        assert subset.optics == optics

    def test_optics_preserved_by_copy(self):
        optics = {"_rlnImagePixelSize": 2.62}
        orientations = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            metadata=dict(self.metadata),
            optics=optics,
        )
        clone = orientations.copy()
        assert clone.optics == optics

    def test_to_star_emits_optics_block(self):
        orientations = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            metadata=dict(self.metadata),
            optics={"_rlnImagePixelSize": 2.62},
        )
        path = generate_tempfile_name(suffix=".star")
        orientations.to_file(path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        assert "data_optics" in text
        assert "_rlnImagePixelSize" in text
        assert "2.62" in text
        assert text.index("data_optics") < text.index("data_particles")

    def test_to_star_omits_optics_block_when_empty(self):
        orientations = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            metadata=dict(self.metadata),
        )
        path = generate_tempfile_name(suffix=".star")
        orientations.to_file(path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        assert "data_optics" not in text

    def test_to_star_writes_arbitrary_keys(self):
        orientations = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            metadata=dict(self.metadata),
            optics={
                "_rlnImagePixelSize": 2.62,
                "_rlnVoltage": 300.0,
                "_rlnSphericalAberration": 2.7,
            },
        )
        path = generate_tempfile_name(suffix=".star")
        orientations.to_file(path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for col in ("_rlnImagePixelSize", "_rlnVoltage", "_rlnSphericalAberration"):
            assert col in text

    def test_from_star_reads_optics(self):
        optics = {"_rlnImagePixelSize": 2.62, "_rlnVoltage": 300.0}
        orientations = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            metadata=dict(self.metadata),
            optics=optics,
        )
        path = generate_tempfile_name(suffix=".star")
        orientations.to_file(path)
        loaded = Orientations.from_file(path)
        assert loaded.optics["_rlnImagePixelSize"] == pytest.approx(2.62)
        assert loaded.optics["_rlnVoltage"] == pytest.approx(300.0)

    def test_from_star_without_optics_returns_empty_dict(self):
        orientations = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            metadata=dict(self.metadata),
        )
        path = generate_tempfile_name(suffix=".star")
        orientations.to_file(path)
        loaded = Orientations.from_file(path)
        assert loaded.optics == {}

    def test_scores_default_when_not_provided(self):
        o = Orientations(
            translations=self.translations,
            rotations=self.rotations,
        )
        assert o.scores.shape == (100,)
        assert o.scores.dtype == np.float32
        assert np.all(o.scores == 0)

    def test_details_default_when_not_provided(self):
        o = Orientations(
            translations=self.translations,
            rotations=self.rotations,
        )
        assert o.details.shape == (100,)
        assert np.all(o.details == -1)

    def test_scores_read_from_metadata(self):
        payload = np.arange(100, dtype=np.float32)
        o = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            metadata={"_pytmeScore": payload},
        )
        assert np.array_equal(o.scores, payload)

    def test_details_read_from_metadata(self):
        payload = np.arange(100)
        o = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            metadata={"_rlnClassNumber": payload},
        )
        assert np.array_equal(o.details, payload)

    def test_explicit_kwarg_overrides_metadata(self):
        zeros = np.zeros(100, dtype=np.float32)
        with pytest.warns(DeprecationWarning):
            o = Orientations(
                translations=self.translations,
                rotations=self.rotations,
                scores=zeros,
                metadata={"_pytmeScore": np.arange(100, dtype=np.float32)},
            )
        assert np.array_equal(o.scores, zeros)
        assert np.array_equal(o.metadata["_pytmeScore"], zeros)

    def test_setstate_migrates_legacy_scores_and_details(self):
        o = Orientations(
            translations=self.translations,
            rotations=self.rotations,
        )
        legacy_state = {
            "translations": self.translations,
            "rotations": self.rotations,
            "scores": self.scores,
            "details": self.details,
        }
        o.__setstate__(legacy_state)
        assert np.array_equal(o.metadata["_pytmeScore"], self.scores)
        assert np.array_equal(o.metadata["_rlnClassNumber"], self.details)
        assert "scores" not in o.__dict__
        assert "details" not in o.__dict__
        assert np.array_equal(o.scores, self.scores)
        assert np.array_equal(o.details, self.details)

    def test_to_star_no_duplicate_columns_when_metadata_has_reserved_keys(self):
        o = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            metadata={
                "_pytmeScore": self.scores.copy(),
                "_rlnClassNumber": self.details.astype(int),
                "_rlnDefocusU": np.full(100, 20000.0),
            },
        )
        path = generate_tempfile_name(suffix=".star")
        o.to_file(path)
        with open(path) as f:
            header_lines = [
                line.strip()
                for line in f
                if line.startswith("_rln") or line.startswith("_pytme")
            ]
        assert header_lines.count("_pytmeScore") == 1
        assert header_lines.count("_rlnClassNumber") == 1
        assert "_rlnDefocusU" in header_lines
