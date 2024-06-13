from tempfile import mkstemp

import pytest
import numpy as np

from tme import Orientations


class TestDensity:
    def setup_method(self):
        self.translations = np.random.rand(100, 3).astype(np.float32)
        self.rotations = np.random.rand(100, 3).astype(np.float32)
        self.scores = np.random.rand(100).astype(np.float32)
        self.details = np.random.rand(100).astype(np.float32)

        self.orientations = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            scores=self.scores,
            details=self.details,
        )

    def teardown_method(self):
        self.translations = None
        self.rotations = None
        self.scores = None
        self.details = None
        self.orientations = None

    def test_initialization(self):
        orientations = Orientations(
            translations=self.translations,
            rotations=self.rotations,
            scores=self.scores,
            details=self.details,
        )

        assert np.array_equal(self.translations, orientations.translations)
        assert np.array_equal(self.rotations, orientations.rotations)
        assert np.array_equal(self.scores, orientations.scores)
        assert np.array_equal(self.details, orientations.details)

    def test_initialization_type(self):
        orientations = Orientations(
            translations=self.translations.astype(int),
            rotations=self.rotations.astype(int),
            scores=self.scores.astype(int),
            details=self.details.astype(int),
        )
        assert np.issubdtype(orientations.translations.dtype, np.floating)
        assert np.issubdtype(orientations.rotations.dtype, np.floating)
        assert np.issubdtype(orientations.scores.dtype, np.floating)
        assert np.issubdtype(orientations.details.dtype, np.floating)

    def test_initialization_error(self):
        with pytest.raises(ValueError):
            _ = Orientations(
                translations=self.translations,
                rotations=np.random.rand(self.translations.shape[0] + 1),
                scores=self.scores,
                details=self.details,
            )

        with pytest.raises(ValueError):
            _ = Orientations(
                translations=np.random.rand(self.translations.shape[0]),
                rotations=np.random.rand(self.translations.shape[0] + 1),
                scores=self.scores,
                details=self.details,
            )
            _ = Orientations(
                translations=self.translations,
                rotations=np.random.rand(self.translations.shape[0]),
                scores=self.scores,
                details=self.details,
            )

        assert True

    @pytest.mark.parametrize("file_format", ("text", "relion", "tbl"))
    def test_to_file(self, file_format: str):
        _, output_file = mkstemp(suffix=f".{file_format}")
        self.orientations.to_file(output_file)
        assert True

    @pytest.mark.parametrize("file_format", ("text", "star", "tbl"))
    def test_from_file(self, file_format: str):
        _, output_file = mkstemp(suffix=f".{file_format}")
        self.orientations.to_file(output_file)
        orientations_new = Orientations.from_file(output_file)

        assert np.array_equal(
            self.orientations.translations, orientations_new.translations
        )

    @pytest.mark.parametrize("input_format", ("text", "star", "tbl"))
    @pytest.mark.parametrize("output_format", ("text", "star", "tbl"))
    def test_file_format_io(self, input_format: str, output_format: str):
        _, output_file = mkstemp(suffix=f".{input_format}")
        _, output_file2 = mkstemp(suffix=f".{output_format}")

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
        orientations = Orientations(
            translations=translations,
            rotations=np.random.rand(*translations.shape),
            scores=np.random.rand(translations.shape[0]),
            details=np.random.rand(translations.shape[0]),
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
        center = np.divide(extraction_shape, 2) + np.mod(extraction_shape, 2)
        center = center.astype(int)
        for index, (cand_slice, obs_slice) in enumerate(zip(cand_slices, obs_slices)):
            out[cand_slice] = data[obs_slice]
            assert np.allclose(
                out[tuple(center)],
                data[tuple(orientations.translations[index].astype(int))],
            )
