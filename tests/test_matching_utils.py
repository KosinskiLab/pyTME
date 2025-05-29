from tempfile import mkstemp
from importlib_resources import files

import pytest
import numpy as np
from scipy.signal import correlate

from tme import Density
from tme.backends import backend as be
from tme.memory import MATCHING_MEMORY_REGISTRY
from tme.matching_utils import (
    compute_parallelization_schedule,
    elliptical_mask,
    box_mask,
    tube_mask,
    create_mask,
    scramble_phases,
    apply_convolution_mode,
    write_pickle,
    load_pickle,
    _normalize_template_overflow_safe,
)

BASEPATH = files("tests.data")


class TestMatchingUtils:
    def setup_method(self):
        self.density = Density.from_file(str(BASEPATH.joinpath("Raw/em_map.map")))
        self.structure_density = Density.from_structure(
            filename_or_structure=str(BASEPATH.joinpath("Structures/5khe.cif")),
            origin=self.density.origin,
            shape=self.density.shape,
            sampling_rate=self.density.sampling_rate,
        )

    @pytest.mark.parametrize("matching_method", list(MATCHING_MEMORY_REGISTRY.keys()))
    @pytest.mark.parametrize("max_cores", range(1, 10, 3))
    @pytest.mark.parametrize("max_ram", [1e5, 1e7, 1e9])
    def test_compute_parallelization_schedule(
        self, matching_method, max_cores, max_ram
    ):
        max_cores, max_ram = int(max_cores), int(max_ram)
        compute_parallelization_schedule(
            self.density.shape,
            self.structure_density.shape,
            matching_method=matching_method,
            max_cores=max_cores,
            max_ram=max_ram,
            max_splits=256,
        )

    def test_create_mask(self):
        create_mask(
            mask_type="ellipse",
            shape=self.density.shape,
            radius=5,
            center=np.divide(self.density.shape, 2),
        )

    def test_create_mask_error(self):
        with pytest.raises(ValueError):
            create_mask(mask_type=None)

    def test_elliptical_mask(self):
        elliptical_mask(
            shape=self.density.shape,
            radius=5,
            center=np.divide(self.density.shape, 2),
        )

    def test_box_mask(self):
        box_mask(
            shape=self.density.shape,
            height=[5, 10, 20],
            center=np.divide(self.density.shape, 2),
        )

    def test_tube_mask(self):
        tube_mask(
            shape=self.density.shape,
            outer_radius=10,
            inner_radius=5,
            height=5,
            base_center=np.divide(self.density.shape, 2),
            symmetry_axis=1,
        )

    def test_tube_mask_error(self):
        with pytest.raises(ValueError):
            tube_mask(
                shape=self.density.shape,
                outer_radius=5,
                inner_radius=10,
                height=5,
                base_center=np.divide(self.density.shape, 2),
                symmetry_axis=1,
            )

        with pytest.raises(ValueError):
            tube_mask(
                shape=self.density.shape,
                outer_radius=5,
                inner_radius=10,
                height=10 * np.max(self.density.shape),
                base_center=np.divide(self.density.shape, 2),
                symmetry_axis=1,
            )

        with pytest.raises(ValueError):
            tube_mask(
                shape=self.density.shape,
                outer_radius=5,
                inner_radius=10,
                height=10 * np.max(self.density.shape),
                base_center=np.divide(self.density.shape, 2),
                symmetry_axis=len(self.density.shape) + 1,
            )

    def test_scramble_phases(self):
        scramble_phases(arr=self.density.data, noise_proportion=0.5)

    @pytest.mark.parametrize("convolution_mode", ["full", "valid", "same"])
    def test_apply_convolution_mode(self, convolution_mode):
        correlation = correlate(
            self.density.data, self.structure_density.data, method="direct", mode="full"
        )
        ret = apply_convolution_mode(
            arr=correlation,
            convolution_mode=convolution_mode,
            s1=self.density.shape,
            s2=self.structure_density.shape,
        )
        if convolution_mode == "full":
            expected_size = correlation.shape
        elif convolution_mode == "same":
            expected_size = self.density.shape
        else:
            expected_size = np.subtract(
                self.density.shape, self.structure_density.shape
            )
            expected_size += np.mod(self.structure_density.shape, 2)
        assert np.allclose(ret.shape, expected_size)

    def test_apply_convolution_mode_error(self):
        correlation = correlate(
            self.density.data, self.structure_density.data, method="direct", mode="full"
        )
        with pytest.raises(ValueError):
            _ = apply_convolution_mode(
                arr=correlation,
                convolution_mode=None,
                s1=self.density.shape,
                s2=self.structure_density.shape,
            )

    def test_pickle_io(self):
        _, filename = mkstemp()

        data = ["Hello", 123, np.array([1, 2, 3])]
        write_pickle(data=data, filename=filename)
        loaded_data = load_pickle(filename)
        assert all([np.array_equal(a, b) for a, b in zip(data, loaded_data)])

        data = 42
        write_pickle(data=data, filename=filename)
        loaded_data = load_pickle(filename)
        assert loaded_data == data

        _, filename = mkstemp()
        data = np.memmap(filename, dtype="float32", mode="w+", shape=(3,))
        data[:] = [1.1, 2.2, 3.3]
        data.flush()
        data = np.memmap(filename, dtype="float32", mode="r+", shape=(3,))
        _, filename = mkstemp()
        write_pickle(data=data, filename=filename)
        loaded_data = load_pickle(filename)
        assert np.array_equal(loaded_data, data)

    def test_normalize_template_overflow_safe(self):
        template = be.random.random((10, 10)).astype(be.float32)
        mask = be.ones_like(template)
        n_observations = 100.0

        result = _normalize_template_overflow_safe(template, mask, n_observations)
        assert result.shape == template.shape
        assert result.dtype == template.dtype
        assert np.allclose(result.mean(), 0, atol=0.1)
        assert np.allclose(result.std(), 1, atol=0.1)
