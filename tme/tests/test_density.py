from tempfile import mkstemp
from itertools import permutations
from os import remove

import pytest
import numpy as np

from tme import Density, Structure, Preprocessor
from tme.matching_utils import create_mask, euler_to_rotationmatrix

DEFAULT_DATA = create_mask(
    mask_type="ellipse",
    center=(20, 20, 20),
    radius=(10, 5, 10),
    shape=(50, 50, 50),
)
DEFAULT_DATA = Preprocessor().gaussian_filter(DEFAULT_DATA * 10, sigma=2)
DEFAULT_DATA = DEFAULT_DATA.astype(np.float32)
DEFAULT_ORIGIN = np.array([0, 0, 0])
DEFAULT_SAMPLING_RATE = np.array([1, 1, 1])


class TestDensity:
    def setup_method(self):
        self.density = Density(
            data=DEFAULT_DATA,
            origin=DEFAULT_ORIGIN,
            sampling_rate=DEFAULT_SAMPLING_RATE,
            metadata={
                "min": DEFAULT_DATA.min(),
                "max": DEFAULT_DATA.max(),
                "mean": DEFAULT_DATA.mean(),
                "std": DEFAULT_DATA.std(),
            },
        )
        _, self.path = mkstemp()
        self.structure_path = "tme/tests/data/Structures/5khe.cif"

    def teardown_method(self):
        del self.density
        remove(self.path)

    def test_initialization(self):
        data = DEFAULT_DATA
        origin = DEFAULT_ORIGIN
        sampling_rate = DEFAULT_SAMPLING_RATE
        metadata = {"test_key": "test_value"}

        density = Density(data, origin, sampling_rate, metadata)

        assert np.array_equal(density.data, data)
        assert np.array_equal(density.origin, origin)
        assert np.array_equal(density.sampling_rate, sampling_rate)
        assert density.metadata == metadata

    @pytest.mark.parametrize(
        "data,origin,sampling_rate,metadata",
        [
            (np.random.rand(50, 50, 50), (0, 0, 0), (1, 2), {}),
            (np.random.rand(50, 50, 50), (0, 0, 0), (1, 2, 3, 4), {}),
            (np.random.rand(50, 50, 50), (0, 0, 0), (1, 2, 3), "not_a_dict"),
            (np.random.rand(50, 50, 50), (0, 0), (1, 2, 3), "not_a_dict"),
        ],
    )
    def test_initialization_errors(self, data, origin, sampling_rate, metadata):
        with pytest.raises(ValueError):
            Density(data, origin, sampling_rate, metadata)

    def test_repr(self):
        data = DEFAULT_DATA
        origin = DEFAULT_ORIGIN
        sampling_rate = DEFAULT_SAMPLING_RATE

        density = Density(data, origin, sampling_rate)
        repr_str = density.__repr__()

        response = "Density object at {}\nOrigin: {}, sampling_rate: {}, Shape: {}"
        response = response.format(
            hex(id(density)),
            tuple(np.round(density.origin, 3)),
            tuple(np.round(density.sampling_rate, 3)),
            density.shape,
        )
        assert response == repr_str

    @pytest.mark.parametrize("gzip", [(False), (True)])
    def test_to_file(self, gzip: bool):
        self.density.to_file(self.path, gzip=gzip)
        assert True

    def test_from_file(self):
        self.test_to_file(gzip=False)
        density = Density.from_file(self.path)
        assert np.allclose(density.data, self.density.data)
        assert np.allclose(density.sampling_rate, self.density.sampling_rate)
        assert np.allclose(density.origin, self.density.origin)
        assert density.metadata == self.density.metadata

    @pytest.mark.parametrize("extension", ("mrc", "em", "tiff", "h5"))
    @pytest.mark.parametrize("gzip", (True, False))
    @pytest.mark.parametrize("use_memmap", (True, False))
    @pytest.mark.parametrize("subset", (True, False))
    def test_file_format_io(self, extension, gzip, subset, use_memmap):
        base = Density(
            data=np.random.rand(50, 50, 50), origin=(0, 0, 0), sampling_rate=(1, 1, 1)
        )
        data_subset = (slice(0, 22), slice(31, 46), slice(12, 25))
        if extension not in ("mrc", "em"):
            base = Density(
                data=np.random.rand(50, 50), origin=(0, 0), sampling_rate=(1, 1)
            )
            data_subset = (slice(0, 22), slice(31, 46))
        if gzip:
            use_memmap = False

        if not subset:
            data_subset = tuple(slice(0, x) for x in base.shape)

        suffix = f".{extension}.gz" if gzip else f".{extension}"
        _, output_file = mkstemp(suffix=suffix)
        base.to_file(output_file, gzip=gzip)
        temp = Density.from_file(output_file, use_memmap=use_memmap, subset=data_subset)
        assert np.allclose(base.data[data_subset], temp.data)
        if extension.upper() == "MRC":
            assert np.allclose(base.origin, temp.origin)
            assert np.allclose(base.sampling_rate, temp.sampling_rate)

    def test__read_binary_subset_error(self):
        base = Density(
            data=np.random.rand(50, 50, 50), origin=(0, 0, 0), sampling_rate=(1, 1, 1)
        )
        _, output_file = mkstemp()
        base.to_file(output_file)
        with pytest.raises(ValueError):
            Density.from_file(output_file, subset=(slice(0, 10),))
            Density.from_file(
                output_file, subset=(slice(-1, 10), slice(5, 10), slice(5, 10))
            )
            Density.from_file(
                output_file, subset=(slice(20, 100), slice(5, 10), slice(5, 10))
            )

    def test_from_structure(self):
        _ = Density.from_structure(
            self.structure_path,
            origin=self.density.origin,
            shape=self.density.shape,
            sampling_rate=self.density.sampling_rate,
        )
        _ = Density.from_structure(self.structure_path, shape=(30, 30, 30))
        _ = Density.from_structure(
            self.structure_path,
            shape=(30, 30, 30),
            origin=self.density.origin,
        )
        _ = Density.from_structure(self.structure_path, origin=self.density.origin)
        _ = Density.from_structure(
            self.structure_path,
            shape=(30, 30, 30),
            sampling_rate=6,
            origin=self.density.origin,
        )
        _ = Density.from_structure(
            self.structure_path,
            shape=(30, 30, 30),
            sampling_rate=None,
            origin=self.density.origin,
        )
        _ = Density.from_structure(
            self.structure_path, weight_type="atomic_weight", chain="A"
        )

    @pytest.mark.parametrize(
        "weight_type",
        [
            "atomic_weight",
            "atomic_number",
            "van_der_waals_radius",
            "scattering_factors",
            "lowpass_scattering_factors",
        ],
    )
    def test_from_structure_weight_types(self, weight_type):
        _ = Density.from_structure(
            self.structure_path,
            weight_type=weight_type,
        )

    def test_from_structure_weight_types_error(self):
        with pytest.raises(NotImplementedError):
            _ = Density.from_structure(
                self.structure_path,
                weight_type=None,
            )

    @pytest.mark.parametrize(
        "weight_type", ["scattering_factors", "lowpass_scattering_factors"]
    )
    @pytest.mark.parametrize(
        "scattering_factors", ["dt1969", "wk1995", "peng1995", "peng1999"]
    )
    def test_from_structure_scattering(self, scattering_factors, weight_type):
        _ = Density.from_structure(
            self.structure_path,
            weight_type=weight_type,
            scattering_args={"source": scattering_factors},
        )

    def test_from_structure_error(self):
        with pytest.raises(NotImplementedError):
            _ = Density.from_structure(self.structure_path, weight_type="RAISERROR")
        with pytest.raises(ValueError):
            _ = Density.from_structure(self.structure_path, sampling_rate=(1, 5))

    def test_empty(self):
        empty_density = self.density.empty
        assert np.allclose(empty_density.data, np.zeros_like(empty_density.data))
        assert np.allclose(empty_density.sampling_rate, self.density.sampling_rate)
        assert np.allclose(empty_density.origin, self.density.origin)
        assert empty_density.metadata == {"min": 0, "max": 0, "mean": 0, "std": 0}

    def test_copy(self):
        copied_density = self.density.copy()
        assert np.allclose(copied_density.data, self.density.data)
        assert np.allclose(copied_density.sampling_rate, self.density.sampling_rate)
        assert np.allclose(copied_density.origin, self.density.origin)
        assert copied_density.metadata == self.density.metadata

    def test_to_memmap(self):
        filename = self.path

        temp = self.density.copy()
        shape, dtype = temp.data.shape, temp.data.dtype

        arr_memmap = np.memmap(filename, mode="w+", dtype=dtype, shape=shape)
        arr_memmap[:] = temp.data[:]
        arr_memmap.flush()
        arr_memmap = np.memmap(filename, mode="r", dtype=dtype, shape=shape)

        temp.to_memmap()

        assert np.allclose(temp.data, arr_memmap)

    def test_to_numpy(self):
        temp = self.density.copy()
        temp.to_memmap()
        temp.to_numpy()

        assert np.allclose(temp.data, self.density.data)

    @pytest.mark.parametrize("threshold", [(0), (0.5), (1), (1.5)])
    def test_to_pointcloud(self, threshold):
        indices = self.density.to_pointcloud(threshold=threshold)
        assert indices.shape[0] == self.density.data.ndim
        assert np.all(self.density.data[tuple(indices)] > threshold)

    def test__pad_slice(self):
        x, y, z = self.density.shape
        box = (slice(-5, z + 5), slice(0, y - 5), slice(2, z + 2))
        padded_data = self.density._pad_slice(box)
        assert padded_data.shape == (60, y, z)

    def test_adjust_box(self):
        box = (slice(10, 40), slice(10, 40), slice(10, 40))
        self.density.adjust_box(box)
        assert self.density.data.shape == (30, 30, 30)
        np.testing.assert_array_equal(
            self.density.origin, DEFAULT_ORIGIN + 10 * DEFAULT_SAMPLING_RATE
        )

    def test_trim_box(self):
        toy_data = np.zeros((20, 20, 20))
        signal = np.ones((5, 5, 5))
        toy_data[0:5, 5:10, 10:15] = signal
        temp = self.density.empty
        temp.data = toy_data
        trim_box = temp.trim_box(cutoff=0.5, margin=0)
        trim_box_margin = temp.trim_box(cutoff=0.5, margin=2)
        temp.adjust_box(trim_box)
        assert np.allclose(temp.data, signal)
        assert trim_box_margin == tuple((slice(0, 7), slice(3, 12), slice(8, 17)))

    def test_pad(self):
        new_shape = (70, 70, 70)
        self.density.pad(new_shape)
        assert self.density.data.shape == new_shape

    @pytest.mark.parametrize("new_shape", [(70,), (70, 70), (70, 70, 70, 70)])
    def test_pad_error(self, new_shape):
        with pytest.raises(ValueError):
            self.density.pad(new_shape)

    def test_minimum_enclosing_box(self):
        # The exact shape may vary, so we will mainly ensure that
        # the data is correctly adapted.
        # Further, more precise tests could be added.
        temp = self.density.copy()
        box = temp.minimum_enclosing_box(cutoff=0.5)
        assert len(box) == temp.data.ndim
        temp = self.density.copy()
        box = temp.minimum_enclosing_box(cutoff=0.5, use_geometric_center=True)
        assert len(box) == temp.data.ndim

    @pytest.mark.parametrize(
        "cutoff", [DEFAULT_DATA.min() - 1, 0, DEFAULT_DATA.max() - 0.1]
    )
    def test_centered(self, cutoff):
        centered_density, translation = self.density.centered(cutoff=cutoff)
        data = centered_density.data
        data[data < cutoff] = 0
        com = centered_density.center_of_mass(data)

        difference = np.abs(
            np.subtract(
                np.rint(np.array(com)).astype(int),
                np.array(centered_density.shape) // 2,
            )
        )
        assert np.all(difference <= self.density.sampling_rate)

    @pytest.mark.parametrize("use_geometric_center", (True, False))
    @pytest.mark.parametrize("create_mask", (True, False))
    @pytest.mark.parametrize("order", (None, 1, 3))
    def test_rotate_array(
        self, use_geometric_center: bool, create_mask: bool, order: int
    ):
        rotation_matrix = np.eye(self.density.data.ndim)
        rotation_matrix[0, 0] = -1

        temp = self.density.copy()
        temp.adjust_box(temp.trim_box(cutoff=0))

        if use_geometric_center:
            box = temp.minimum_enclosing_box(cutoff=0, use_geometric_center=True)
            temp.adjust_box(box)
        else:
            temp, translation = temp.centered(cutoff=0)

        out = np.zeros_like(temp.data)
        arr_mask, out_mask = None, None
        if create_mask:
            mask = temp.copy()
            mask.data[mask.data < 0.5] = 0
            mask.data[mask.data >= 0.5] = 1
            arr_mask = mask.data
            out_mask = np.zeros_like(mask.data)

        Density.rotate_array(
            arr=temp.data,
            rotation_matrix=rotation_matrix,
            translation=np.zeros(temp.data.ndim),
            use_geometric_center=use_geometric_center,
            arr_mask=arr_mask,
            out_mask=out_mask,
            out=out,
            order=order,
        )

        ret = Density.rotate_array(
            arr=temp.data,
            rotation_matrix=rotation_matrix,
            translation=np.zeros(temp.data.ndim),
            use_geometric_center=use_geometric_center,
            arr_mask=arr_mask,
            out_mask=None,
            out=None,
            order=order,
        )

        out2 = np.zeros_like(out)
        extra = Density.rotate_array(
            arr=temp.data,
            rotation_matrix=rotation_matrix,
            translation=np.zeros(temp.data.ndim),
            use_geometric_center=use_geometric_center,
            arr_mask=arr_mask,
            out_mask=None,
            out=out2,
            order=order,
        )

        if create_mask:
            ret, ret_mask = ret
            ret_mask = Density(
                ret_mask,
                origin=self.density.origin,
                sampling_rate=self.density.sampling_rate,
            )
            out_mask = Density(
                out_mask,
                origin=self.density.origin,
                sampling_rate=self.density.sampling_rate,
            )
            ret_mask.adjust_box(ret_mask.trim_box(0))
            out_mask.adjust_box(out_mask.trim_box(0))
            assert np.allclose(ret_mask.data, out_mask.data, rtol=0.5)
            assert np.allclose(out, out2)

            extra = Density(
                extra,
                origin=self.density.origin,
                sampling_rate=self.density.sampling_rate,
            )
            extra.adjust_box(extra.trim_box(0))
            ret_mask.adjust_box(ret_mask.trim_box(0))
            assert np.allclose(ret_mask.data, extra.data)

        ret = Density(
            ret, origin=self.density.origin, sampling_rate=self.density.sampling_rate
        )
        out = Density(
            out, origin=self.density.origin, sampling_rate=self.density.sampling_rate
        )
        ret.adjust_box(ret.trim_box(0))
        out.adjust_box(out.trim_box(0))
        assert np.allclose(ret.data, out.data)

    @pytest.mark.parametrize("use_geometric_center", (False, True))
    @pytest.mark.parametrize("create_mask", (False, True))
    def test_rotate_array_coordinates(
        self, use_geometric_center: bool, create_mask: bool
    ):
        rotation_matrix = np.eye(self.density.data.ndim)
        rotation_matrix[0, 0] = -1

        temp = self.density.copy()
        temp.adjust_box(temp.trim_box(cutoff=0))
        if use_geometric_center:
            box = temp.minimum_enclosing_box(cutoff=0, use_geometric_center=True)
            temp.adjust_box(box)
        else:
            temp, translation = temp.centered(cutoff=0)

        out = np.zeros_like(temp.data)
        arr_mask, mask_coordinates, out_mask = None, None, None
        if create_mask:
            mask = temp.copy()
            mask.data[mask.data < 0.5] = 0
            mask.data[mask.data >= 0.5] = 1
            arr_mask = mask.data
            mask_coordinates = mask.to_pointcloud(threshold=0)
            out_mask = np.zeros_like(mask.data)

        Density.rotate_array_coordinates(
            arr=temp.data,
            coordinates=temp.to_pointcloud(threshold=0),
            mask_coordinates=mask_coordinates,
            rotation_matrix=rotation_matrix,
            translation=np.zeros(temp.data.ndim),
            use_geometric_center=use_geometric_center,
            arr_mask=arr_mask,
            out_mask=out_mask,
            out=out,
        )

        ret = Density.rotate_array_coordinates(
            arr=temp.data,
            coordinates=temp.to_pointcloud(threshold=0),
            mask_coordinates=mask_coordinates,
            rotation_matrix=rotation_matrix,
            translation=np.zeros(temp.data.ndim),
            use_geometric_center=use_geometric_center,
            arr_mask=arr_mask,
            out_mask=None,
            out=None,
        )

        out2 = np.zeros_like(out)
        extra = Density.rotate_array_coordinates(
            arr=temp.data,
            coordinates=temp.to_pointcloud(threshold=0),
            mask_coordinates=mask_coordinates,
            rotation_matrix=rotation_matrix,
            translation=np.zeros(temp.data.ndim),
            use_geometric_center=use_geometric_center,
            arr_mask=arr_mask,
            out_mask=None,
            out=out2,
        )

        if create_mask:
            ret, ret_mask = ret
            ret_mask = Density(
                ret_mask,
                origin=self.density.origin,
                sampling_rate=self.density.sampling_rate,
            )
            out_mask = Density(
                out_mask,
                origin=self.density.origin,
                sampling_rate=self.density.sampling_rate,
            )
            ret_mask.adjust_box(ret_mask.trim_box(0))
            out_mask.adjust_box(out_mask.trim_box(0))

            assert np.allclose(ret_mask.data, out_mask.data)
            assert np.allclose(out, out2)

            extra = Density(
                extra,
                origin=self.density.origin,
                sampling_rate=self.density.sampling_rate,
            )
            extra.adjust_box(extra.trim_box(0))
            ret_mask.adjust_box(ret_mask.trim_box(0))
            assert np.allclose(ret_mask.data, extra.data)

        ret = Density(
            ret, origin=self.density.origin, sampling_rate=self.density.sampling_rate
        )
        out = Density(
            out, origin=self.density.origin, sampling_rate=self.density.sampling_rate
        )
        ret.adjust_box(ret.trim_box(0))
        out.adjust_box(out.trim_box(0))
        assert np.allclose(ret.data, out.data)

    @pytest.mark.parametrize("use_geometric_center", (True, False))
    def test_rigid_transform(self, use_geometric_center: bool):
        temp = self.density.copy()
        if use_geometric_center:
            box = temp.minimum_enclosing_box(cutoff=0, use_geometric_center=True)
            print(box)
            temp.adjust_box(box)
        else:
            temp, translation = temp.centered()

        swaps = set(permutations([0, 1, 2]))
        temp_matrix = np.eye(temp.data.ndim).astype(np.float32)
        rotation_matrix = np.zeros_like(temp_matrix)

        initial_weight = np.sum(np.abs(temp.data))
        for z, y, x in swaps:
            rotation_matrix[:, 0] = temp_matrix[:, z]
            rotation_matrix[:, 1] = temp_matrix[:, y]
            rotation_matrix[:, 2] = temp_matrix[:, x]

            transformed = temp.rigid_transform(
                rotation_matrix=rotation_matrix,
                translation=np.zeros(temp.data.ndim),
                use_geometric_center=use_geometric_center,
            )
            transformed_weight = np.sum(np.abs(transformed.data))
            print(initial_weight, transformed_weight)
            assert np.abs(1 - initial_weight / transformed_weight) < 0.01

    def test_align_origins_same_apix(self):
        map1 = self.density.copy()
        map2 = self.density.copy()
        map2.origin = np.add(map1.origin, (5, 1, 3))

        map3 = map1.align_origins(map2)

        assert np.array_equal(map1.origin, map3.origin)

    def test_align_origins_different_apix(self):
        map1 = self.density.copy()
        map2 = self.density.copy()
        map2.sampling_rate = np.multiply(map2.sampling_rate, 2)
        map2.origin = np.add(map1.origin, (5, 1, 3))

        with pytest.raises(ValueError):
            map1.align_origins(map2)

    @pytest.mark.parametrize(
        "new_sampling_rate,order",
        [(2, 1), (4, 3)],
    )
    def test_resample(self, new_sampling_rate, order):
        resampled = self.density.resample(
            new_sampling_rate=new_sampling_rate, order=order
        )
        assert np.allclose(
            resampled.shape,
            np.divide(self.density.shape, new_sampling_rate).astype(int),
        )

    @pytest.mark.parametrize(
        "fraction_surface,volume_factor",
        [(0.5, 2), (1, 3)],
    )
    def test_density_boundary(self, fraction_surface, volume_factor):
        # TODO: Pre compute volume boundary on real data
        boundary = self.density.density_boundary(
            weight=1000, fraction_surface=fraction_surface, volume_factor=volume_factor
        )
        assert boundary[0] < boundary[1]

    @pytest.mark.parametrize(
        "fraction_surface,volume_factor",
        [(-0.5, 0), (1, -3)],
    )
    def test_density_boundary_error(self, fraction_surface, volume_factor):
        with pytest.raises(ValueError):
            _ = self.density.density_boundary(
                weight=1000,
                fraction_surface=fraction_surface,
                volume_factor=volume_factor,
            )

    @pytest.mark.parametrize(
        "method",
        [("ConvexHull"), ("Weight"), ("Sobel"), ("Laplace"), ("Minimum")],
    )
    def test_surface_coordinates(self, method):
        density_boundaries = self.density.density_boundary(weight=1000)
        self.density.surface_coordinates(
            density_boundaries=density_boundaries, method=method
        )

    def test_surface_coordinates_error(self):
        density_boundaries = self.density.density_boundary(weight=1000)
        with pytest.raises(ValueError):
            self.density.surface_coordinates(
                density_boundaries=density_boundaries, method=None
            )

    def test_normal_vectors(self):
        density_boundaries = self.density.density_boundary(weight=1000)
        coordinates = self.density.surface_coordinates(
            density_boundaries=density_boundaries, method="ConvexHull"
        )
        self.density.normal_vectors(coordinates=coordinates)

    def test_normal_vectors_error(self):
        coordinates = np.random.rand(10, 10, 10)
        with pytest.raises(ValueError):
            self.density.normal_vectors(coordinates=coordinates)

        coordinates = np.random.rand(10, 4)
        with pytest.raises(ValueError):
            self.density.normal_vectors(coordinates=coordinates)

        coordinates = np.random.rand(10, 3) * -10
        with pytest.raises(ValueError):
            self.density.normal_vectors(coordinates=coordinates)

    def test_core_mask(self):
        mask = self.density.core_mask()
        assert mask.sum() > 0

    def test_center_of_mass(self):
        center, shape, radius = (10, 10), (20, 20), 5
        n = len(shape)
        position = np.array(center).reshape((-1,) + (1,) * n)
        arr = np.linalg.norm(np.indices(shape) - position, axis=0)
        arr = (arr <= radius).astype(np.float32)

        center_of_mass = Density.center_of_mass(arr)
        assert np.allclose(center, center_of_mass)

    @pytest.mark.parametrize(
        "method",
        [
            ("CrossCorrelation"),
            ("NormalizedCrossCorrelation"),
        ],
    )
    def test_match_densities(self, method: str):
        target = np.zeros((30, 30, 30))
        target[5:10, 15:22, 10:13] = 1

        target = Density(target, sampling_rate=(1, 1, 1), origin=(0, 0, 0))
        target, translation = target.centered(cutoff=0)

        template = target.copy()

        initial_translation = np.array([-1, 3, 0])
        initial_rotation = euler_to_rotationmatrix((180, 0, 0))

        template = template.rigid_transform(
            rotation_matrix=initial_rotation,
            translation=initial_translation,
            use_geometric_center=False,
        )

        target.sampling_rate = np.array(target.sampling_rate[0])
        template.sampling_rate = np.array(template.sampling_rate[0])

        aligned, *_ = Density.match_densities(
            target=target,
            template=template,
            cutoff_target=0.2,
            cutoff_template=0.2,
            scoring_method=method,
        )

        aligned = Density.align_coordinate_systems(target=target, template=aligned)

        assert np.allclose(target.data, aligned.data, rtol=0.5)

        template.sampling_rate = template.sampling_rate * 2

        aligned, translation, rotation = Density.match_densities(
            target=target,
            template=template,
            cutoff_target=0.2,
            cutoff_template=0.2,
            scoring_method=method,
        )

    def test_match_structure_to_density(self):
        density = Density.from_file("tme/tests/data/Maps/emd_8621.mrc.gz")
        density = density.resample(density.sampling_rate * 4)
        structure = Structure.from_file(
            "tme/tests/data/Structures/5uz4.cif", filter_by_residues=None
        )

        initial_translation = np.array([-1, 0, 5])
        initial_rotation = np.eye(density.data.ndim)
        structure.rigid_transform(
            translation=initial_translation, rotation_matrix=initial_rotation
        )

        ret = Density.match_structure_to_density(
            target=density,
            template=structure,
            cutoff_target=0.0309,
            scoring_method="CrossCorrelation",
        )
        structure_aligned, translation, rotation_matrix = ret

        assert np.allclose(
            structure_aligned.atom_coordinate.shape, structure.atom_coordinate.shape
        )

    def test_align_coordinate_systems(self):
        target = self.density.copy()
        target, translation = target.centered()
        template = target.copy()

        translation = np.array([5, 1, -3])
        template = template.rigid_transform(
            rotation_matrix=np.eye(template.data.ndim),
            translation=translation,
            use_geometric_center=False,
        )
        template.origin -= np.multiply(translation, template.sampling_rate)
        template_aligned = Density.align_coordinate_systems(
            target=target, template=template
        )

        assert np.allclose(target.origin, template_aligned.origin)
        assert np.allclose(target.data, template_aligned.data, rtol=0.5)

    def test_align_coordinate_systems_error(self):
        target = self.density.copy()
        target, translation = target.centered()
        template = target.copy()

        template.sampling_rate = np.multiply(target.sampling_rate, 2)

        with pytest.raises(ValueError):
            _ = Density.align_coordinate_systems(target=target, template=template)

    def test_fourier_shell_correlation(self):
        fsc = Density.fourier_shell_correlation(
            self.density.copy(), self.density.copy()
        )
        assert fsc.shape[1] == 2
