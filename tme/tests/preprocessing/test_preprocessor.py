import pytest
import numpy as np

from tme import Density, Structure, Preprocessor


class TestPreprocessor:
    def setup_method(self):
        self.density = Density.from_file(filename="./tme/tests/data/Raw/em_map.map")
        self.structure = Structure.from_file("./tme/tests/data/Structures/5khe.cif")
        self.structure_density = Density.from_structure(
            filename_or_structure="./tme/tests/data/Structures/5khe.cif",
            origin=self.density.origin,
            shape=self.density.shape,
            sampling_rate=self.density.sampling_rate,
        )
        self.preprocessor = Preprocessor()

    def teardown_method(self):
        self.density = None
        self.structure_density = None

    def test_initialization(self):
        _ = Preprocessor()

    def test_apply_method_error(self):
        with pytest.raises(TypeError):
            self.preprocessor.apply_method(method=None, parameters={})

        with pytest.raises(NotImplementedError):
            self.preprocessor.apply_method(method="None", parameters={})

    def test_method_to_id_error(self):
        with pytest.raises(TypeError):
            self.preprocessor.method_to_id(method=None, parameters={})

        with pytest.raises(NotImplementedError):
            self.preprocessor.method_to_id(method="None", parameters={})

    def test_method_to_id(self):
        ret = self.preprocessor.method_to_id(method="gaussian_filter", parameters={})
        assert isinstance(ret, str)

    @pytest.mark.parametrize("low_sigma,high_sigma", [(0, 1), (3, 5)])
    def test_difference_of_gaussian_filter(self, low_sigma, high_sigma):
        _ = self.preprocessor.difference_of_gaussian_filter(
            template=self.structure_density.data,
            low_sigma=low_sigma,
            high_sigma=high_sigma,
        )

    @pytest.mark.parametrize("smallest_size,largest_size", [(1, 10), (2, 20)])
    def test_bandpass_filter(self, smallest_size, largest_size):
        _ = self.preprocessor.bandpass_filter(
            template=self.structure_density.data,
            lowpass=smallest_size,
            highpass=largest_size,
            sampling_rate=1,
        )

    @pytest.mark.parametrize("lbd,sigma_range", [(1, (2, 4)), (20, (1, 6))])
    def test_local_gaussian_alignment_filter(self, lbd, sigma_range):
        _ = self.preprocessor.local_gaussian_alignment_filter(
            template=self.structure_density.data,
            target=self.density.data,
            lbd=lbd,
            sigma_range=sigma_range,
        )

    @pytest.mark.parametrize(
        "lbd,sigma_range,gaussian_sigma", [(1, (2, 4), 1), (20, (1, 6), 3)]
    )
    def test_local_gaussian_filter(self, lbd, sigma_range, gaussian_sigma):
        _ = self.preprocessor.local_gaussian_filter(
            template=self.structure_density.data,
            lbd=lbd,
            sigma_range=sigma_range,
            gaussian_sigma=gaussian_sigma,
        )

    @pytest.mark.parametrize(
        "edge_algorithm",
        ["sobel", "prewitt", "laplace", "gaussian", "gaussian_laplace"],
    )
    @pytest.mark.parametrize("reverse", [(True), (False)])
    def test_edge_gaussian_filter(self, edge_algorithm, reverse):
        _ = self.preprocessor.edge_gaussian_filter(
            template=self.structure_density.data,
            edge_algorithm=edge_algorithm,
            reverse=reverse,
            sigma=3,
        )

    @pytest.mark.parametrize("width", range(1, 9, 3))
    def test_mean_filter(self, width):
        _ = self.preprocessor.mean_filter(
            template=self.structure_density.data,
            width=width,
        )

    @pytest.mark.parametrize("width", range(1, 9, 3))
    def test_kaiserb_filter(self, width):
        _ = self.preprocessor.kaiserb_filter(
            template=self.structure_density.data,
            width=width,
        )

    @pytest.mark.parametrize("width", range(1, 9, 3))
    def test_blob_filter(self, width):
        _ = self.preprocessor.blob_filter(
            template=self.structure_density.data,
            width=width,
        )

    @pytest.mark.parametrize("width", range(1, 9, 3))
    def test_hamming_filter(self, width):
        _ = self.preprocessor.hamming_filter(
            template=self.structure_density.data,
            width=width,
        )

    @pytest.mark.parametrize("rank", range(1, 9, 3))
    def test_rank_filter(self, rank):
        _ = self.preprocessor.rank_filter(
            template=self.structure_density.data,
            rank=rank,
        )

    def test_wedge_mask(self):
        angles = np.zeros((3, 10))
        angles[0, :] = np.random.rand(angles.shape[1])
        _ = self.preprocessor.wedge_mask(
            shape=self.structure_density.data.shape,
            tilt_angles=angles,
        )

    def test_molmap(self):
        _ = self.preprocessor.molmap(
            coordinates=self.structure.atom_coordinate,
            weights=self.structure._get_atom_weights(self.structure.element_symbol),
            resolution=10,
        )

    @pytest.mark.parametrize("extrude_plane", [False, True])
    def test_continuous_wedge_mask(self, extrude_plane):
        _ = self.preprocessor.continuous_wedge_mask(
            start_tilt=50,
            stop_tilt=-40,
            shape=(50, 50, 50),
            extrude_plane=extrude_plane,
        )
