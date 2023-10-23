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
        assert type(ret) == str

    @pytest.mark.parametrize("n", [10, 100, 1000])
    @pytest.mark.parametrize("sampling_rate", range(1, 4))
    def test_fftfreqn(self, n, sampling_rate):
        assert np.allclose(
            self.preprocessor.fftfreqn(shape=(n,), sampling_rate=sampling_rate),
            np.abs(np.fft.ifftshift(np.fft.fftfreq(n=n, d=sampling_rate))),
        )

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
            minimum_frequency=smallest_size,
            maximum_frequency=largest_size,
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

    @pytest.mark.parametrize("sigma_range", [(2, 4), (1, 6)])
    def test_ntree_filter(self, sigma_range):
        _ = self.preprocessor.ntree_filter(
            template=self.structure_density.data,
            sigma_range=sigma_range,
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

    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_mipmap_filter(self, level):
        _ = self.preprocessor.mipmap_filter(
            template=np.random.rand(50, 50, 50),
            level=level,
        )

    @pytest.mark.parametrize("level", [0, 3, 5])
    def test_wavelet_filter(self, level):
        _ = self.preprocessor.wavelet_filter(
            template=self.structure_density.data,
            level=level,
        )

    def test_wedge_mask(self):
        angles = np.zeros((3, 10))
        angles[0, :] = np.random.rand(angles.shape[1])
        _ = self.preprocessor.wedge_mask(
            shape=self.structure_density.data.shape,
            tilt_angles=angles,
        )

    def test_fourier_crop(self):
        reciprocal_template_filter = np.ones_like(self.structure_density.data)
        _ = self.preprocessor.fourier_crop(
            template=self.structure_density.data,
            reciprocal_template_filter=reciprocal_template_filter,
        )

    def test_fourier_uncrop(self):
        reciprocal_template_filter = np.ones_like(self.structure_density.data)
        _ = self.preprocessor.fourier_uncrop(
            template=self.structure_density.data,
            reciprocal_template_filter=reciprocal_template_filter,
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


# @pytest.fixture(name="VarManager", scope="class")
# def variable_manager():
#     class VarManager:
#         def __init__(self):
#             self._e_map = Map.from_file(filename="./dge/tests/data/Raw/em_map.map")
#             self._s_map = Map.from_structure(
#                 filename_or_structure="./dge/tests/data/Structures/5khe.cif",
#                 origin=self._e_map.origin,
#                 shape=self._e_map.shape,
#                 sampling_rate=self._e_map.sampling_rate,
#             )

#     return VarManager()


# def test_gaussian(VarManager):
#     PB = ProteinBlurrer()
#     blur1 = PB.gaussian_blur(
#         template=VarManager._s_map.data, apix=VarManager._e_map.sampling_rate, sigma=2
#     )
#     blur2 = PB.gaussian_blur(
#         template=VarManager._s_map.data,
#         apix=VarManager._e_map.sampling_rate,
#         sigma_coeff=1,
#         resolution=2,
#     )
#     assert np.allclose(blur1, blur2)
#     valid = np.load("./dge/tests/data/Blurring/gaussian_sigma2.npy")
#     assert np.allclose(blur1, valid, atol=1e-6)


# def test_mean(VarManager):
#     PB = ProteinBlurrer()
#     blur = PB.mean_blur(template=VarManager._s_map.data, filter_size=5)
#     valid = np.load("./dge/tests/data/Blurring/mean_size5.npy")
#     assert np.allclose(blur, valid, atol=1e-6)


# def test_rank(VarManager):
#     PB = ProteinBlurrer()
#     blur = PB.rank_blur(template=VarManager._s_map.data, rank=3)
#     valid = np.load("./dge/tests/data/Blurring/rank_rank3.npy")
#     assert np.allclose(blur, valid, atol=1e-6)


# def test_ntree(VarManager):
#     PB = ProteinBlurrer()
#     blur = PB.ntree_blur(
#         template=VarManager._s_map.data,
#         apix=VarManager._s_map.sampling_rate,
#         sigma_range=(0.5, 10),
#         target=VarManager._s_map.data,
#     )
#     valid = np.load("./dge/tests/data/Blurring/ntree_sigma0510.npy")
#     assert np.allclose(blur, valid, atol=1e-6)


# def test_hamming(VarManager):
#     PB = ProteinBlurrer()
#     blur = PB.hamming_blur(template=VarManager._s_map.data, width=6)
#     valid = np.load("./dge/tests/data/Blurring/hamming_width6.npy")
#     assert np.allclose(blur, valid, atol=1e-6)


# def test_blob(VarManager):
#     PB = ProteinBlurrer()
#     blur = PB.blob_blur(template=VarManager._s_map.data, width=18)
#     valid = np.load("./dge/tests/data/Blurring/blob_width18.npy")
#     assert np.allclose(blur, valid, atol=1e-6)


# def test_kaiserb(VarManager):
#     PB = ProteinBlurrer()
#     blur = PB.kaiserb_blur(template=VarManager._s_map.data, width=18)
#     valid = np.load("./dge/tests/data/Blurring/kaiserb_width18.npy")
#     assert np.allclose(blur, valid, atol=1e-6)


# # TODO: CREATE GROUND TRUTH
# def test_edgegaussian_sobel(VarManager):
#     PB = ProteinBlurrer()
#     blur = PB.edge_gaussian(
#         template=VarManager._s_map.data,
#         apix=VarManager._s_map.sampling_rate,
#         sigma=3,
#         edge_algorithm="sobel",
#         reverse=False,
#     )
#     valid = np.load("./dge/tests/data/Blurring/edgegaussian_sigma3.npy")
#     assert True


# def test_edgegaussian_gaussian_laplace(VarManager):
#     PB = ProteinBlurrer()
#     blur = PB.edge_gaussian(
#         template=VarManager._s_map.data,
#         apix=VarManager._s_map.sampling_rate,
#         sigma=3,
#         edge_algorithm="gaussian_laplace",
#         reverse=False,
#     )
#     valid = np.load("./dge/tests/data/Blurring/edgegaussian_sigma3.npy")
#     assert True


# def test_edgegaussian_sobel_reverse(VarManager):
#     PB = ProteinBlurrer()
#     blur = PB.edge_gaussian(
#         template=VarManager._s_map.data,
#         apix=VarManager._s_map.sampling_rate,
#         sigma=3,
#         edge_algorithm="sobel",
#         reverse=True,
#     )
#     valid = np.load("./dge/tests/data/Blurring/edgegaussian_sigma3.npy")
#     assert True


# def test_edgegaussian_gaussian_laplace_reverse(VarManager):
#     PB = ProteinBlurrer()
#     blur = PB.edge_gaussian(
#         template=VarManager._s_map.data,
#         apix=VarManager._s_map.sampling_rate,
#         sigma=3,
#         edge_algorithm="gaussian_laplace",
#         reverse=True,
#     )
#     valid = np.load("./dge/tests/data/Blurring/edgegaussian_sigma3.npy")
#     assert True


# def test_gaussian_local_gaussian(VarManager):
#     PB = ProteinBlurrer()
#     blur = PB.gaussian_local_gaussian(
#         template=VarManager._s_map.data,
#         apix=VarManager._s_map.sampling_rate,
#         gaussian_sigma=3,
#         lbd=20,
#         sigma_range=(0.5, 10),
#     )
#     valid = np.load("./dge/tests/data/Blurring/localgaussian_sigma0510.npy")
#     assert np.allclose(blur, valid, atol=1e-6)
