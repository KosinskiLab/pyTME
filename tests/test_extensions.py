import pytest
import numpy as np

from scipy.spatial import distance

from tme.extensions import (
    absolute_minimum_deviation,
    max_euclidean_distance,
    find_candidate_indices,
    find_candidate_coordinates,
    max_index_by_label,
    online_statistics,
)


COORDINATES, N_COORDINATES = {}, 50
for i in range(1, 4):
    COORDINATES[i] = np.random.choice(np.arange(100), size=50 * i).reshape(
        N_COORDINATES, i
    )

np.random.seed(42)
TEST_DATA = 10 * np.random.rand(5000)


class TestExtensions:
    @pytest.mark.parametrize("dimension", list(COORDINATES.keys()))
    @pytest.mark.parametrize("dtype", [np.int32, int, np.float32, np.float64])
    def test_absolute_minimum_deviation(self, dimension, dtype):
        coordinates = COORDINATES[dimension].astype(dtype)
        output = np.zeros(
            (coordinates.shape[0], coordinates.shape[0]), dtype=coordinates.dtype
        )
        absolute_minimum_deviation(coordinates=coordinates, output=output)
        expected_output = distance.cdist(
            coordinates, coordinates, lambda u, v: np.min(np.abs(u - v))
        )
        assert np.allclose(output, expected_output)

    @pytest.mark.parametrize("dimension", list(COORDINATES.keys()))
    @pytest.mark.parametrize("dtype", [np.int32, int, np.float32, np.float64])
    def test_max_euclidean_distance(self, dimension, dtype):
        coordinates = COORDINATES[dimension].astype(dtype)
        print(coordinates.shape)

        max_distance, pair = max_euclidean_distance(coordinates=coordinates)
        distances = distance.cdist(coordinates, coordinates, "euclidean")
        distances_max = distances.max()
        assert np.allclose(max_distance, distances_max)

    @pytest.mark.parametrize("dimension", list(COORDINATES.keys()))
    @pytest.mark.parametrize("dtype", [np.int32, int, np.float32, np.float64])
    @pytest.mark.parametrize("min_distance", [0, 5, 10])
    def test_find_candidate_indices(self, dimension, dtype, min_distance):
        coordinates = COORDINATES[dimension].astype(dtype)
        print(coordinates.shape)

        min_distance = np.array([min_distance]).astype(dtype)[0]

        candidates = find_candidate_indices(
            coordinates=coordinates, min_distance=min_distance
        )

        distances = distance.cdist(coordinates[candidates], coordinates[candidates])
        np.fill_diagonal(distances, np.inf)
        assert np.all(distances >= min_distance)

    @pytest.mark.parametrize("dimension", list(COORDINATES.keys()))
    @pytest.mark.parametrize("dtype", [np.int32, int, np.float32, np.float64])
    @pytest.mark.parametrize("min_distance", [0, 5, 10])
    def test_find_candidate_coordinates(self, dimension, dtype, min_distance):
        coordinates = COORDINATES[dimension].astype(dtype)
        print(coordinates.shape)

        min_distance = np.array([min_distance]).astype(dtype)[0]

        filtered_coordinates = find_candidate_coordinates(
            coordinates=coordinates, min_distance=min_distance
        )

        distances = distance.cdist(filtered_coordinates, filtered_coordinates)
        np.fill_diagonal(distances, np.inf)
        assert np.all(distances >= min_distance)

    @pytest.mark.parametrize("dtype_labels", [np.int32, int, np.float32, np.float64])
    @pytest.mark.parametrize("dtype_scores", [np.int32, int, np.float32, np.float64])
    def test_max_index_by_label(self, dtype_labels, dtype_scores):
        labels = np.array([1, 1, 2, 3, 3, 3, 2], dtype=dtype_labels)
        scores = np.array([0.5, 0.8, 0.7, 0.2, 0.9, 0.6, 0.5])
        scores = (10 * scores).astype(dtype_scores)
        expected_result = {}
        for label in np.unique(labels):
            mask = labels == label
            expected_result[label] = np.argmax(scores * mask)

        ret = max_index_by_label(labels=labels, scores=scores)
        print(ret)
        for k in expected_result:
            assert np.allclose(expected_result[k], ret[k])

    @pytest.mark.parametrize("dtype", [np.int32, int, np.float32, np.float64])
    @pytest.mark.parametrize("splits", [1, 5, 10])
    def test_online_statistics(self, dtype, splits):
        parts = np.array_split(TEST_DATA, splits)
        n, rmean, ssqd, reference = 0, 0, 0, 0.0
        better_or_equal = 0
        end_idx = 0
        for part in parts:
            start_idx = end_idx
            end_idx = start_idx + len(part)

            n, rmean, ssqd, nbetter_or_equal, max_value, min_value = online_statistics(
                arr=part, n=n, rmean=rmean, ssqd=ssqd
            )
            better_or_equal += nbetter_or_equal

            data = TEST_DATA[:end_idx]
            dn = len(data)
            drmean = np.mean(data)
            dssqd = np.sum((data - rmean) ** 2)
            dnbetter_or_equal = np.sum(data >= reference)
            dmax_value = np.max(data)
            dmin_value = np.min(data)

            assert n == dn
            assert np.isclose(rmean, drmean, atol=1e-8)
            assert np.isclose(ssqd, dssqd, atol=1e-8)
            assert better_or_equal == dnbetter_or_equal
            assert np.isclose(max_value, dmax_value, atol=1e-1)
            assert np.isclose(min_value, dmin_value, atol=1e-1)
