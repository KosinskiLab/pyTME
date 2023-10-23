/*  Pybind extensions for template matching score space analyzers.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
*/

#include <vector>
#include <iostream>
#include <limits>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename T>
void absolute_minimum_deviation(
    py::array_t<T, py::array::c_style> coordinates,
    py::array_t<T, py::array::c_style> output) {
    auto coordinates_data = coordinates.data();
    auto output_data = output.mutable_data();
    int n = coordinates.shape(0);
    int k = coordinates.shape(1);
    int ik, jk, in, jn;

    for (int i = 0; i < n; ++i) {
        ik = i * k;
        in = i * n;
        for (int j = i + 1; j < n; ++j) {
            jk = j * k;
            jn = j * n;
            T min_distance = std::abs(coordinates_data[ik] - coordinates_data[jk]);
            for (int p = 1; p < k; ++p) {
                min_distance = std::min(min_distance,
                    std::abs(coordinates_data[ik + p] - coordinates_data[jk + p]));
            }
            output_data[in + j] = min_distance;
            output_data[jn + i] = min_distance;
        }
        output_data[in + i] = 0;
    }
}

template <typename T>
std::pair<double, std::pair<int, int>> max_euclidean_distance(
    py::array_t<T, py::array::c_style> coordinates) {
    auto coordinates_data = coordinates.data();
    int n = coordinates.shape(0);
    int k = coordinates.shape(1);

    double distance = 0.0;
    double difference = 0.0;
    double max_distance = -1;
    double squared_distances = 0.0;

    int ik, jk;
    int max_i = -1, max_j = -1;

    for (int i = 0; i < n; ++i) {
        ik = i * k;
        for (int j = i + 1; j < n; ++j) {
            jk = j * k;
            squared_distances = 0.0;
            for (int p = 0; p < k; ++p) {
                difference = static_cast<double>(
                    coordinates_data[ik + p] - coordinates_data[jk + p]
                );
                squared_distances += (difference * difference);
            }
            distance = std::sqrt(squared_distances);
            if (distance > max_distance) {
                max_distance = distance;
                max_i = i;
                max_j = j;
            }
        }
    }

    return std::make_pair(max_distance, std::make_pair(max_i, max_j));
}


template <typename T>
inline py::array_t<int, py::array::c_style> find_candidate_indices(
    py::array_t<T, py::array::c_style> coordinates,
    T min_distance) {
    auto coordinates_data = coordinates.data();
    int n = coordinates.shape(0);
    int k = coordinates.shape(1);
    int ik, jk;

    std::vector<int> candidate_indices;
    candidate_indices.reserve(n / 2);
    candidate_indices.push_back(0);

    for (int i = 1; i < n; ++i) {
        bool is_candidate = true;
        ik = i * k;
        for (int candidate_index : candidate_indices) {
            jk = candidate_index * k;
            T distance = std::abs(coordinates_data[ik] - coordinates_data[jk]);
            for (int p = 1; p < k; ++p) {
                distance = std::max(distance,
                    std::abs(coordinates_data[ik + p] - coordinates_data[jk + p]));
            }
            if (distance <= min_distance) {
                is_candidate = false;
                break;
            }
        }
        if (is_candidate) {
            candidate_indices.push_back(i);
        }
    }

    py::array_t<int, py::array::c_style> output({(int)candidate_indices.size()});
    auto output_data = output.mutable_data();

    for (size_t i = 0; i < candidate_indices.size(); ++i) {
        output_data[i] = candidate_indices[i];
    }

    return output;
}

template <typename T>
py::array_t<T, py::array::c_style> find_candidate_coordinates(
    py::array_t<T, py::array::c_style> coordinates,
    T min_distance) {

    py::array_t<int, py::array::c_style> candidate_indices_array = find_candidate_indices(
        coordinates, min_distance);
    auto candidate_indices_data = candidate_indices_array.data();
    int num_candidates = candidate_indices_array.shape(0);
    int k = coordinates.shape(1);
    auto coordinates_data = coordinates.data();

    py::array_t<T, py::array::c_style> output({num_candidates, k});
    auto output_data = output.mutable_data();

    for (int i = 0; i < num_candidates; ++i) {
        int candidate_index = candidate_indices_data[i] * k;
        std::copy(
            coordinates_data + candidate_index,
            coordinates_data + candidate_index + k,
            output_data + i * k
        );
    }

    return output;
}

template <typename U, typename T>
py::dict max_index_by_label(
    py::array_t<U, py::array::c_style> labels,
    py::array_t<T, py::array::c_style> scores
    ) {

    const U* labels_ptr = labels.data();
    const T* scores_ptr = scores.data();

    std::unordered_map<U, std::pair<T, ssize_t>> max_scores;

    U label;
    T score;
    for (ssize_t i = 0; i < labels.size(); ++i) {
        label = labels_ptr[i];
        score = scores_ptr[i];

        auto it = max_scores.insert({label, {score, i}});

        if (score > it.first->second.first) {
            it.first->second = {score, i};
        }
    }

    py::dict ret;
    for (auto& item: max_scores) {
        ret[py::cast(item.first)] = py::cast(item.second.second);
    }

    return ret;
}


template <typename T>
py::tuple online_statistics(
    py::array_t<T, py::array::c_style> arr,
    unsigned long long int n = 0,
    double rmean = 0,
    double ssqd = 0,
    T reference = 0) {

    auto in = arr.data();
    int size = arr.size();

    T max_value = std::numeric_limits<T>::lowest();
    T min_value = std::numeric_limits<T>::max();
    double delta, delta_prime;

    unsigned long long int nbetter_or_equal = 0;

    for(int i = 0; i < size; i++){
        n++;
        delta = in[i] - rmean;
        rmean += delta / n;
        delta_prime = in[i] - rmean;
        ssqd += delta * delta_prime;

        max_value = std::max(in[i], max_value);
        min_value = std::min(in[i], min_value);
        if (in[i] >= reference)
            nbetter_or_equal++;
    }

    return py::make_tuple(n, rmean, ssqd, nbetter_or_equal, max_value, min_value);
}

PYBIND11_MODULE(extensions, m) {

    m.def("absolute_minimum_deviation", absolute_minimum_deviation<double>,
        "Compute pairwise absolute minimum deviation for a set of coordinates (float64).",
        py::arg("coordinates"), py::arg("output"));
    m.def("absolute_minimum_deviation", absolute_minimum_deviation<float>,
        "Compute pairwise absolute minimum deviation for a set of coordinates (float32).",
        py::arg("coordinates"), py::arg("output"));
    m.def("absolute_minimum_deviation", absolute_minimum_deviation<int64_t>,
        "Compute pairwise absolute minimum deviation for a set of coordinates (int64).",
        py::arg("coordinates"), py::arg("output"));
    m.def("absolute_minimum_deviation", absolute_minimum_deviation<int32_t>,
        "Compute pairwise absolute minimum deviation for a set of coordinates (int32).",
        py::arg("coordinates"), py::arg("output"));


    m.def("max_euclidean_distance", max_euclidean_distance<double>,
        "Identify pair of points with maximal euclidean distance (float64).",
        py::arg("coordinates"));
    m.def("max_euclidean_distance", max_euclidean_distance<float>,
        "Identify pair of points with maximal euclidean distance (float32).",
          py::arg("coordinates"));
    m.def("max_euclidean_distance", max_euclidean_distance<int64_t>,
        "Identify pair of points with maximal euclidean distance (int64).",
          py::arg("coordinates"));
    m.def("max_euclidean_distance", max_euclidean_distance<int32_t>,
        "Identify pair of points with maximal euclidean distance (int32).",
          py::arg("coordinates"));


    m.def("find_candidate_indices", &find_candidate_indices<double>,
          "Finds candidate indices with minimum distance (float64).",
          py::arg("coordinates"), py::arg("min_distance"));
    m.def("find_candidate_indices", &find_candidate_indices<float>,
          "Finds candidate indices with minimum distance (float32).",
          py::arg("coordinates"), py::arg("min_distance"));
    m.def("find_candidate_indices", &find_candidate_indices<int64_t>,
          "Finds candidate indices with minimum distance (int64).",
          py::arg("coordinates"), py::arg("min_distance"));
    m.def("find_candidate_indices", &find_candidate_indices<int32_t>,
          "Finds candidate indices with minimum distance (int32).",
          py::arg("coordinates"), py::arg("min_distance"));


    m.def("find_candidate_coordinates", &find_candidate_coordinates<double>,
          "Finds candidate coordinates with minimum distance (float64).",
          py::arg("coordinates"), py::arg("min_distance"));
    m.def("find_candidate_coordinates", &find_candidate_coordinates<float>,
          "Finds candidate coordinates with minimum distance (float32).",
          py::arg("coordinates"), py::arg("min_distance"));
    m.def("find_candidate_coordinates", &find_candidate_coordinates<int64_t>,
          "Finds candidate coordinates with minimum distance (int64).",
          py::arg("coordinates"), py::arg("min_distance"));
    m.def("find_candidate_coordinates", &find_candidate_coordinates<int32_t>,
          "Finds candidate coordinates with minimum distance (int32).",
          py::arg("coordinates"), py::arg("min_distance"));


    m.def("max_index_by_label", &max_index_by_label<double, double>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));
    m.def("max_index_by_label", &max_index_by_label<double, float>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));
    m.def("max_index_by_label", &max_index_by_label<double, int64_t>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));
    m.def("max_index_by_label", &max_index_by_label<double, int32_t>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));

    m.def("max_index_by_label", &max_index_by_label<float, double>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));
    m.def("max_index_by_label", &max_index_by_label<float, float>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));
    m.def("max_index_by_label", &max_index_by_label<float, int64_t>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));
    m.def("max_index_by_label", &max_index_by_label<float, int32_t>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));

    m.def("max_index_by_label", &max_index_by_label<int64_t, double>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));
    m.def("max_index_by_label", &max_index_by_label<int64_t, float>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));
    m.def("max_index_by_label", &max_index_by_label<int64_t, int64_t>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));
    m.def("max_index_by_label", &max_index_by_label<int64_t, int32_t>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));

    m.def("max_index_by_label", &max_index_by_label<int32_t, double>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));
    m.def("max_index_by_label", &max_index_by_label<int32_t, float>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));
    m.def("max_index_by_label", &max_index_by_label<int32_t, int64_t>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));
    m.def("max_index_by_label", &max_index_by_label<int32_t, int32_t>,
          "Maximum value by label", py::arg("labels"), py::arg("scores"));


    m.def("online_statistics", &online_statistics<double>, py::arg("arr"),
          py::arg("n") = 0, py::arg("rmean") = 0,
          py::arg("ssqd") = 0, py::arg("reference") = 0,
          "Compute running online statistics on a numpy array.");
    m.def("online_statistics", &online_statistics<float>, py::arg("arr"),
          py::arg("n") = 0, py::arg("rmean") = 0,
          py::arg("ssqd") = 0, py::arg("reference") = 0,
          "Compute running online statistics on a numpy array.");
    m.def("online_statistics", &online_statistics<int64_t>, py::arg("arr"),
          py::arg("n") = 0, py::arg("rmean") = 0,
          py::arg("ssqd") = 0, py::arg("reference") = 0,
          "Compute running online statistics on a numpy array.");
    m.def("online_statistics", &online_statistics<int32_t>, py::arg("arr"),
          py::arg("n") = 0, py::arg("rmean") = 0,
          py::arg("ssqd") = 0, py::arg("reference") = 0,
          "Compute running online statistics on a numpy array.");
}
