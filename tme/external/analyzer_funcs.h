/*  Template matching score space analyzer extensions.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
*/

#ifndef EXTENSIONS_H
#define EXTENSIONS_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename T>
void absolute_minimum_deviation(
    py::array_t<T, py::array::c_style> coordinates,
    py::array_t<T, py::array::c_style> output);

template <typename T>
std::pair<double, std::pair<int, int>> max_euclidean_distance(
    py::array_t<T, py::array::c_style> coordinates);

template <typename T>
py::array_t<int, py::array::c_style> find_candidate_indices(
    py::array_t<T, py::array::c_style> coordinates,
    T min_distance);

template <typename T>
py::array_t<T, py::array::c_style> find_candidate_coordinates(
    py::array_t<T, py::array::c_style> coordinates,
    T min_distance);

template <typename U, typename T>
py::dict max_index_by_label(
    py::array_t<U, py::array::c_style> labels,
    py::array_t<T, py::array::c_style> scores);

template <typename T>
py::tuple online_statistics(
    py::array_t<T, py::array::c_style> arr,
    unsigned long long int n,
    double rmean,
    double ssqd,
    T reference);

void register_analyzer_bindings(py::module& m);

#endif