#ifndef VERTEX_COVER_H
#define VERTEX_COVER_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct LeafBox {
    std::vector<int32_t> shape;
    std::vector<int32_t> pos;
    int32_t original_index;
};

py::list generate_candidates(
    py::tuple segmentation_shape,
    py::tuple valid_box_sizes,
    int32_t leaf_size,
    bool tile_candidates = false);

py::tuple setup_vertex_cover(
    py::list candidates,
    py::list active_leaves,
    py::tuple padding,
    int32_t leaf_size = 1);

py::list jitter_candidates(
    py::list candidates,
    py::array_t<int32_t> active_leaves,
    int32_t margin = 1,
    int32_t stride = 1);

py::list decompose_boxes(
    py::list current_boxes,
    py::tuple valid_box_sizes,
    int32_t leaf_size,
    bool nearest = false,
    bool overlapping = true);


void register_set_cover_bindings(py::module& m);

#endif