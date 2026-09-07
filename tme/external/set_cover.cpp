/*  Set cover and box decomposition functions.

    Copyright (c) 2025 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
*/

#include "set_cover.h"

#include <cmath>
#include <vector>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace std {
    template<typename T>
    struct hash<std::vector<T>> {
        size_t operator()(const std::vector<T>& v) const {
            size_t seed = v.size();
            for (auto& i : v) {
                seed ^= std::hash<T>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
}

struct CandidateHash {
    size_t operator()(const std::pair<std::vector<int32_t>, std::vector<int32_t>>& p) const {
        size_t h1 = std::hash<std::vector<int32_t>>()(p.first);
        size_t h2 = std::hash<std::vector<int32_t>>()(p.second);
        return h1 ^ (h2 << 1);
    }
};


template<typename T>
inline std::vector<T> tuple_to_vector(py::tuple t, int32_t n_dims) {
    std::vector<T> vec(n_dims);
    for (int32_t d = 0; d < n_dims; ++d) {
        vec[d] = t[d].cast<T>();
    }
    return vec;
}

LeafBox to_leaf_coords(
    py::tuple candidate,
    int32_t leaf_size,
    int32_t n_dims,
    int32_t original_index = -1
) {
    py::tuple box_shape_tuple = candidate[0];
    py::tuple box_pos_tuple = candidate[1];

    LeafBox box;
    box.shape.resize(n_dims);
    box.pos.resize(n_dims);
    box.original_index = original_index;

    for (int32_t d = 0; d < n_dims; ++d) {
        int32_t shape = box_shape_tuple[d].cast<int32_t>();

        box.shape[d] = shape / leaf_size;
        box.pos[d] = box_pos_tuple[d].cast<int32_t>() / leaf_size;
    }
    return box;
}

std::unordered_map<std::vector<int32_t>, int32_t> build_leaf_map(py::array_t<int32_t> active_leaves) {
    int32_t n_leaves = active_leaves.shape(0);
    int32_t n_dims = active_leaves.shape(1);
    auto leaves_buf = active_leaves.unchecked<2>();

    std::unordered_map<std::vector<int32_t>, int32_t> leaf_to_idx;
    leaf_to_idx.reserve(n_leaves);

    for (int32_t i = 0; i < n_leaves; ++i) {
        std::vector<int32_t> leaf(n_dims);
        for (int32_t d = 0; d < n_dims; ++d) {
            leaf[d] = leaves_buf(i, d);
        }
        leaf_to_idx.emplace(leaf, i);
    }

    return leaf_to_idx;
}

template<typename Func>
inline void iterate_ranges(
    const std::vector<int32_t>& start,
    const std::vector<int32_t>& stop,
    const std::vector<int32_t>& step,
    int32_t n_dims,
    Func callback
) {
    std::vector<int32_t> current(start);

    while (true) {
        // Early exit if callback returns false
        if (!callback(current)) {
            return;
        }

        // Increment odometer-style right to left
        int32_t d = n_dims - 1;
        while (d >= 0) {
            current[d] += step[d];
            if (current[d] < stop[d]) {
                break;
            }
            current[d] = start[d];
            d--;
        }
        if (d < 0) break;
    }
}

template<typename Func>
void iterate_box_positions(
    const std::vector<int32_t>& box_pos,
    const std::vector<int32_t>& box_shape,
    int32_t n_dims,
    Func callback
) {
    std::vector<int32_t> box_end(n_dims);
    for (int32_t d = 0; d < n_dims; ++d) {
        box_end[d] = box_pos[d] + box_shape[d];
    }

    std::vector<int32_t> step(n_dims, 1);
    iterate_ranges(box_pos, box_end, step, n_dims, callback);
}


std::vector<int32_t>
leaf_assignment(
    const LeafBox& box,
    const std::unordered_map<std::vector<int32_t>, int32_t>& leaf_to_idx,
    int32_t n_dims,
    bool collect_indices = false
) {
    std::vector<int32_t> covered_indices;
    iterate_box_positions(box.pos, box.shape, n_dims, [&](const std::vector<int32_t>& current) {
        auto it = leaf_to_idx.find(current);
        if (it != leaf_to_idx.end()) {
            covered_indices.emplace_back(it->second);
            // Break iteration
            if (!collect_indices){
                return false;
            }
        }
        return true;
    });
    return covered_indices;
}

py::list generate_candidates(
    py::tuple segmentation_shape,
    py::tuple valid_box_sizes,
    int32_t leaf_size,
    bool tile_candidates
) {
    int32_t n_dims = py::len(segmentation_shape);
    std::vector<int32_t> seg_shape = tuple_to_vector<int32_t>(segmentation_shape, n_dims);

    py::list result;

    // Generate candidates for each box size
    for (size_t size_idx = 0; size_idx < valid_box_sizes.size(); ++size_idx) {
        py::tuple box_size_tuple = valid_box_sizes[size_idx];
        std::vector<int32_t> box_size(n_dims);
        std::vector<int32_t> box_scaled(n_dims);

        bool valid_size = true;
        for (int32_t d = 0; d < n_dims; ++d) {
            box_size[d] = box_size_tuple[d].cast<int32_t>();
            box_scaled[d] = box_size[d] / leaf_size;
            if (box_scaled[d] == 0) {
                valid_size = false;
                break;
            }
        }

        if (!valid_size) {
            continue;
        }

        std::vector<int32_t> starts(n_dims, 0);
        std::vector<int32_t> stops(n_dims);
        std::vector<int32_t> steps(n_dims, 1);

        for (int32_t d = 0; d < n_dims; ++d) {
            stops[d] = (seg_shape[d] + leaf_size - 1) / leaf_size;
            if (tile_candidates){
                steps[d] = box_scaled[d];
            }
        }
        iterate_ranges(starts, stops, steps, n_dims,
            [&](const std::vector<int32_t>& leaf_pos) {
                result.append(py::make_tuple(
                    py::cast(box_scaled),
                    py::cast(leaf_pos)
                ));
                return true;
            }
        );
    }
    return result;
}


std::tuple<py::array_t<int32_t>, py::array_t<int32_t>, py::array_t<double>>
setup_vertex_cover(
    py::list candidates,
    py::array_t<int32_t> active_leaves,
    py::tuple padding,
    int32_t leaf_size
) {
    int32_t n_candidates = candidates.size();

    if (n_candidates == 0) {
        return std::make_tuple(
            py::array_t<int32_t>(0),
            py::array_t<int32_t>(0),
            py::array_t<double>(0)
        );
    }

    py::tuple first_candidate = candidates[0];
    int32_t n_dims = py::len(first_candidate[0]);
    int32_t n_leaves = active_leaves.shape(0);

    auto leaf_map = build_leaf_map(active_leaves);
    std::vector<float> padding_vec = tuple_to_vector<float>(padding, n_dims);

    std::vector<int32_t> rows_vec;
    std::vector<int32_t> cols_vec;
    std::vector<double> objective_vec;
    objective_vec.reserve(n_candidates);

    for (int32_t box_idx = 0; box_idx < n_candidates; ++box_idx) {
        py::tuple candidate = candidates[box_idx];
        LeafBox box = to_leaf_coords(candidate, 1, n_dims, box_idx);

        double objective = 1.0;
        for (int32_t d = 0; d < n_dims; ++d) {
            objective *= (box.shape[d] + padding_vec[d]) * leaf_size;
        }
        objective = objective * log1p(objective);
        objective_vec.push_back(objective);

        auto covered_indices = leaf_assignment(box, leaf_map, n_dims, true);
        for (int32_t leaf_idx : covered_indices) {
            rows_vec.push_back(leaf_idx);
            cols_vec.push_back(box_idx);
        }
    }

    int32_t n_entries = rows_vec.size();

    auto rows = py::array_t<int32_t>(n_entries);
    auto cols = py::array_t<int32_t>(n_entries);
    auto objective = py::array_t<double>(n_candidates);

    std::memcpy(rows.mutable_data(), rows_vec.data(), n_entries * sizeof(int32_t));
    std::memcpy(cols.mutable_data(), cols_vec.data(), n_entries * sizeof(int32_t));
    std::memcpy(objective.mutable_data(), objective_vec.data(), n_candidates * sizeof(double));

    return std::make_tuple(rows, cols, objective);
}

py::list jitter_candidates(
    py::list candidates,
    py::array_t<int32_t> active_leaves,
    int32_t margin,
    int32_t stride
) {
    if (candidates.size() == 0) {
        return candidates;
    }

    int32_t n_leaves = active_leaves.shape(0);
    if (n_leaves == 0) {
        return py::list();
    }

    py::tuple first_candidate = candidates[0];
    int32_t n_dims = py::len(first_candidate[0]);

    auto leaf_map = build_leaf_map(active_leaves);

    // Make sure grid always includes 0
    std::vector<int32_t> grid;
    for (int32_t val = -margin; val <= margin; val += stride) {
        grid.push_back(val);
    }
    if (std::find(grid.begin(), grid.end(), 0) == grid.end()) {
        grid.push_back(0);
        std::sort(grid.begin(), grid.end());
    }

    // Use set to deduplicate jittered candidates
    std::unordered_set<std::pair<std::vector<int32_t>, std::vector<int32_t>>, CandidateHash>
        jittered_set;

    for (size_t i = 0; i < candidates.size(); ++i) {
        py::tuple candidate = candidates[i];
        std::vector<int32_t> shape = tuple_to_vector<int32_t>(candidate[0], n_dims);
        std::vector<int32_t> pos = tuple_to_vector<int32_t>(candidate[1], n_dims);

        std::vector<int32_t> grid_start(n_dims, 0);
        std::vector<int32_t> grid_stop(n_dims, grid.size());
        std::vector<int32_t> grid_step(n_dims, 1);
        iterate_ranges(grid_start, grid_stop, grid_step, n_dims,
            [&](const std::vector<int32_t>& grid_indices) {
                std::vector<int32_t> shifted_pos(n_dims);
                bool valid = true;

                for (int32_t d = 0; d < n_dims; ++d) {
                    int32_t delta = grid[grid_indices[d]];
                    shifted_pos[d] = pos[d] + delta;
                    if (shifted_pos[d] < 0) {
                        valid = false;
                        break;
                    }
                }

                // Only keep candidate if it covers an active leaf
                if (valid) {
                    LeafBox box;
                    box.shape = shape;
                    box.pos = shifted_pos;

                    auto covered_indices = leaf_assignment(box, leaf_map, n_dims, false);
                    if (covered_indices.size() > 0) {
                        jittered_set.insert(std::make_pair(shape, shifted_pos));
                    }
                }
                return true;
            }
        );
    }

    py::list result;
    for (const auto& candidate : jittered_set) {
        result.append(py::make_tuple(
            py::cast(candidate.first),
            py::cast(candidate.second)
        ));
    }

    return result;
}


bool is_valid_decomposition(
    const std::vector<int32_t>& smaller_box,
    const std::vector<int32_t>& box_shape,
    int32_t leaf_size,
    int32_t n_dims
) {
    // Cant decompose into the same size
    for (int32_t d = 0; d < n_dims; ++d) {
        if (smaller_box[d] != box_shape[d]) {
            goto not_same_size;
        }
    }
    return false;

not_same_size:
    // Smaller box must fit inside the larger box
    for (int32_t d = 0; d < n_dims; ++d) {
        if (smaller_box[d] > box_shape[d]) {
            return false;
        }
    }

    // Must have non-zero size in leaf coordinates
    for (int32_t d = 0; d < n_dims; ++d) {
        if (smaller_box[d] / leaf_size == 0) {
            return false;
        }
    }

    return true;
}

std::vector<std::pair<std::vector<int32_t>, std::vector<int32_t>>>
compute_decomposition_patterns(
    const LeafBox& box,
    const std::vector<std::vector<int32_t>>& valid_sizes,
    int32_t leaf_size,
    int32_t n_dims,
    bool nearest = false,
    bool non_overlapping = false
) {
    std::vector<std::pair<std::vector<int32_t>, std::vector<int32_t>>> patterns;
    patterns.push_back(std::make_pair(std::vector<int32_t>(n_dims, 0), box.shape));

    // Calculate current box volume for comparison
    std::vector<int32_t> original_box_shape(n_dims);
    int64_t current_volume = 1;
    for (int32_t d = 0; d < n_dims; ++d) {
        original_box_shape[d] = box.shape[d] * leaf_size;
        current_volume *= original_box_shape[d];
    }

    size_t target_size_idx = valid_sizes.size();
    if (nearest) {
        int64_t best_volume = -1;
        for (size_t i = 0; i < valid_sizes.size(); ++i) {
            int64_t vol = 1;
            for (int32_t d = 0; d < n_dims; ++d) {
                vol *= valid_sizes[i][d];
            }
            if (vol < current_volume && vol > best_volume) {
                best_volume = vol;
                target_size_idx = i;
            }
        }
        if (target_size_idx == valid_sizes.size()) {
            return patterns;
        }
    }

    size_t start_idx = nearest ? target_size_idx : 0;
    size_t end_idx = nearest ? target_size_idx + 1 : valid_sizes.size();

    for (size_t i = start_idx; i < end_idx; ++i) {
        const auto& smaller_box = valid_sizes[i];

        if (!is_valid_decomposition(smaller_box, original_box_shape, leaf_size, n_dims)) {
            continue;
        }

        std::vector<int32_t> start(n_dims, 0);
        std::vector<int32_t> max_offset(n_dims);
        std::vector<int32_t> leaf_smaller_box(n_dims);

        // Non overlapping tiling grid
        std::vector<int32_t> step(n_dims, 1);

        for (int32_t d = 0; d < n_dims; ++d) {
            leaf_smaller_box[d] = smaller_box[d] / leaf_size;
            max_offset[d] = box.shape[d] - leaf_smaller_box[d] + 1;
            if (non_overlapping){
                step[d] = leaf_smaller_box[d];
            }

        }

        iterate_ranges(start, max_offset, step, n_dims, [&](const std::vector<int32_t>& offset) {
            patterns.push_back(std::make_pair(offset, leaf_smaller_box));
            return true;
        });
    }

    return patterns;
}


py::list decompose_boxes(
    py::list current_boxes,
    py::tuple valid_box_sizes,
    int32_t leaf_size,
    bool nearest,
    bool overlapping
) {
    if (current_boxes.size() == 0) {
        return py::list();
    }

    py::tuple first_box = current_boxes[0];
    int32_t n_dims = py::len(first_box[0]);

    std::vector<std::vector<int32_t>> valid_sizes;
    valid_sizes.reserve(valid_box_sizes.size());
    for (size_t i = 0; i < valid_box_sizes.size(); ++i) {
        valid_sizes.push_back(tuple_to_vector<int32_t>(valid_box_sizes[i], n_dims));
    }

    std::unordered_map<std::vector<int32_t>, std::vector<std::pair<std::vector<int32_t>, std::vector<int32_t>>>>
        shape_to_patterns;

    std::vector<LeafBox> leaf_boxes;
    leaf_boxes.reserve(current_boxes.size());

    for (size_t i = 0; i < current_boxes.size(); ++i) {
        py::tuple box = current_boxes[i];
        LeafBox leaf_box = to_leaf_coords(box, leaf_size, n_dims);
        leaf_boxes.push_back(leaf_box);

        if (shape_to_patterns.find(leaf_box.shape) == shape_to_patterns.end()) {
            shape_to_patterns[leaf_box.shape] = compute_decomposition_patterns(
                leaf_box, valid_sizes, leaf_size, n_dims, nearest, !overlapping
            );
        }
    }

    // Generate all unique candidates from decomposition patterns
    std::unordered_set<std::pair<std::vector<int32_t>, std::vector<int32_t>>, CandidateHash>
        candidates;

    for (size_t box_idx = 0; box_idx < leaf_boxes.size(); ++box_idx) {
        const LeafBox& leaf_box = leaf_boxes[box_idx];
        const auto& patterns = shape_to_patterns.at(leaf_box.shape);

        for (size_t p = 0; p < patterns.size(); ++p) {
            const std::vector<int32_t>& offset = patterns[p].first;
            const std::vector<int32_t>& shape = patterns[p].second;

            std::vector<int32_t> final_pos(n_dims);
            bool valid_position = true;

            for (int32_t d = 0; d < n_dims; ++d) {
                final_pos[d] = leaf_box.pos[d] + offset[d];
                if (final_pos[d] < 0) {
                    valid_position = false;
                    break;
                }
            }

            if (valid_position) {
                candidates.insert(std::make_pair(shape, final_pos));
            }
        }
    }

    py::list result;
    for (const auto& candidate : candidates) {
        result.append(py::make_tuple(
            py::cast(candidate.first),
            py::cast(candidate.second)
        ));
    }
    return result;
}

void register_set_cover_bindings(py::module& m) {
    m.def("generate_candidates",
          [](py::tuple segmentation_shape, py::tuple valid_box_sizes, int32_t leaf_size,
            bool tile_candidates) {
              return generate_candidates(segmentation_shape, valid_box_sizes, leaf_size, tile_candidates);
          },
          "Generate candidate box placements on the octree grid",
          py::arg("segmentation_shape"),
          py::arg("valid_box_sizes"),
          py::arg("leaf_size"),
          py::arg("tile_candidates"));

    m.def("setup_vertex_cover",
          [](py::list candidates, py::array_t<int32_t> active_leaves, py::tuple padding,
            int32_t leaf_size) {
              return setup_vertex_cover(candidates, active_leaves, padding, leaf_size);
          },
          "Build coverage matrix and objective for vertex cover problem",
          py::arg("candidates"),
          py::arg("active_leaves"),
          py::arg("padding"),
          py::arg("leaf_size") = 1);

    m.def("jitter_candidates",
          [](py::list candidates, py::array_t<int32_t> active_leaves,
            int32_t margin, int32_t stride) {
              return jitter_candidates(candidates, active_leaves, margin, stride);
          },
          "Generate all possible translations of candidates avoiding duplication.",
          py::arg("candidates"),
          py::arg("active_leaves"),
          py::arg("margin") = 1,
          py::arg("stride") = 1);

    m.def("decompose_boxes",
          [](py::list current_boxes, py::tuple valid_box_sizes, int32_t leaf_size,
            bool nearest, bool overlapping) {
              return decompose_boxes(current_boxes, valid_box_sizes, leaf_size, nearest, overlapping);
          },
          "Decompose boxes into candidate positions",
          py::arg("current_boxes"),
          py::arg("valid_box_sizes"),
          py::arg("leaf_size"),
          py::arg("nearest") = false,
          py::arg("overlapping") = true);
}
