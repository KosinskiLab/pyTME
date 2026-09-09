/*  Pybind extensions for pytme.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
*/

#include <pybind11/pybind11.h>

#include "analyzer_funcs.h"

namespace py = pybind11;

PYBIND11_MODULE(extensions, m) {
    m.doc() = "pytme extensions";

    register_analyzer_bindings(m);

}
