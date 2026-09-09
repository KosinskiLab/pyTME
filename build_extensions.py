"""Build script for pybind11 extensions."""

import platform
from os import listdir
from os.path import join

from pybind11.setup_helpers import Pybind11Extension, build_ext


def build(setup_kwargs):
    """Build pybind11 extensions."""
    cpp_args = [
        "-O3",
        "-std=c++17",
        "-funroll-loops",
        "-ffast-math",
        "-flto",
        "-ftree-vectorize",
    ]

    if platform.processor() not in ("arm", "i386"):
        cpp_args.extend(["-march=native", "-fno-math-errno", "-fno-trapping-math"])

    cpp_dir = "tme/external"
    ext_modules = [
        Pybind11Extension(
            "tme.extensions",
            [join(cpp_dir, x) for x in listdir(cpp_dir) if x.endswith(".cpp")],
            extra_compile_args=cpp_args,
        ),
    ]

    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": build_ext},
        }
    )
