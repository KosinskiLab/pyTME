import os
import platform

from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

about = {}
with open(os.path.join("tme", "__version__.py"), mode="r") as f:
    exec(f.read(), about)
__version__ = about["__version__"]

if __name__ == "__main__":
    if platform.processor() in ("arm", "i386"):
        c_args = ["-O3", "-ftree-vectorize", "-std=c17"]
        cpp_args = [
            "-O3",
            "-ftree-vectorize",
            "-std=c++11",
            "-funroll-loops",
            "-ffast-math",
        ]
    else:
        c_args = ["-O3", "-march=native", "-std=c17"]
        cpp_args = [
            "-O3",
            "-march=native",
            "-std=c++11",
            "-funroll-loops",
            "-ffast-math",
        ]

    ext_modules = [
        Pybind11Extension(
            "tme.extensions",
            ["src/extensions.cpp"],
            define_macros=[("VERSION_INFO", __version__)],
            extra_compile_args=cpp_args,
        ),
    ]

    package_data = {
        "tme": [
            os.path.join("data", "*.npy"),
            os.path.join("data", "metadata.yaml"),
            os.path.join("data", "scattering_factors.pickle"),
        ]
    }
    setup(
        name="pytme",
        author="Valentin Maurer",
        author_email="valentin.maurer@embl-hamburg.de",
        version=__version__,
        description="Template matching on electron microscopy data.",
        package_data=package_data,
        packages=find_packages(),
        scripts=[
            "scripts/match_template.py",
            "scripts/estimate_ram_usage.py",
            "scripts/preprocessor_gui.py",
            "scripts/preprocess.py",
            "scripts/postprocess.py",
        ],
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
    )
