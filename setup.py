import subprocess
from glob import glob
from typing import List

import setuptools

from Cython.Build import cythonize
from setuptools import Extension
import numpy

__version__ = "0.0.1"


def get_gdal_config(option: str) -> str:
    gdal_config_cmd = ["gdal-config", f"--{option}"]
    result = subprocess.run(gdal_config_cmd, capture_output=True, text=True)
    return result.stdout.strip()


def get_arg(s: str, prefix: str) -> List[str]:
    return [x.replace(prefix, "") for x in s.split() if x.startswith(prefix)]


gdal_include_dir: List[str] = get_arg(get_gdal_config("cflags"), "-I")
gdal_library_dir: List[str] = get_arg(get_gdal_config("libs"), "-L")
gdal_deplibs_dir: List[str] = get_arg(get_gdal_config("dep-libs"), "-L")
gdal_libs: List[str] = get_arg(get_gdal_config("libs"), "-l")
gdal_deplibs: List[str] = get_arg(get_gdal_config("dep-libs"), "-l")

# NOTE:
# ImportError: dynamic module does not define module export function (PyInit__funmixer_native)
# Means that extension.cpp probably has a mismatched name at PYBIND11_MODULE

cyth_ext = cythonize(
    Extension(
        "funmixer.flow_acc_cfuncs",
        ["funmixer/flow_acc_cfuncs.pyx"],
        extra_compile_args=["--std=c++20"],
    )
)[0]

cpp_ext = Extension(
    "_funmixer_native",
    [
        "funmixer/native/extension.cpp",
        "funmixer/native/faster-unmixer.cpp",
    ]
    + glob("submodules/richdem/src/*.cpp"),
    include_dirs=[
        "funmixer/native/",
        "submodules/richdem/include",
    ]
    + gdal_include_dir,
    library_dirs=gdal_library_dir + gdal_deplibs_dir,
    libraries=gdal_libs + gdal_deplibs,
    define_macros=[
        ("USEGDAL", None),
    ],
    extra_compile_args=["--std=c++20"],
)

ext_modules = [cyth_ext, cpp_ext]


# TODO: https://packaging.python.org/tutorials/distributing-packages/#configuring-your-project
setuptools.setup(
    name="funmixer",
    version=__version__,
    description="Convex unmixing of fluid networks",
    url="https://github.com/r-barnes/faster-unmixer",
    author="Richard Barnes",
    author_email="rijard.barnes@gmail.com",
    license="GPLv3",
    packages=setuptools.find_packages(),
    # pyre-fixme[6]: For 9th argument expected `List[Extension]` but got
    #  `List[Pybind11Extension]`.
    ext_modules=ext_modules,
    keywords="GIS hydrology raster networks",
    python_requires=">= 3.8, <4",
    # TODO: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Other Audience",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    include_dirs=[numpy.get_include()],
    install_requires=[
        "cvxpy",
        "cython",
        # "gdal",
        "hypothesis",
        "imageio",
        "matplotlib",
        "networkx",
        "numpy",
        "pandas",
        "pybind11",
        "pytest",
        "pygraphviz",
        "tqdm",
    ],
)
