from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "tsp_core",
        sources=["src/tsp_core.pyx", "src/tsp_utils.cpp"],
        include_dirs=["include"],
        language="c++",
        extra_compile_args=["-O3"]
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level="3",
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    )
)
