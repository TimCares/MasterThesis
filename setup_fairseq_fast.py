from setuptools import Extension, setup
import sys


if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]

class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy

        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


extensions = [
    Extension(
        "src.fairseq.libbleu",
        sources=[
            "src/fairseq/clib/libbleu/libbleu.cpp",
            "src/fairseq/clib/libbleu/module.cpp",
        ],
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "src.fairseq.data.data_utils_fast",
        sources=["src/fairseq/data/data_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "src.fairseq.data.token_block_utils_fast",
        sources=["src/fairseq/data/token_block_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage: setup_fairseq_fast.py build_ext --inplace")
        sys.exit(1)

    setup(ext_modules=extensions)

    print()
    print()
    print("Setup successful, build Cython components.")