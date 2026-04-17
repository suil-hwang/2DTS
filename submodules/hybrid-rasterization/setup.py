from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from os.path import abspath, dirname, join

glm_path = join(dirname(dirname(abspath(__file__))), "glm")
pkg_name = "hybrid_rasterization"

setup(
    name=pkg_name,
    version="1.0.0",
    packages=[pkg_name],
    ext_modules=[
        CUDAExtension(
            name=f"{pkg_name}._C",
            sources=[
                "src/extension_interface.cu",
                "src/forward.cu",
                "src/rasterizer.cu",
                "src/backward.cu",
                "ext.cpp",
            ],
            extra_compile_args={"nvcc": ["-I" + glm_path]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
