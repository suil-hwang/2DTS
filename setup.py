from setuptools import setup, find_packages
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

install_requires = [
    f"custom-gaussian-rasterization @ file://localhost/{root_dir}/submodules/custom-gaussian-rasterization",
    f"simple-knn @ file://localhost/{root_dir}/submodules/simple-knn",
    f"diff-triangle-rasterization @ file://localhost/{root_dir}/submodules/diff-triangle-rasterization",
    f"hybrid-rasterization @ file://localhost/{root_dir}/submodules/hybrid-rasterization",
]
# install_requires += [x.strip() for x in open("requirements.txt").read().splitlines() if x.strip() and not x.strip().startswith("#")]
 
setup(
    name="diff_recon",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    author="Kaifeng Sheng",
    author_email="kaifeng.skf@gmail.com",
    description="Differentiable Reconstruction",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # python_requires=">=3.10",
)
