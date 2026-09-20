<div align="center">

# 2D Triangle Splatting for Direct Differentiable Mesh Training

[Arxiv][1] | [Project Page][4]

Kaifeng Sheng*, Zheng Zhou*, Yingliang Peng, Qianwei Wang (*Equal Contribution)

Amap, Alibaba Group

</div>

## - Project Overview

Official implementation of [2DTS][1] (2D Triangle Splatting for Direct Differentiable Mesh Training)

We provide a complete training pipeline for 2DTS, a differentiable 3D Geometric Representation adapted from [3DGS][2] (3D Gaussian Splatting) that replace the Gaussians primitives with triangle primitives, while retaining the full differentiability of the model.
The proposed method is capable of producing triangle meshes with high visual fidelity through an end-to-end training pipeline.

This checkout contains the 2DTS pipeline only. Standalone 3DGS and hybrid Gaussian/triangle rendering have been removed. The two native dependencies are `simple-knn` and `diff-triangle-rasterization`; point-cloud initialization accepts RGB or SH DC colors from PLY files. Triangle mesh and GLB/glTF utilities remain available.

![demo_image](./assets/demo_image.png)

Our method can be applied to large-scale datasets, such as MatrixCity, which contains 6000+ images. Such datasets are challenging for existing mesh reconstruction methods, but our method can handle them efficiently.
The reconstructed meshes can be directly used in modern game engines, such as Blender, for relighting, shadow rendering, and other advanced rendering effects. See the following image for an example of relighting effect on a reconstructed mesh from MatrixCity dataset:

![relighting_image](./assets/relighting_image.png)

## - Abstract

Differentiable rendering with 3D Gaussian primitives has emerged as a powerful method for reconstructing high-fidelity 3D scenes from multi-view images.
While it offers improvements over NeRF-based methods, this representation still encounters challenges with rendering speed and advanced rendering effects, such as relighting and shadow rendering, compared to mesh-based models.
In this paper, we propose 2D Triangle Splatting (2DTS), a novel method that replaces 3D Gaussian primitives with 2D triangle facelets.
This representation naturally forms a discrete mesh-like structure while retaining the benefits of continuous volumetric modeling.
By incorporating a compactness parameter into the triangle primitives, we enable direct training of photorealistic meshes.
Our experimental results demonstrate that our triangle-based method, in its vanilla version (without compactness tuning), achieves higher fidelity compared to state-of-the-art Gaussian-based methods.
Furthermore, our approach produces reconstructed meshes with superior visual quality compared to existing mesh reconstruction methods.

## - Installation

For this checkout's Python 3.12 / CUDA 13.0 environment, use [`environment.yml`](./environment.yml) and the steps below.

1. Prepare CUDA Toolkit 13.0 and the host C++ compiler, and set `CUDA_HOME` to the toolkit path. On Windows, use an x64 Visual Studio developer shell.
2. From this checkout's root, create the environment: `conda env create -f environment.yml`.
3. Activate it: `conda activate 2DTS`.
4. Build the native dependencies: `python -m pip install --no-build-isolation --no-deps ./submodules/simple-knn ./submodules/diff-triangle-rasterization`.
5. Install the project: `python -m pip install --no-build-isolation --no-deps -e .`.

### Install with AI

If you use an AI coding agent in your editor or terminal, you can ask it to install this repository for you. Make sure CUDA Toolkit 13.0 is already installed and that `CUDA_HOME` is set correctly.

From the project root, give the agent a prompt like this:

```text
Install this 2DTS repository for local development using environment.yml. Activate 2DTS, build simple-knn and diff-triangle-rasterization with --no-build-isolation --no-deps, then install the root package with the same flags and -e . Verify the CUDA toolkit, host compiler, extension imports, and a small CUDA forward/backward run.
```


## - Usage
### Training
Execute `run_experiments.py` to train 2DTS models on one of Mip-NeRF 360, NerfSynthetic, DTU, Tanks and Blending, Tanks and Temples, or MatrixCity datasets by running the following command:
```bash
python run_experiments.py --type {experiment_type} --dataset_path /path/to/dataset --num_workers 0
```
`experiment_type` can be one of `MipNerf360`, `NerfSynthetic`, `DTU`, `TanksAndBlending`, `TanksAndTemples`, or `MatrixCity`.

The script requires the dataset to be downloaded beforehand, and the dataset path should point to the root directory of the dataset.
For example, if you want to train on the NerfSynthetic dataset, and have the dataset stored in `./data/nerf_synthetic`, you can run the following command:
```bash
python run_experiments.py --type NerfSynthetic --dataset_path ./data/nerf_synthetic --num_workers 0
```

### Logs
Training logs will be saved in the `./outputs` directory. You can use TensorBoard to visualize the training process:
```bash
tensorboard --logdir ./outputs
```

### Rendering
We provide an interactive web viewer based on [Viser Viewer][3] for visualizing the trained triangle splats and meshes.
The legacy viewer requires Viser, which is excluded from `environment.yml`. `py_viewer/` is reserved for the planned ModernGL viewer and does not yet contain an implementation.
You can run the viewer by executing the following command:
```bash
python viser_viewer.py --config /path/to/config --dataset /path/to/dataset --scene {scene_name}
```
For example, if you ran the `NerfSynthetic` experiment and want to visualize the `ship` scene, and have the dataset stored in `./data/nerf_synthetic`, you can run the following command:
```bash
python viser_viewer.py --config config/NerfSynthetic_VanillaTS_mesh.yaml --dataset ./data/nerf_synthetic --scene ship
```

Then, open your web browser and navigate to `http://localhost:8080` to view the rendered scene. If you are running the viewer on a remote server, make sure to set up port forwarding or access the server's IP address directly.

## - Notes
We provided two distinct training configurations: VanillaTS and VanillaTS_mesh.
- VanillaTS is a close mimick of the original 3DGS method, with compactness parameter set to 1.0 and generating transparent and diffuse triangle splats (See [2DTS][1] for details).
- VanillaTS_mesh will produce a solid triangle mesh at the end of training through a compactness annealing process. The triangle mesh is saved in the `.ply` and `.glb` formats. Note that when **back_culling** is **disabled** for the training process, **the mesh file will contain each triangle <span style="color:red">twice</span>**, once for the front face and once for the back face.

The difference between a diffuse and a solid triangle is visualized in the following image:

![triangle_splatting](./assets/triangle_splatting.png) 

## - License

This repository contains code under **two different licenses**:

- 🟥 **Gaussian Splatting Research License** — applies to components derived from the original [Gaussian Splatting][2] project:
  - `submodules/simple-knn/`
  - These components are licensed for **non-commercial research use only**.
  - See [LICENSE.gausplat.md](./LICENSE.gausplat.md)

- 🟩 **MIT License** — applies to other parts of the repository, including:
  - `src/diff_recon/`
  - `submodules/diff-triangle-rasterization/`
  - See [LICENSE](./LICENSE)

Please make sure to comply with both licenses when using this repository.

## - Citation

If you find our work useful, please consider citing our paper:
```bibtex
@misc{sheng20252dtrianglesplattingdirect,
      title={2D Triangle Splatting for Direct Differentiable Mesh Training}, 
      author={Kaifeng Sheng and Zheng Zhou and Yingliang Peng and Qianwei Wang},
      year={2025},
      eprint={2506.18575},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.18575}, 
}
```

<!-- Reference -->
[1]: https://arxiv.org/abs/2506.18575
[2]: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
[3]: https://github.com/nerfstudio-project/viser
[4]: https://gaoderender.github.io/triangle-splatting/
