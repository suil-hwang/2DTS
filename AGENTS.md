# Repository Guidelines

This checkout focuses on 2D triangle splatting (2DTS).

## Project Structure & Module Organization

- `src/diff_recon/` contains models, trainers, datasets, rendering, and utilities. `TSModel(nn.Module)` owns triangle parameters and optimizer state; `TSTrainer` coordinates training.
- `config/` holds dataset-specific YAML presets; `scripts/` contains evaluation and COLMAP helpers; `assets/` stores documentation images.
- `submodules/simple-knn/` and `submodules/diff-triangle-rasterization/` contain vendored CUDA extension sources, tracked directly by this repository. The optional `submodules/fused-ssim/` is a git submodule pinned to an unmodified upstream commit; adapt calling code instead of patching it.
- `tests/` holds unittest regressions for the trainers and extension adapters.
- `py_viewer/` reserves the future ModernGL viewer. `data/`, `outputs/`, `docs/`, `paper/`, and temporary artifacts are Git-ignored.

## Build, Test, and Development Commands

Use Python 3.12 and CUDA Toolkit 13.0. On Windows, initialize an x64 VS2022 developer shell and set `CUDA_HOME` to the toolkit directory.

```sh
git submodule update --init --recursive
conda env create -f environment.yml
conda activate 2DTS
python -m pip install --no-build-isolation --no-deps ./submodules/simple-knn ./submodules/diff-triangle-rasterization ./submodules/fused-ssim
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python -m unittest discover -s tests -t . -v
```

These commands fetch the submodules, create the environment, compile native dependencies, install the editable package, check dependency consistency, and run the regressions. `fused-ssim` is optional: without it SSIM falls back to PyTorch. Rebuild extensions after C++/CUDA changes. On Windows with torch 2.11 and MSVC 14.44, set `NVCC_APPEND_FLAGS=-Xcompiler /permissive- -DWIN32_LEAN_AND_MEAN` before building, or nvcc fails with `C2872: 'std': ambiguous symbol`.

Training entrypoint: `python run_experiments.py --type NerfSynthetic --dataset_path ./data/nerf_synthetic --num_workers 0`. NerfSynthetic Chamfer evaluation uses libigl on the CPU. The legacy `viser_viewer.py` requires excluded Viser. Inspect logs with `tensorboard --logdir outputs`.

## Coding Style & Naming Conventions

Use four-space indentation, descriptive snake_case functions/variables, PascalCase classes, and existing type annotations. Preserve established module names such as `TS_model.py` and existing public methods. No formatter or linter is configured; match adjacent code and avoid unrelated formatting.

Preserve parameter keys, Adam group ordering, parameter identity during pruning/growth, and rendering contracts when refactoring.

## Commit & Pull Request Guidelines

History uses short descriptive subjects; Conventional Commits are not required. Prefer imperative summaries and focused commits.

PRs should explain scope, relevant issues, runtime/configuration assumptions, and validation commands/results. Include before/after images or metrics for rendering changes. Preserve unrelated edits and running GUI/GPU processes.
