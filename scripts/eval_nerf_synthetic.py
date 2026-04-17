import argparse

import torch
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import trimesh
import json

# https://github.com/otaheri/chamfer_distance
from chamfer_distance import ChamferDistance


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) > 0
        mesh = trimesh.util.concatenate(tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in scene_or_mesh.geometry.values()))
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def sample_mesh(m, n):
    vpos, _ = trimesh.sample.sample_surface(m, n)
    return torch.tensor(vpos, dtype=torch.float32, device="cuda")


def eval_nerf_synthetic(mesh_path, ref_path, n_sample, output_dir=None):
    chamfer_dist = ChamferDistance()

    mesh = as_mesh(trimesh.load(mesh_path))
    ref = as_mesh(trimesh.load(ref_path))

    R = Rot.from_euler("x", 90, degrees=True)
    ref.vertices = R.apply(ref.vertices)

    # Make sure l=1.0 maps to 1/10th of the AABB. https://arxiv.org/pdf/1612.00603.pdf
    scale = 10.0 / np.amax(np.amax(ref.vertices, axis=0) - np.amin(ref.vertices, axis=0))
    ref.vertices = ref.vertices * scale
    mesh.vertices = mesh.vertices * scale

    # Sample mesh surfaces
    vpos_mesh = sample_mesh(mesh, n_sample)
    vpos_ref = sample_mesh(ref, n_sample)

    d2s, s2d, _, _ = chamfer_dist(vpos_mesh[None, ...], vpos_ref[None, ...])
    mean_d2s = torch.mean(d2s).item()
    mean_s2d = torch.mean(s2d).item()
    result = {
        "mean_d2s": mean_d2s,
        "mean_s2d": mean_s2d,
        "overall_mean": (mean_d2s + mean_s2d) / 2,
        "overall_sum": mean_d2s + mean_s2d,
        "n_data": vpos_mesh.shape[0],
        "n_gt": vpos_ref.shape[0],
    }
    if output_dir is not None:
        with open(f"{output_dir}/results.json", "w") as fp:
            json.dump(result, fp, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chamfer loss")
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--ref_path", type=str, required=True)
    parser.add_argument("-n", type=int, default=2_500_000)
    FLAGS = parser.parse_args()

    result = eval_nerf_synthetic(FLAGS.mesh_path, FLAGS.ref_path, FLAGS.n)
    print(result)
