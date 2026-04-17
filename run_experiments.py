import os
import argparse
import torch

from src.diff_recon import VanillaTSTrainer, loadConfig, run_exp_with_args
from scripts.eval_dtu import cull_and_eval
from scripts.eval_nerf_synthetic import eval_nerf_synthetic


def exp(
    config_path: str,
    dataset_path: str,
    scene_id: str,
    resolution: int,
    device: int,
    point_count_dense: int = None,
    point_count_prune: int = None,
    eval_cd: bool = False,
):
    config = loadConfig(config_path)
    config.dataset.local_dir = dataset_path
    config.dataset.scene_id = scene_id
    config.dataset.train_target_res = resolution
    config.dataset.test_target_res = resolution
    if point_count_dense is not None and config.model.model_update.densification is not None:
        config.model.model_update.densification.target_point_num = point_count_dense
    if point_count_prune is not None and config.model.model_update.contribution_pruning is not None:
        config.model.model_update.contribution_pruning.target_point_num = point_count_prune

    trainer = VanillaTSTrainer(config, exp_name=scene_id, device=device)
    trainer.train()

    if eval_cd:
        if config.dataset.type == "NerfSynthetic":
            mesh_path = os.path.join(trainer.output_dir, "mesh_ply", f"{config.trainer.save_mesh_iterations[-1]}_mesh.ply")
            output_dir = os.path.join(trainer.output_dir, "eval_cd")
            ref_path = os.path.join(dataset_path, scene_id, f"{scene_id}.obj")
            n_sample = 2_500_000
            cd_result = eval_nerf_synthetic(mesh_path, ref_path, n_sample, output_dir)
            trainer.logger.info(f"Chamfer distance eval result: \n{cd_result}")
        else:
            scan_id = int(scene_id.replace("scan", ""))
            output_dir = os.path.join(trainer.output_dir, "eval_cd")
            ply_file = os.path.join(trainer.output_dir, "mesh_ply", f"{config.trainer.save_pcd_iterations[-1]}_pcd.ply")
            dtu_dir = os.path.dirname(dataset_path)
            cd_result = cull_and_eval(scan_id, output_dir, ply_file, dtu_dir, "pcd")
            trainer.logger.info(f"Chamfer distance eval result: \n{cd_result}")


def train_MipNerf360_VanillaTS(dataset_path: str, num_workers: int, scene: str = None):
    config_path_splat = "config/MipNerf360_VanillaTS.yaml"
    device_count = min(torch.cuda.device_count(), num_workers) if num_workers > 0 else 1

    scenes = ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"]
    resolutions = [4, 4, 4, 4, 4, 2, 2, 2, 2]
    point_counts_dense = [3_000_000, 3_000_000, 3_000_000, 3_000_000, 3_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000]

    if scene is not None:
        if scene not in scenes:
            raise ValueError(f"Scene {scene} not in predefined scenes: {scenes}")
        resolutions = [resolutions[scenes.index(scene)]]
        point_counts_dense = [point_counts_dense[scenes.index(scene)]]
        scenes = [scene]

    # Optimize triangle splats
    args_list = []
    for i in range(len(scenes)):
        device = i % device_count
        args_list.append((config_path_splat, dataset_path, scenes[i], resolutions[i], device, point_counts_dense[i]))
    run_exp_with_args(exp, args_list, num_workers=num_workers)


def train_TanksAndBlending_VanillaTS(dataset_path: str, num_workers: int, scene: str = None):
    config_path_splat = "config/TanksAndBlending_VanillaTS.yaml"
    device_count = min(torch.cuda.device_count(), num_workers) if num_workers > 0 else 1

    scenes = ["tandt/truck", "tandt/train", "db/drjohnson", "db/playroom"]
    point_counts_dense = [2_000_000, 1_000_000, 3_000_000, 1_800_000]

    if scene is not None:
        if scene not in scenes:
            raise ValueError(f"Scene {scene} not in predefined scenes: {scenes}")
        point_counts_dense = [point_counts_dense[scenes.index(scene)]]
        scenes = [scene]

    # Optimize triangle splats
    args_list = []
    for i in range(len(scenes)):
        device = i % device_count
        args_list.append((config_path_splat, dataset_path, scenes[i], 1, device, point_counts_dense[i]))
    run_exp_with_args(exp, args_list, num_workers=num_workers)


def train_TanksAndTemples_VanillaTS(dataset_path: str, num_workers: int, scene: str = None):
    config_path_splat = "config/TanksAndTemples_VanillaTS.yaml"
    device_count = min(torch.cuda.device_count(), num_workers) if num_workers > 0 else 1

    scenes = ["Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom", "Truck"]
    point_counts_dense = [2_000_000, 2_000_000, 2_000_000, 2_000_000, 2_000_000, 2_000_000]

    if scene is not None:
        if scene not in scenes:
            raise ValueError(f"Scene {scene} not in predefined scenes: {scenes}")
        point_counts_dense = [point_counts_dense[scenes.index(scene)]]
        scenes = [scene]

    # Optimize triangle splats
    args_list = []
    for i in range(len(scenes)):
        device = i % device_count
        args_list.append((config_path_splat, dataset_path, scenes[i], 1, device, point_counts_dense[i]))
    run_exp_with_args(exp, args_list, num_workers=num_workers)


def train_NerfSynthetic_VanillaTS(dataset_path: str, num_workers: int, scene: str = None):
    config_path_splat = "config/NerfSynthetic_VanillaTS.yaml"
    config_path_mesh = "config/NerfSynthetic_VanillaTS_mesh.yaml"
    device_count = min(torch.cuda.device_count(), num_workers) if num_workers > 0 else 1

    scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
    point_counts_dense = [300_000, 300_000, 178_000, 150_000, 300_000, 238_000, 275_000, 300_000]
    point_counts_prune = [89_000, 82_000, 41_000, 58_000, 112_000, 78_000, 83_000, 93_000]

    if scene is not None:
        if scene not in scenes:
            raise ValueError(f"Scene {scene} not in predefined scenes: {scenes}")
        point_counts_dense = [point_counts_dense[scenes.index(scene)]]
        point_counts_prune = [point_counts_prune[scenes.index(scene)]]
        scenes = [scene]

    # Optimize triangle splats
    args_list = []
    for i in range(len(scenes)):
        device = i % device_count
        args_list.append((config_path_splat, dataset_path, scenes[i], 1, device, point_counts_dense[i], None))
    run_exp_with_args(exp, args_list, num_workers=num_workers)

    # Optimize mesh
    args_list = []
    for i in range(len(scenes)):
        device = i % device_count
        args_list.append((config_path_mesh, dataset_path, scenes[i], 1, device, None, point_counts_prune[i], True))
    run_exp_with_args(exp, args_list, num_workers=num_workers)


def train_DTU_VanillaTS(dataset_path: str, num_workers: int, scene: str = None):
    config_path_mesh = "config/DTU_VanillaTS_mesh.yaml"
    device_count = min(torch.cuda.device_count(), num_workers) if num_workers > 0 else 1

    scan_ids = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
    scenes = [f"scan{scan_id}" for scan_id in scan_ids]
    # point_counts_dense = [i * 1000 for i in [709, 928, 1187, 702, 110, 140, 151, 52, 331, 192, 329, 118, 358, 141, 140]]
    point_counts_dense = [1_000_000 for _ in scan_ids]

    if scene is not None:
        if scene not in scenes:
            raise ValueError(f"Scene {scene} not in predefined scenes: {scenes}")
        scan_ids = [scan_ids[scenes.index(scene)]]
        point_counts_dense = [point_counts_dense[scenes.index(scene)]]
        scenes = [scene]

    # Optimize mesh
    args_list = []
    for i in range(len(scenes)):
        device = i % device_count
        args_list.append((config_path_mesh, dataset_path, scenes[i], 1, device, point_counts_dense[i], None, True))
    run_exp_with_args(exp, args_list, num_workers=num_workers)


def train_MatrixCity_VanillaTS(dataset_path: str, num_workers: int, scene: str = None):
    config_path_mesh = "config/MatrixCity_VanillaTS_mesh.yaml"
    device_count = min(torch.cuda.device_count(), num_workers) if num_workers > 0 else 1

    scenes = ["small_city/aerial"]
    point_counts_dense = [6_000_000]

    if scene is not None:
        if scene not in scenes:
            raise ValueError(f"Scene {scene} not in predefined scenes: {scenes}")
        point_counts_dense = [point_counts_dense[scenes.index(scene)]]
        scenes = [scene]

    # Optimize mesh
    args_list = []
    for i in range(len(scenes)):
        device = i % device_count
        args_list.append((config_path_mesh, dataset_path, scenes[i], 1, device, point_counts_dense[i]))
    run_exp_with_args(exp, args_list, num_workers=num_workers)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, help="One of MipNerf360, TanksAndBlending, TanksAndTemples, NerfSynthetic, DTU, MatrixCity")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--num_workers", type=int, default=0, help="Number of concurrent training jobs")
    parser.add_argument("--scene", type=str, default=None, help="If set, only run this scene")
    args = parser.parse_args()
    dataset_path = args.dataset_path
    num_workers = args.num_workers
    scene = args.scene

    match args.type:
        case "MipNerf360":
            train_MipNerf360_VanillaTS(dataset_path, num_workers, scene)
        case "TanksAndBlending":
            train_TanksAndBlending_VanillaTS(dataset_path, num_workers, scene)
        case "TanksAndTemples":
            train_TanksAndTemples_VanillaTS(dataset_path, num_workers, scene)
        case "NerfSynthetic":
            train_NerfSynthetic_VanillaTS(dataset_path, num_workers, scene)
        case "DTU":
            train_DTU_VanillaTS(dataset_path, num_workers, scene)
        case "MatrixCity":
            train_MatrixCity_VanillaTS(dataset_path, num_workers, scene)
        case _:
            raise ValueError(f"Unknown type: {args.type}")
