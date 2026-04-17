import os
import numpy as np
import trimesh
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import json
import torch
import torch.nn.functional as F
import glob
from skimage.morphology import binary_dilation, disk
import cv2

# import sys
# script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "eval_dtu")
# sys.path.append(script_dir)
# from scripts.eval_dtu.evaluate_single_scene import cull_scan


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[: n1 + 1, : n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def write_vis_pcd(file, points, colors):
    point_cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
    point_cloud.export(file)


def eval(mesh_file, scan, mode, dataset_dir, out_dir):
    mp.freeze_support()

    downsample_density = 0.2
    patch_size = 60
    max_dist = 20
    visualize_threshold = 10

    thresh = downsample_density
    if mode == "mesh":
        pbar = tqdm(total=9)
        pbar.set_description("read data mesh")
        data_mesh = trimesh.load(mesh_file)

        vertices = data_mesh.vertices
        triangles = data_mesh.faces
        tri_vert = vertices[triangles]

        pbar.update(1)
        pbar.set_description("sample pcd from mesh")
        v1 = tri_vert[:, 1] - tri_vert[:, 0]
        v2 = tri_vert[:, 2] - tri_vert[:, 0]
        l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
        area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
        non_zero_area = (area2 > 0)[:, 0]
        l1, l2, area2, v1, v2, tri_vert = [arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]]
        thr = thresh * np.sqrt(l1 * l2 / area2)
        n1 = np.floor(l1 / thr)
        n2 = np.floor(l2 / thr)

        with mp.Pool() as mp_pool:
            new_pts = mp_pool.map(
                sample_single_tri,
                ((n1[i, 0], n2[i, 0], v1[i : i + 1], v2[i : i + 1], tri_vert[i : i + 1, 0]) for i in range(len(n1))),
                chunksize=1024,
            )

        new_pts = np.concatenate(new_pts, axis=0)
        data_pcd = np.concatenate([vertices, new_pts], axis=0)

    elif mode == "pcd":
        pbar = tqdm(total=8)
        pbar.set_description("read data pcd")
        data_pcd_trimesh = trimesh.load(mesh_file)
        data_pcd = data_pcd_trimesh.vertices

    pbar.update(1)
    pbar.set_description("random shuffle pcd index")
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description("downsample pcd")
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm="kd_tree", n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description("masking data pcd")
    ground_plane = loadmat(f"{dataset_dir}/ObsMask/Plane{scan}.mat")["P"]
    obs_mask_file = loadmat(f"{dataset_dir}/ObsMask/ObsMask{scan}_10.mat")
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ["ObsMask", "BB", "Res"]]
    BB = BB.astype(np.float32)

    patch = patch_size
    inbound = ((data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)).sum(axis=-1) == 3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    # data_hom = np.concatenate([data_in_obs, np.ones_like(data_in_obs[:, :1])], -1)
    # data_above = (ground_plane.reshape((1, 4)) * data_hom).sum(-1) > 0
    data_above = np.ones_like(data_in_obs[:, 0], dtype=bool)  # temporarily disable ground plane filtering
    data_in_obs = data_in_obs[data_above]

    pbar.update(1)
    pbar.set_description("read STL pcd")
    stl_pcd = trimesh.load(f"{dataset_dir}/Points/stl/stl{scan:03}_total.ply")
    stl = stl_pcd.vertices

    pbar.update(1)
    pbar.set_description("compute data2stl")
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description("compute stl2data")

    stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
    above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    pbar.update(1)
    pbar.set_description("visualize error")
    vis_dist = visualize_threshold
    R = np.array([[1, 0, 0]], dtype=np.float64)
    G = np.array([[0, 1, 0]], dtype=np.float64)
    B = np.array([[0, 0, 1]], dtype=np.float64)
    W = np.array([[1, 1, 1]], dtype=np.float64)
    data_color = np.tile(B, (data_down.shape[0], 1))
    data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
    data_color[np.where(inbound)[0][grid_inbound][in_obs][data_above]] = R * data_alpha + W * (1 - data_alpha)
    data_color[np.where(inbound)[0][grid_inbound][in_obs][data_above][dist_d2s[:, 0] >= max_dist]] = G
    write_vis_pcd(f"{out_dir}/vis_{scan:03}_d2s.ply", data_down, data_color)
    stl_color = np.tile(B, (stl.shape[0], 1))
    stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    stl_color[np.where(above)[0]] = R * stl_alpha + W * (1 - stl_alpha)
    stl_color[np.where(above)[0][dist_s2d[:, 0] >= max_dist]] = G
    write_vis_pcd(f"{out_dir}/vis_{scan:03}_s2d.ply", stl, stl_color)

    pbar.update(1)
    pbar.set_description("done")
    pbar.close()

    result = {
        "mean_d2s": mean_d2s,
        "mean_s2d": mean_s2d,
        "overall": (mean_d2s + mean_s2d) / 2,
        "n_data": data_down.shape[0],
        "n_data_eval": (dist_d2s < max_dist).sum().item(),
        "n_stl": stl.shape[0],
        "n_stl_eval": (dist_s2d < max_dist).sum().item(),
    }

    print(f"Saving results to {out_dir}/results.json")
    with open(f"{out_dir}/results.json", "w") as fp:
        json.dump(result, fp, indent=2)
    return result


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def cull_scan(mesh_path, result_mesh_file, instance_dir):
    # load poses
    image_dir = "{0}/images".format(instance_dir)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    n_images = len(image_paths)
    cam_file = "{0}/cameras.npz".format(instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict["scale_mat_%d" % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict["world_mat_%d" % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())

    # load mask
    mask_dir = "{0}/mask".format(instance_dir)
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    masks = []
    for p in mask_paths:
        mask = cv2.imread(p)
        masks.append(mask)

    # hard-coded image shape
    W, H = 1600, 1200

    # load mesh
    mesh = trimesh.load(mesh_path)

    # load transformation matrix

    vertices = mesh.vertices

    # project and filter
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()

    if len(masks) != n_images:
        print(f"mask number {len(masks)} != image number {n_images}, skip culling")
        mask = torch.ones_like(vertices[0, :], dtype=torch.bool).cpu().numpy()
    else:
        sampled_masks = []
        for i in tqdm(range(n_images), desc="Culling mesh given masks"):
            pose = pose_all[i]
            w2c = torch.inverse(pose).cuda()
            intrinsic = intrinsics_all[i].cuda()

            with torch.no_grad():
                # transform and project
                cam_points = intrinsic @ w2c @ vertices
                pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
                pix_coords = pix_coords.permute(1, 0)
                pix_coords[..., 0] /= W - 1
                pix_coords[..., 1] /= H - 1
                pix_coords = (pix_coords - 0.5) * 2
                valid = ((pix_coords > -1.0) & (pix_coords < 1.0)).all(dim=-1).float()

                # dialate mask similar to unisurf
                maski = masks[i][:, :, 0].astype(np.float32) / 256.0
                maski = torch.from_numpy(binary_dilation(maski, disk(24))).float()[None, None].cuda()

                sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode="nearest", padding_mode="zeros", align_corners=True)[0, -1, 0]

                sampled_mask = sampled_mask + (1.0 - valid)
                sampled_masks.append(sampled_mask)
        sampled_masks = torch.stack(sampled_masks, -1)
        mask = (sampled_masks > 0.0).all(dim=-1).cpu().numpy()

    if hasattr(mesh, "faces"):  # mesh mode
        face_mask = mask[mesh.faces].all(axis=1)
        mesh.update_faces(face_mask)
        mesh.update_vertices(mask)
    else:  # pcd mode
        mesh.vertices = mesh.vertices[mask]

    # transform vertices to world
    scale_mat = scale_mats[0]
    mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
    mesh.export(result_mesh_file)
    del mesh


def cull_and_eval(scan, output_dir, ply_file, DTU_dir, mode):
    result_mesh_file = os.path.join(output_dir, "culled_mesh.ply")
    instance_dir = f"{DTU_dir}/DTU/scan{scan}"

    os.makedirs(output_dir, exist_ok=True)
    cull_scan(ply_file, result_mesh_file, instance_dir)
    result = eval(result_mesh_file, scan, mode, DTU_dir, output_dir)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=int, required=True, help="DTU scan id")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory")
    parser.add_argument("--ply_file", type=str, required=True, help="path to the ply file to be evaluated")
    parser.add_argument("--DTU_dir", type=str, required=True, help="DTU dataset directory")
    parser.add_argument("--mode", type=str, help="one of mesh or pcd", default="mesh")
    args = parser.parse_args()

    cull_and_eval(args.scan, args.output_dir, args.ply_file, args.DTU_dir, args.mode)
