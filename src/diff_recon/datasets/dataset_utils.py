import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist

from .colmap_loader import CameraInfo, readColmapCameras, read_intrinsics_binary, read_intrinsics_text
from ..utils.camera import rotmat2qvec
from ..models.point_cloud import PointCloud


def read_validated_colmap_cameras(images_path: str, cameras_path: str, images_folder: str) -> list[CameraInfo]:
    """Keep legacy parsing, then validate calibration before cameras reach the renderer."""
    cam_infos = readColmapCameras(images_path, cameras_path, images_folder)
    intrinsics = read_intrinsics_binary(cameras_path) if cameras_path.endswith(".bin") else read_intrinsics_text(cameras_path)
    for camera_id in {cam.camera_id for cam in cam_infos}:
        intr = intrinsics[camera_id]
        if intr.width <= 0 or intr.height <= 0 or not np.isfinite(intr.params).all():
            raise ValueError(f"Camera {intr.id} requires positive image dimensions and finite intrinsics")
        if intr.model == "PINHOLE":
            if len(intr.params) != 4:
                raise ValueError(f"Camera {intr.id} PINHOLE requires four parameters")
            fx, fy, cx, cy = intr.params
        elif intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            expected = 4 if intr.model == "SIMPLE_RADIAL" else 3
            if len(intr.params) != expected:
                raise ValueError(f"Camera {intr.id} {intr.model} requires {expected} parameters")
            fx, cx, cy = intr.params[:3]
            fy = fx
            if intr.model == "SIMPLE_RADIAL" and abs(intr.params[3]) > 1e-12:
                raise ValueError(f"Camera {intr.id} must be undistorted; run COLMAP image_undistorter before loading")
        else:
            raise ValueError(f"Camera model {intr.model} is unsupported; only centered, undistorted pinhole cameras are supported")
        if fx <= 0 or fy <= 0:
            raise ValueError(f"Camera {intr.id} requires positive focal lengths")
        # Preserve the existing half-pixel convention tolerance; offsets are not corrected.
        if abs(cx - intr.width / 2) > 0.5 or abs(cy - intr.height / 2) > 0.5:
            raise ValueError(f"Camera {intr.id} principal point must be centered within 0.5 pixels; only centered, undistorted cameras are supported")
    return [cam._replace(image_name=Path(cam.image_path).stem) for cam in cam_infos]


def getCameraExtent(cam_infos: list[CameraInfo]) -> float:
    if not cam_infos:
        raise ValueError("At least one camera is required to estimate the scene extent")
    cam_centers = [-cam.R @ cam.T for cam in cam_infos]
    cam_centers = np.stack(cam_centers, axis=0)
    if not np.isfinite(cam_centers).all():
        raise ValueError("Camera centers must be finite")
    center = cam_centers.mean(axis=0, keepdims=True)
    extent = np.linalg.norm(cam_centers - center, axis=1).max() * 1.1
    # Coincident centers provide no scene scale; use a unit scale instead of a zero near plane/loss divisor.
    return extent if extent > 0 else 1.0


def camInfosToColmap(cam_infos: list[CameraInfo], save_dir: str) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    cam_dict = {}
    quats = rotmat2qvec(np.array([cam.R.T for cam in cam_infos]).reshape(-1, 3, 3))  # world-to-camera, as COLMAP stores it
    for cam in cam_infos:
        width, height = cam.width, cam.height
        fx = 0.5 * width / np.tan(0.5 * cam.FovX)
        fy = 0.5 * height / np.tan(0.5 * cam.FovY)
        intrinsics = (width, height, fx, fy)
        if cam.camera_id in cam_dict and cam_dict[cam.camera_id] != intrinsics:
            raise ValueError(f"Camera ID {cam.camera_id} has inconsistent parameters: {cam_dict[cam.camera_id]} vs {intrinsics}")
        cam_dict[cam.camera_id] = intrinsics
    with open(f"{save_dir}/images.txt", "w", encoding="utf-8") as f_images:
        for i, (cam, quat) in enumerate(zip(cam_infos, quats)):
            f_images.write(f"{i} {quat[0]} {quat[1]} {quat[2]} {quat[3]} {cam.T[0]} {cam.T[1]} {cam.T[2]} {cam.camera_id} {cam.image_name}\n\n")

    with open(f"{save_dir}/cameras.txt", "w", encoding="utf-8") as f_cameras:
        for cam_id, (width, height, fx, fy) in cam_dict.items():
            f_cameras.write(f"{cam_id} PINHOLE {width} {height} {fx} {fy} {width / 2} {height / 2}\n")


def pointCloudToColmap(pcd: PointCloud, save_dir: str) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/points3D.txt", "w", encoding="utf-8") as f:
        for i, (point, color) in enumerate(zip(pcd.points, pcd.colors)):
            color = (color * 255).astype(np.uint8)
            f.write(f"{i} {point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]} {0}\n")


def jsonCamerasToCameraInfos(json_data: list[dict]) -> list[CameraInfo]:
    cam_infos = []
    for cam in json_data:
        R = np.array(cam["rotation"], dtype=np.float32) @ np.diag([1, -1, -1])
        T = -R.T @ np.array(cam["position"], dtype=np.float32)
        fovx = 2 * np.arctan(cam["width"] / (2 * cam["fx"]))
        fovy = 2 * np.arctan(cam["height"] / (2 * cam["fy"]))
        cam_info = CameraInfo(
            camera_id=cam["id"],
            R=R,
            T=T,
            FovY=fovy,
            FovX=fovx,
            image_path=None,
            image_name=None,
            width=cam["width"],
            height=cam["height"],
        )
        cam_infos.append(cam_info)
    return cam_infos


def cameraInfosToJsonCameras(cam_infos: list[CameraInfo]) -> list[dict]:
    json_data = []
    for cam in cam_infos:
        fx = 0.5 * cam.width / np.tan(0.5 * cam.FovX)
        fy = 0.5 * cam.height / np.tan(0.5 * cam.FovY)
        position = (-cam.R @ cam.T).tolist()
        rotation = (cam.R @ np.diag([1, -1, -1])).tolist()
        json_cam = dict(
            id=cam.camera_id,
            width=cam.width,
            height=cam.height,
            fx=fx,
            fy=fy,
            position=position,
            rotation=rotation,
        )
        json_data.append(json_cam)
    return json_data


def interpolateCameraInfos(
    source_cam_infos: list[CameraInfo],
    target_cam_infos: list[CameraInfo],
    num_interp: int,
    ground_z: float = None,
) -> list[CameraInfo]:
    """
    Find the closest source camera view for each target view and interpolate between these pairs.

    Args:
        source_cam_infos: List of source camera information
        target_cam_infos: List of target camera information
        num_interp: Number of interpolation steps between each source-target pair

    Returns:
        List of interpolated camera information
    """
    if not isinstance(num_interp, (int, np.integer)) or isinstance(num_interp, bool) or num_interp < 0:
        raise ValueError("num_interp must be a nonnegative integer")
    if num_interp == 0 or not target_cam_infos:
        return []
    if not source_cam_infos:
        raise ValueError("At least one source camera is required for interpolation")
    # Extract camera positions and rotations
    s_positions = np.array([-cam.R @ cam.T for cam in source_cam_infos])  # Camera centers in world coordinates
    t_positions = np.array([-cam.R @ cam.T for cam in target_cam_infos])

    s_Rs = np.array([cam.R for cam in source_cam_infos])  # Camera to world rotation
    t_Rs = np.array([cam.R for cam in target_cam_infos])

    # Find closest source camera for each target camera
    if ground_z is not None:
        s_lookat = s_Rs[:, :, 2]
        t_lookat = t_Rs[:, :, 2]
        if not np.isfinite(ground_z) or np.any(np.abs(s_lookat[:, 2]) <= 1e-12) or np.any(np.abs(t_lookat[:, 2]) <= 1e-12):
            raise ValueError("Camera viewing directions must intersect the finite ground plane")
        s_lookat_pos = s_positions + s_lookat * (ground_z - s_positions[:, 2:3]) / s_lookat[:, 2:3]
        t_lookat_pos = t_positions + t_lookat * (ground_z - t_positions[:, 2:3]) / t_lookat[:, 2:3]
        position_dist = cdist(t_lookat_pos, s_lookat_pos)
    else:
        position_dist = cdist(t_positions, s_positions)  # [num_targets, num_sources]
    rotation_dist = 1 - (t_Rs[:, None, :, 0] * s_Rs[None, :, :, 0]).sum(axis=-1)
    blend_dist = position_dist / 200 + rotation_dist
    closest_source_indices = np.argmin(blend_dist, axis=1)

    # Interpolate all targets at once per step: SLERP (as scipy's Slerp does) for rotation, LERP for camera center
    alphas = np.sqrt(np.linspace(0, 1, num_interp + 1)[1:])
    source_rots = Rotation.from_matrix(s_Rs[closest_source_indices])
    rel_rotvecs = (source_rots.inv() * Rotation.from_matrix(t_Rs)).as_rotvec()  # shortest arc from each source to its target
    interp_Rs = np.stack([(source_rots * Rotation.from_rotvec(alpha * rel_rotvecs)).as_matrix() for alpha in alphas])  # (num_interp, N, 3, 3)
    interp_positions = (1 - alphas)[:, None, None] * s_positions[closest_source_indices] + alphas[:, None, None] * t_positions
    interp_Ts = -np.einsum("jnki,jnk->jni", interp_Rs, interp_positions)  # T = -R^T @ camera center

    return [
        CameraInfo(
            camera_id=0,
            R=interp_Rs[j, i],
            T=interp_Ts[j, i],
            FovY=target_cam.FovY,
            FovX=target_cam.FovX,
            image_path=None,
            image_name=f"interp_{j:02d}_{i:04d}",
            width=target_cam.width,
            height=target_cam.height,
        )
        for j in range(num_interp)
        for i, target_cam in enumerate(target_cam_infos)
    ]
