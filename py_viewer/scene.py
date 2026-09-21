"""Scene settings, camera geometry, and file loading; no graphics context or Torch at import."""
from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial.transform import Rotation
import trimesh
import yaml


@dataclass
class RenderSettings:
    mode: str = "RGB"
    white_background: bool = True
    resolution: int = 800
    wireframe: bool = False
    cull_backfaces: bool = False
    gamma: float = 1.0
    sh_degree: int = 3
    sort_level: int = 2
    threshold_enabled: bool = False
    threshold: float = 0.3
    gamma_rescale: bool = False
    supersampling: int = 1


@dataclass
class CameraView:
    name: str
    c2w: np.ndarray
    fov_y: float
    aspect: float
    image_path: Path

    def image(self, background: float) -> np.ndarray:
        rgba = np.asarray(Image.open(self.image_path).convert("RGBA"), dtype=np.float32) / 255
        return rgba[..., :3] * rgba[..., 3:] + background * (1 - rgba[..., 3:])


def load_views(path: Path) -> dict[str, list[CameraView]]:
    """Read NeRF/Blender transforms without starting dataset workers or training."""
    files = sorted(path.glob("transforms_*.json")) if path.is_dir() else [path]
    views = {}
    for file in files:
        metadata = json.loads(file.read_text(encoding="utf-8"))
        group = file.stem.removeprefix("transforms_")
        views[group] = []
        for index, frame in enumerate(metadata["frames"]):
            image_path = file.parent / frame["file_path"]
            if not image_path.suffix:
                image_path = image_path.with_suffix(".png")
            with Image.open(image_path) as image:
                width, height = image.size
            fov_y = 2 * math.atan(math.tan(metadata["camera_angle_x"] / 2) * height / width)
            views[group].append(CameraView(f"{group}/{index}: {image_path.stem}",
                                          np.asarray(frame["transform_matrix"], dtype=float),
                                          fov_y, width / height, image_path))
    return views


class OrbitCamera:
    def __init__(self):
        self.c2w = np.eye(4)
        self.target = np.zeros(3)
        self.radius = 1.0
        self.fov_y = math.radians(50)
        self.near, self.far = 0.001, 100.0

    def fit(self, bounds: np.ndarray):
        self.target = np.asarray(bounds).mean(axis=0)
        self.radius = max(float(np.linalg.norm(bounds[1] - bounds[0])) / 2, 1e-4)
        back = np.array([1.0, -1.5, 0.8])
        back /= np.linalg.norm(back)
        right = np.cross([0, 0, 1], back)
        right /= np.linalg.norm(right)
        self.c2w = np.eye(4)
        self.c2w[:3, :3] = np.column_stack([right, np.cross(back, right), back])
        self.c2w[:3, 3] = self.target + back * self.radius / math.sin(self.fov_y / 2) * 1.1
        self.update_clip()

    def set_view(self, view: CameraView):
        distance = max(np.linalg.norm(self.c2w[:3, 3] - self.target), self.radius)
        self.c2w = view.c2w.copy()
        self.target = self.c2w[:3, 3] - self.c2w[:3, 2] * distance
        self.fov_y = view.fov_y
        self.update_clip()

    def update_clip(self):
        distance = np.linalg.norm(self.c2w[:3, 3] - self.target)
        self.near = self.radius * 0.001
        self.far = max(distance + self.radius * 10, self.near * 100)

    def orbit(self, dx: float, dy: float):
        rotation = Rotation.from_rotvec(self.c2w[:3, 1] * (-dx * 0.006)).as_matrix()
        rotation = Rotation.from_rotvec((rotation @ self.c2w[:3, 0]) * (-dy * 0.006)).as_matrix() @ rotation
        self.c2w[:3, :3] = rotation @ self.c2w[:3, :3]
        self.c2w[:3, 3] = self.target + rotation @ (self.c2w[:3, 3] - self.target)

    def pan(self, dx: float, dy: float, height: float):
        distance = np.linalg.norm(self.c2w[:3, 3] - self.target)
        scale = 2 * distance * math.tan(self.fov_y / 2) / max(height, 1)
        offset = (-dx * self.c2w[:3, 0] + dy * self.c2w[:3, 1]) * scale
        self.c2w[:3, 3] += offset
        self.target += offset

    def dolly(self, amount: float):
        offset = self.c2w[:3, 3] - self.target
        distance = max(np.linalg.norm(offset) * math.exp(-amount * 0.12), self.radius * 0.002)
        self.c2w[:3, 3] = self.target + offset / np.linalg.norm(offset) * distance
        self.update_clip()

    def matrix(self, aspect: float) -> np.ndarray:
        near, far = self.near, self.far
        scale = 1 / math.tan(self.fov_y / 2)
        projection = np.array([[scale / aspect, 0, 0, 0], [0, scale, 0, 0],
                               [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                               [0, 0, -1, 0]])
        return projection @ np.linalg.inv(self.c2w)

    def ts_camera(self, width: int, height: int, device):
        from src.diff_recon.utils.camera import Camera

        c2w = self.c2w.copy()
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        return Camera(R=w2c[:3, :3].T, T=w2c[:3, 3],
                      FoVx=2 * math.atan(math.tan(self.fov_y / 2) * width / height),
                      FoVy=self.fov_y, image_width=width, image_height=height,
                      znear=self.near, zfar=self.far).to(device)


def model_kind(path: Path) -> str:
    if path.suffix.lower() == ".ckpt":
        return "2DTS"
    if path.suffix.lower() == ".ply":
        ply = PlyData.read(str(path))
        names = {prop.name for prop in ply.elements[0].properties}
        if {"x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "opacity"} <= names:
            return "2DTS"
    return "Mesh"


def find_config(path: Path) -> Path | None:
    candidates = [path.parent / "config.yaml", path.parent.parent / "config.yaml",
                  path.parent.parent.parent / f"{path.parent.parent.name}.yaml"]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def camera_path(config_path: Path | None) -> Path | None:
    if config_path is None:
        return None
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = config.get("dataset", {})
    if dataset.get("type") == "NerfSynthetic" and dataset.get("local_dir"):
        path = Path(dataset["local_dir"]) / dataset["scene_id"]
        if (path / "transforms_test.json").is_file():
            return path
    return None


@dataclass
class MeshPart:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray
    uv: np.ndarray
    texture: np.ndarray


def load_mesh(path: Path) -> tuple[list[MeshPart], np.ndarray]:
    scene = trimesh.load_scene(path, process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        mesh.apply_transform(transform)
        if not len(mesh.faces):
            continue
        visual = mesh.visual
        if visual.kind == "texture":
            material = visual.material
            assignments = getattr(visual, "face_materials", None)
            materials = material.materials if hasattr(material, "materials") else [material]
            assignments = np.zeros(len(mesh.faces), dtype=int) if assignments is None else assignments
            for material_id, material in enumerate(materials):
                faces = mesh.faces[assignments == material_id]
                if not len(faces):
                    continue
                image = getattr(material, "baseColorTexture", None)
                if image is None:
                    image = getattr(material, "image", None)
                factor = getattr(material, "baseColorFactor", None)
                if factor is None:
                    factor = getattr(material, "diffuse", [255, 255, 255, 255])
                colors = np.broadcast_to(np.asarray(factor)[:3] / 255, (len(faces), 3, 3)).copy()
                texture = np.asarray(image.convert("RGBA")) if image is not None else np.full((1, 1, 4), 255, np.uint8)
                uv = np.asarray(visual.uv) if visual.uv is not None else np.zeros((len(mesh.vertices), 2))
                parts.append(MeshPart(mesh.vertices, faces, colors, uv, texture))
        else:
            if visual.kind == "vertex":
                colors = np.asarray(visual.vertex_colors)[mesh.faces, :3] / 255
            else:
                colors = np.repeat(np.asarray(visual.face_colors)[:, None, :3] / 255, 3, axis=1)
            parts.append(MeshPart(mesh.vertices, mesh.faces, colors, np.zeros((len(mesh.vertices), 2)),
                                  np.full((1, 1, 4), 255, np.uint8)))
    if not parts:
        raise ValueError("The file contains no triangle mesh.")
    bounds = np.array([np.min([part.vertices.min(axis=0) for part in parts], axis=0),
                       np.max([part.vertices.max(axis=0) for part in parts], axis=0)])
    return parts, bounds
