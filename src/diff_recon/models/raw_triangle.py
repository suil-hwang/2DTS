import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path
import os
from scipy.spatial import KDTree
from copy import deepcopy
from PIL import Image
import trimesh

from ..utils.gltf_utils import build_texture_atlas, ensure_rgba
from ..utils.sh_utils import SH2RGB, RGB2SH


class RawTriangle:
    def __init__(
        self,
        vertex: np.ndarray = None,
        opacity: np.ndarray = None,
        shs: np.ndarray = None,
        uv: np.ndarray = None,
        texture: np.ndarray = None,
        *,
        ply_path: str = None,
        glb_path: str = None,
    ) -> None:
        self.vertex = vertex
        self.opacity = opacity
        self.shs = shs
        self.uv = uv
        self.texture = texture

        if ply_path is not None:
            self.loadPLY(ply_path)
        if glb_path is not None:
            self.loadGLB(glb_path)

        self.contained_idx = np.ones(len(self), dtype=bool)

    @property
    def center(self):
        return self.vertex.mean(axis=1)

    def printStats(self):
        banner = "=" * 20 + " RawTriangle Stats " + "=" * 20
        print(banner)

        print(f"Number of points: {len(self)}")
        print(f"Number of SHs: {self.shs.shape[-1] // 3 if self.shs is not None and self.shs.size > 0 else 0}")
        print(f"Has texture: {self.hasTexture()}")
        print(f"x range: {self.vertex[..., 0].min():>8.1f} - {self.vertex[..., 0].max():>8.1f}")
        print(f"y range: {self.vertex[..., 1].min():>8.1f} - {self.vertex[..., 1].max():>8.1f}")
        print(f"z range: {self.vertex[..., 2].min():>8.1f} - {self.vertex[..., 2].max():>8.1f}")
        print(f"z mean: {self.vertex[..., 2].mean():>8.1f}")
        print(f"z median: {np.median(self.vertex[..., 2]):>8.1f}")

        print("=" * len(banner))

    def shDegree(self):
        if self.shs is None or self.shs.size == 0:
            return 0
        return int(np.sqrt(self.shs.shape[-1] / 3) - 1)

    def hasTexture(self) -> bool:
        return self.uv is not None and self.texture is not None and len(self.uv) == len(self) and self.texture.size > 0

    def __len__(self):
        return len(self.vertex) if self.vertex is not None else 0

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            idx = range(len(self))[idx]
            idx = slice(idx, idx + 1)
        return RawTriangle(
            self.vertex[idx] if self.vertex is not None else None,
            self.opacity[idx] if self.opacity is not None else None,
            self.shs[idx] if self.shs is not None else None,
            self.uv[idx] if self.uv is not None else None,
            self.texture.copy() if self.texture is not None else None,
        )

    def __iadd__(self, other):
        if len(other) == 0:
            return self

        if len(self) == 0:
            self.vertex = other.vertex
            self.opacity = other.opacity
            self.shs = other.shs
            self.uv = other.uv
            self.texture = other.texture.copy() if other.texture is not None else None
            self.contained_idx = other.contained_idx.copy() if other.contained_idx is not None else None
            return self

        self_has_texture = self.hasTexture()
        other_has_texture = other.hasTexture()

        self.vertex = np.concatenate((self.vertex, other.vertex))
        self.opacity = np.concatenate((self.opacity, other.opacity)) if self.opacity is not None else other.opacity
        self.shs = np.concatenate((self.shs, other.shs)) if self.shs is not None else other.shs

        if self_has_texture and other_has_texture:
            atlas, transforms = build_texture_atlas([self.texture, other.texture])
            self.uv = np.concatenate((self._remap_uv(self.uv, transforms[0]), self._remap_uv(other.uv, transforms[1])), axis=0)
            self.texture = atlas[..., :3] if atlas.shape[2] == 4 and np.allclose(atlas[..., 3], 1.0) else atlas
        elif self_has_texture != other_has_texture:
            self.uv = None
            self.texture = None

        self.contained_idx = np.concatenate((self.contained_idx, other.contained_idx)) if self.contained_idx is not None else other.contained_idx
        return self

    def __add__(self, other):
        result = deepcopy(self)
        result += other
        return result

    def resetContainedIdx(self):
        self.contained_idx = np.ones(len(self), dtype=bool)

    def __isub__(self, other):
        if len(other) == 0:
            return self

        kdTree = KDTree(other.center)
        distance, _ = kdTree.query(self.center)
        self.contained_idx &= distance > 1e-5
        self.reduce()
        return self

    def __sub__(self, other):
        diff = deepcopy(self)
        diff -= other
        return diff

    def reduce(self):
        if np.all(self.contained_idx):
            return RawTriangle()

        removed_triangle = self[~self.contained_idx]
        self.vertex, self.opacity, self.shs, self.uv = (
            self.vertex[self.contained_idx],
            self.opacity[self.contained_idx],
            self.shs[self.contained_idx],
            self.uv[self.contained_idx] if self.uv is not None else None,
        )
        self.resetContainedIdx()

        return removed_triangle

    def replace(self, indices, other):
        if len(indices) != len(other):
            raise ValueError(f"Expected {len(indices)} replacement triangles, got {len(other)}")

        if self.uv is not None or other.uv is not None:
            if self.uv is None or other.uv is None:
                raise ValueError("Cannot replace textured triangles with non-textured triangles or vice versa")
            if self.texture is not None and other.texture is not None and not np.array_equal(self.texture, other.texture):
                raise ValueError("Replace only supports textured triangles sharing the same texture atlas")

        self.vertex[indices] = other.vertex
        self.opacity[indices] = other.opacity
        self.shs[indices] = other.shs
        if self.uv is not None:
            self.uv[indices] = other.uv

    def loadPLY(self, path):
        if not os.path.exists(path):
            print(f"[Warning] File {path} does not exist! From loadPLY function in RawTriangle class.")
            return

        self.ply_path = path

        try:
            plydata = PlyData.read(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return

        element = plydata.elements[0]
        element_keys = [p.name for p in element.properties]
        use_vertex_color = "f_dc_0_0" in element_keys

        vertex_properties = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3"]
        vertex = np.stack([np.asarray(element[name]) for name in vertex_properties], axis=1).astype(np.float32).reshape(-1, 3, 3)
        opacities = np.asarray(element["opacity"])[..., np.newaxis].astype(np.float32)

        dc_f_names = sorted([p for p in element_keys if p.startswith("f_dc_")])
        features_dc = np.stack([np.asarray(element[name]) for name in dc_f_names], axis=1)
        if use_vertex_color:
            features_dc = features_dc.reshape(-1, 3, 3)

        extra_f_names = sorted([p for p in element_keys if p.startswith("f_rest_")], key=lambda x: tuple(map(int, x.split("_")[2:])))
        if len(extra_f_names) > 0:
            features_extra = np.stack([np.asarray(element[name]) for name in extra_f_names], axis=1)
            if use_vertex_color:
                features_extra = features_extra.reshape(-1, 3, features_extra.shape[1] // 3)
            shs = np.concatenate([features_dc, features_extra], axis=-1).astype(np.float32)
        else:
            shs = features_dc

        self.vertex = vertex
        self.opacity = opacities
        self.shs = shs
        self.uv = None
        self.texture = None
        self.resetContainedIdx()

        assert len(self.vertex) == len(self.opacity) == len(self.shs)
        return self

    def savePLY(self, path, save_empty=False, save_extra=False):
        if not save_empty and len(self) == 0:
            return

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        use_vertex_color = len(self.shs.shape) == 3
        f_dc, f_rest = self.shs[..., :3], self.shs[..., 3:]
        property_names = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "opacity"]
        if use_vertex_color:
            property_names += [f"f_dc_{i}_{j}" for i in range(3) for j in range(3)]
            if save_extra:
                property_names += [f"f_rest_{i}_{j}" for i in range(3) for j in range(f_rest.shape[-1])]
        else:
            property_names += [f"f_dc_{i}" for i in range(3)]
            if save_extra:
                property_names += [f"f_rest_{i}" for i in range(f_rest.shape[-1])]

        dtype_full = [(name, "f4") for name in property_names]
        elements = np.empty(len(self), dtype=dtype_full)

        n_point = len(self.vertex)
        color_count = 3 if use_vertex_color else 1
        attributes = [self.vertex.reshape(n_point, 9), self.opacity, f_dc.reshape(n_point, color_count * 3)]
        if save_extra:
            attributes.append(f_rest.reshape(n_point, color_count * f_rest.shape[-1]))
        attributes = np.concatenate(attributes, axis=1)
        for column, name in enumerate(elements.dtype.names):
            elements[name] = attributes[:, column]

        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def saveGLB(self, path, save_empty=False, save_back=True, process=False):
        if not save_empty and len(self) == 0:
            return

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        has_texture = self.hasTexture()
        vertices = self.vertex.reshape(-1, 3)
        faces = np.arange(len(vertices), dtype=np.int32 if has_texture else np.int64).reshape(-1, 3)
        if save_back:
            faces = np.concatenate([faces, faces[:, ::-1]], axis=0)
        opacity = 1 / (1 + np.exp(-self.opacity))

        if has_texture:
            texture_rgba = ensure_rgba(self.texture)
            texture_img = Image.fromarray(np.clip(texture_rgba * 255, 0, 255).astype(np.uint8), mode="RGBA")
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture_img,
                baseColorFactor=[255, 255, 255, 255],
                metallicFactor=0.0,
                roughnessFactor=1.0,
            )
            visual = trimesh.visual.TextureVisuals(uv=self.uv.reshape(-1, 2), material=material)
            opacity_flat = np.repeat(opacity.astype(np.float32).reshape(-1), 3, axis=0)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, vertex_attributes={"opacity": opacity_flat}, process=process)
        else:
            color = np.clip(SH2RGB(self.shs[..., :3]), 0, 1)
            use_vertex_color = len(self.shs.shape) == 3
            if use_vertex_color:
                rgba = np.concatenate([color, opacity[:, None, :].repeat(3, axis=1)], axis=-1)
            else:
                rgba = np.concatenate([color, opacity], axis=-1)
            rgba = (rgba * 255).astype(np.uint8)
            if save_back and not use_vertex_color:
                rgba = np.concatenate([rgba, rgba], axis=0)

            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                face_colors=rgba if not use_vertex_color else None,
                vertex_colors=rgba.reshape(-1, 4) if use_vertex_color else None,
                process=process,
            )
        mesh.export(path)

    def loadGLB(self, path):
        self.glb_path = path
        mesh = trimesh.load_mesh(path, process=False)

        vertices = np.asarray(mesh.vertices[mesh.faces], dtype=np.float32)
        mesh_uv = getattr(mesh.visual, "uv", None)
        material = getattr(mesh.visual, "material", None)
        base_texture = getattr(material, "baseColorTexture", None)
        base_factor = getattr(material, "baseColorFactor", None)

        uv = texture = None
        if mesh_uv is not None and (base_texture is not None or base_factor is not None):
            texture = ensure_rgba(np.asarray(base_texture.convert("RGBA"), dtype=np.float32) / 255.0 if base_texture is not None else None)
            if base_factor is not None:
                factor = np.asarray(base_factor, dtype=np.float32)
                if factor.max() > 1.0:
                    factor /= 255.0
                texture = np.clip(texture * factor.reshape(1, 1, 4), 0.0, 1.0)

            uv = np.asarray(mesh_uv[mesh.faces], dtype=np.float32)
            rgba = self._sample_texture(texture, uv.reshape(-1, 2)).reshape(-1, 3, 4)

            opacity_attr = None
            attributes = getattr(mesh, "vertex_attributes", {})
            if "_opacity" in attributes:
                opacity_attr = np.asarray(attributes["_opacity"], dtype=np.float32)
            elif "opacity" in attributes:
                opacity_attr = np.asarray(attributes["opacity"], dtype=np.float32)
            vertex_colors = getattr(mesh.visual, "vertex_colors", None)
            if opacity_attr is not None:
                opacity = opacity_attr.reshape(-1)[mesh.faces].mean(axis=1, keepdims=True)
            elif vertex_colors is not None:
                vertex_alpha = np.asarray(vertex_colors, dtype=np.float32)[..., 3:4] / 255.0
                opacity = vertex_alpha[mesh.faces].mean(axis=1)
            else:
                opacity = rgba[..., 3].mean(axis=1, keepdims=True)
            texture = texture[..., :3] if np.allclose(texture[..., 3], 1.0) else texture
        elif mesh.visual.kind == "vertex":
            rgba = np.asarray(mesh.visual.vertex_colors[mesh.faces], dtype=np.float32) / 255.0
            opacity = rgba[:, 0, 3:]
        else:
            rgba = np.asarray(mesh.visual.face_colors, dtype=np.float32)[: len(vertices)] / 255.0
            opacity = rgba[:, 3:]

        self.vertex = vertices
        self.opacity = self._alpha_to_logit(opacity)
        self.shs = RGB2SH(rgba[..., :3].astype(np.float32)).astype(np.float32)
        self.uv = uv
        self.texture = texture
        self.resetContainedIdx()
        return self

    @staticmethod
    def _alpha_to_logit(alpha: np.ndarray) -> np.ndarray:
        alpha = np.clip(alpha, 1e-5, 1.0 - 1e-5)
        return np.log(alpha / (1.0 - alpha)).astype(np.float32)

    @staticmethod
    def _remap_uv(uv: np.ndarray, transform: np.ndarray) -> np.ndarray:
        remapped = uv.astype(np.float32, order="C")
        remapped[..., 0] = transform[2] + remapped[..., 0] * transform[0]
        remapped[..., 1] = transform[3] + remapped[..., 1] * transform[1]
        return remapped

    @staticmethod
    def _wrap_uv(coord: np.ndarray) -> np.ndarray:
        return np.mod(coord, 1.0)

    @classmethod
    def _sample_texture(cls, texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
        texture_rgba = ensure_rgba(texture)
        u = cls._wrap_uv(uv[:, 0])
        v = cls._wrap_uv(1.0 - uv[:, 1])

        height, width = texture_rgba.shape[:2]
        x = np.clip(u * (width - 1), 0.0, width - 1)
        y = np.clip(v * (height - 1), 0.0, height - 1)

        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, width - 1)
        y1 = np.clip(y0 + 1, 0, height - 1)

        wx = (x - x0).astype(np.float32)[:, None]
        wy = (y - y0).astype(np.float32)[:, None]

        top = texture_rgba[y0, x0] * (1.0 - wx) + texture_rgba[y0, x1] * wx
        bottom = texture_rgba[y1, x0] * (1.0 - wx) + texture_rgba[y1, x1] * wx
        return (top * (1.0 - wy) + bottom * wy).astype(np.float32)
