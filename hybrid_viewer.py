import argparse
import math
import os
from time import time

import cv2
import numpy as np
import torch
import trimesh
import viser
from PIL import Image
from viser import ClientHandle

from src.diff_recon import AnimatedTriangle, HybridGTModel, RawTriangle
from src.diff_recon.utils.gltf_utils import ensure_rgba
from src.diff_recon.utils.camera import Camera, qvec2rotmat, rotmat2qvec
from src.diff_recon.utils.vis_utils import alpha_to_image, depth_to_image, normal_to_image


REST_POSE_LABEL = "Rest Pose"


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return fallback.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _look_at_rotation(camera_position: np.ndarray, target: np.ndarray) -> np.ndarray:
    default_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    forward = _normalize(target - camera_position, default_rotation[:, 2])
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(np.dot(forward, up)) > 0.99:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    right = _normalize(np.cross(forward, up), default_rotation[:, 0])
    down = _normalize(np.cross(forward, right), default_rotation[:, 1])
    rotation = np.stack([right, down, forward], axis=1)
    return rotation.astype(np.float32)


def _compute_scene_stats(model: HybridGTModel) -> tuple[np.ndarray, float]:
    points = []
    if model.gau_xyz.numel() > 0:
        points.append(model.gau_xyz.detach())
    if model.tri_vertex.numel() > 0:
        points.append(model.tri_vertex.detach().reshape(-1, 3))

    if not points:
        raise ValueError("HybridGTModel has no geometry loaded.")

    scene_points = torch.cat(points, dim=0)
    scene_center = scene_points.mean(dim=0)
    scene_extent = torch.linalg.norm(scene_points - scene_center, dim=1).max().item()
    scene_extent = max(scene_extent, 1.0)
    return scene_center.cpu().numpy().astype(np.float32), float(scene_extent)


def _triangle_to_mesh(raw_triangle: RawTriangle) -> tuple[np.ndarray, np.ndarray]:
    vertices = raw_triangle.vertex.reshape(-1, 3).astype(np.float32)
    faces = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
    return vertices, faces


def _triangle_to_trimesh(raw_triangle: RawTriangle) -> trimesh.Trimesh:
    vertices, faces = _triangle_to_mesh(raw_triangle)

    if raw_triangle.hasTexture():
        texture_rgba = ensure_rgba(raw_triangle.texture)
        texture_img = Image.fromarray(np.clip(texture_rgba * 255.0, 0, 255).astype(np.uint8), mode="RGBA")
        visual = trimesh.visual.TextureVisuals(
            uv=raw_triangle.uv.reshape(-1, 2).astype(np.float32),
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture_img,
                baseColorFactor=[255, 255, 255, 255],
                metallicFactor=0.0,
                roughnessFactor=1.0,
            ),
        )
        return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)

    rgb = np.clip(raw_triangle.shs[..., :3].reshape(-1, 3), 0.0, 1.0)
    rgba = np.concatenate([rgb, np.ones((len(rgb), 1), dtype=np.float32)], axis=1)
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=(rgba * 255).astype(np.uint8), process=False)


class HybridVisClient:
    def __init__(
        self,
        client: ClientHandle,
        model: HybridGTModel,
        mesh_vertices: np.ndarray,
        mesh_faces: np.ndarray,
        scene_center: np.ndarray,
        scene_extent: float,
        output_dir: str,
    ):
        self._client_handle: ClientHandle = client
        self._model = model
        self._raw_triangle: RawTriangle | None = None
        self._mesh_vertices = mesh_vertices
        self._mesh_faces = mesh_faces
        self._scene_center = scene_center
        self._scene_extent = scene_extent
        self._output_dir = output_dir
        self._curr_image = None
        self._mesh_dict = {}

        client.scene.enable_default_lights(True)
        self._update = True
        self._terminate = False

        self._register_gui()
        self._set_initial_camera()

    def _display_mesh(self, use_wireframe: bool = False):
        if use_wireframe:
            mesh_name = "hybrid_mesh_wireframe"
            if mesh_name not in self._mesh_dict:
                self._mesh_dict[mesh_name] = self._client_handle.scene.add_mesh_simple(
                    mesh_name,
                    self._mesh_vertices,
                    self._mesh_faces,
                    color=(128, 128, 128),
                    wireframe=True,
                    visible=False,
                )
            else:
                self._mesh_dict[mesh_name].vertices = self._mesh_vertices
                self._mesh_dict[mesh_name].faces = self._mesh_faces
                self._mesh_dict[mesh_name].wireframe = True
        else:
            mesh_name = "hybrid_mesh_textured"
            if mesh_name in self._mesh_dict:
                self._mesh_dict[mesh_name].remove()
            self._mesh_dict[mesh_name] = self._client_handle.scene.add_mesh_trimesh(
                mesh_name,
                _triangle_to_trimesh(self._raw_triangle),
                visible=False,
            )

        for name, handle in self._mesh_dict.items():
            handle.visible = name == mesh_name

    def set_triangle(self, raw_triangle: RawTriangle):
        self._raw_triangle = raw_triangle

    def update_mesh(self, raw_triangle: RawTriangle, mesh_vertices: np.ndarray, mesh_faces: np.ndarray):
        self._raw_triangle = raw_triangle
        self._mesh_vertices = mesh_vertices
        self._mesh_faces = mesh_faces
        with self._client_handle.atomic():
            if "hybrid_mesh_wireframe" in self._mesh_dict:
                self._mesh_dict["hybrid_mesh_wireframe"].vertices = self._mesh_vertices
                self._mesh_dict["hybrid_mesh_wireframe"].faces = self._mesh_faces
            if "hybrid_mesh_textured" in self._mesh_dict:
                self._mesh_dict["hybrid_mesh_textured"].remove()
                del self._mesh_dict["hybrid_mesh_textured"]
            self._update_mesh_visibility()

    def request_update(self):
        self._update = True

    def _update_mesh_visibility(self):
        if not self._show_mesh_ckbx.value:
            for handle in self._mesh_dict.values():
                handle.visible = False
            return
        self._display_mesh(use_wireframe=self._wireframe_ckbx.value)

    def _register_gui(self):
        client: ClientHandle = self._client_handle

        @client.camera.on_update
        async def _(_):
            self._update = True

        self._render_mode_dropdown = client.gui.add_dropdown("Render Mode", ["image", "depth", "normal", "alpha"], "image")

        @self._render_mode_dropdown.on_update
        async def _(_):
            self._update = True

        self._background_dropdown = client.gui.add_dropdown("Background", ["black", "white"], "black")

        @self._background_dropdown.on_update
        async def _(_):
            self._update = True

        self._resolution_slider = client.gui.add_slider("Resolution", 384, 4096, 1, 1024)

        @self._resolution_slider.on_update
        async def _(_):
            self._update = True

        self._back_culling_ckbx = client.gui.add_checkbox("Back Culling", self._model.back_culling)

        @self._back_culling_ckbx.on_update
        async def _(_):
            self._update = True

        self._sort_level_slider = client.gui.add_slider("Sort Level", 0, 2, 1, self._model.sort_level)

        @self._sort_level_slider.on_update
        async def _(_):
            self._update = True

        self._tri_gamma_slider = client.gui.add_slider("Triangle Gamma", 1.0, 200.0, 0.5, float(self._model.tri_gamma))

        @self._tri_gamma_slider.on_update
        async def _(_):
            self._update = True

        self._gau_gamma_slider = client.gui.add_slider("Gaussian Gamma", 0.1, 50.0, 0.1, float(self._model.gau_gamma))

        @self._gau_gamma_slider.on_update
        async def _(_):
            self._update = True

        self._ambient_slider = client.gui.add_slider("Ambient", 0.0, 5.0, 0.01, float(self._model.ambient_intensity))

        @self._ambient_slider.on_update
        async def _(_):
            self._update = True

        self._light_intensity_slider = client.gui.add_slider("Light Intensity", 0.0, 10.0, 0.01, float(self._model.light_intensity))

        @self._light_intensity_slider.on_update
        async def _(_):
            self._update = True

        self._light_dir_x_slider = client.gui.add_slider("Light Dir X", -1.0, 1.0, 0.01, float(self._model.light_dir[0]))
        self._light_dir_y_slider = client.gui.add_slider("Light Dir Y", -1.0, 1.0, 0.01, float(self._model.light_dir[1]))
        self._light_dir_z_slider = client.gui.add_slider("Light Dir Z", -1.0, 1.0, 0.01, float(self._model.light_dir[2]))

        @self._light_dir_x_slider.on_update
        async def _(_):
            self._update = True

        @self._light_dir_y_slider.on_update
        async def _(_):
            self._update = True

        @self._light_dir_z_slider.on_update
        async def _(_):
            self._update = True

        self._light_r_slider = client.gui.add_slider("Light R", 0.0, 1.0, 0.01, float(self._model.light_color[0]))
        self._light_g_slider = client.gui.add_slider("Light G", 0.0, 1.0, 0.01, float(self._model.light_color[1]))
        self._light_b_slider = client.gui.add_slider("Light B", 0.0, 1.0, 0.01, float(self._model.light_color[2]))

        @self._light_r_slider.on_update
        async def _(_):
            self._update = True

        @self._light_g_slider.on_update
        async def _(_):
            self._update = True

        @self._light_b_slider.on_update
        async def _(_):
            self._update = True

        self._show_mesh_ckbx = client.gui.add_checkbox("Mesh Overlay", False)

        @self._show_mesh_ckbx.on_update
        async def _(_):
            self._wireframe_ckbx.disabled = not self._show_mesh_ckbx.value
            with client.atomic():
                self._update_mesh_visibility()

        self._wireframe_ckbx = client.gui.add_checkbox("Wireframe", False, disabled=True)

        @self._wireframe_ckbx.on_update
        async def _(_):
            with client.atomic():
                self._update_mesh_visibility()

        self._reset_camera_btn = client.gui.add_button("Reset Camera")

        @self._reset_camera_btn.on_click
        async def _(_):
            self._set_initial_camera()
            self._update = True

        self._snapshot_btn = client.gui.add_button("Snapshot")

        @self._snapshot_btn.on_click
        async def _(_):
            if self._curr_image is not None:
                os.makedirs(self._output_dir, exist_ok=True)
                file_name = os.path.join(self._output_dir, f"hybrid_{self._render_mode_dropdown.value}.jpg")
                cv2.imwrite(file_name, cv2.cvtColor((self._curr_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                print(f"[INFO] Snapshot saved to {file_name}")

    def _get_camera(self) -> Camera:
        client: ClientHandle = self._client_handle
        fovy = client.camera.fov
        aspect = client.camera.aspect
        resolution = max(self._resolution_slider.value, 384)
        image_width = int(resolution)
        image_height = max(int(round(resolution / max(aspect, 1e-6))), 1)
        fovx = 2 * math.atan(math.tan(fovy / 2) * aspect)
        rotation = qvec2rotmat(np.asarray(client.camera.wxyz))
        position = np.asarray(client.camera.position, dtype=np.float32)
        translation = rotation.T @ -position

        return Camera(
            R=rotation,
            T=translation,
            FoVx=fovx,
            FoVy=fovy,
            image_width=image_width,
            image_height=image_height,
            znear=client.camera.near,
            zfar=client.camera.far,
        )

    def _set_camera(self, camera: Camera):
        client: ClientHandle = self._client_handle
        with client.atomic():
            client.camera.wxyz = rotmat2qvec(camera.R)
            client.camera.position = camera.camera_center.detach().cpu().numpy()
            client.camera.fov = camera.FoVy
            client.camera.near = camera.znear
            client.camera.far = camera.zfar

    def _set_initial_camera(self):
        client: ClientHandle = self._client_handle
        position = np.zeros(3, dtype=np.float32)
        rotation = _look_at_rotation(position, self._scene_center)
        aspect = max(float(client.camera.aspect), 1e-6)
        fovy = float(client.camera.fov)
        fovx = 2 * math.atan(math.tan(fovy / 2) * aspect)
        znear = min(0.01, self._scene_extent * 0.01)
        zfar = max(self._scene_extent * 8.0, 100.0)
        camera = Camera(
            R=rotation,
            T=rotation.T @ -position,
            FoVx=fovx,
            FoVy=fovy,
            image_width=1280,
            image_height=max(int(round(1280 / aspect)), 1),
            znear=znear,
            zfar=zfar,
        )
        self._set_camera(camera)

    def update_image(self):
        camera = self._get_camera().to(self._model.device)
        render_mode = self._render_mode_dropdown.value
        background = self._background_dropdown.value
        light_dir = np.array(
            [
                self._light_dir_x_slider.value,
                self._light_dir_y_slider.value,
                self._light_dir_z_slider.value,
            ],
            dtype=np.float32,
        )
        light_dir = _normalize(light_dir, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        light_color = torch.tensor(
            [self._light_r_slider.value, self._light_g_slider.value, self._light_b_slider.value],
            dtype=torch.float32,
            device=self._model.device,
        )

        start_time = time()
        with torch.no_grad():
            output = self._model.forward(
                camera,
                background=background,
                tri_gamma=self._tri_gamma_slider.value,
                gau_gamma=self._gau_gamma_slider.value,
                back_culling=self._back_culling_ckbx.value,
                sort_level=self._sort_level_slider.value,
                ambient_intensity=self._ambient_slider.value,
                light_color=light_color,
                light_dir=torch.tensor(light_dir, dtype=torch.float32, device=self._model.device),
                light_intensity=self._light_intensity_slider.value,
                rich_info=render_mode != "image",
                debug=False,
            )

        if render_mode == "image":
            image = output["render"].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
        elif render_mode == "depth":
            image = depth_to_image(output["depth"].detach().cpu().numpy()).astype(np.float32) / 255.0
        elif render_mode == "normal":
            image = output["normal"].permute(1, 2, 0) @ camera.world_view_transform[:3, :3].T
            image = normal_to_image(image.detach().cpu().numpy()).clip(0, 1)
        elif render_mode == "alpha":
            image = alpha_to_image(output["alpha_mask"].detach().cpu().numpy()).astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unknown render mode: {render_mode}")

        rendered_time = time()
        print(f"[INFO] Rendered {image.shape} {image.dtype}, time: {rendered_time - start_time:.3f}s")

        client: ClientHandle = self._client_handle
        with client.atomic():
            self._curr_image = image
            client.scene.set_background_image(image, "jpeg", 85)

        sent_time = time()
        print(f"[INFO] Sent image {image.shape} {image.dtype}, time: {sent_time - rendered_time:.3f}s")
        self._update = False

    def need_update(self) -> bool:
        return self._update

    def terminate(self):
        self._terminate = True

    def need_terminate(self) -> bool:
        return self._terminate


class HybridVisViewer:
    def __init__(
        self,
        model: HybridGTModel,
        triangle_sampler: AnimatedTriangle,
        initial_triangle: RawTriangle,
        scene_center: np.ndarray,
        scene_extent: float,
        output_dir: str,
    ):
        self._server = viser.ViserServer()
        self._model = model
        self._triangle_sampler = triangle_sampler
        self._rest_triangle = initial_triangle
        self._current_triangle = initial_triangle
        self._mesh_vertices, self._mesh_faces = _triangle_to_mesh(initial_triangle)
        self._scene_center = scene_center
        self._scene_extent = scene_extent
        self._output_dir = output_dir

        self._client_dict: dict[int, HybridVisClient] = {}
        self._close_viewer = False
        self._animation_enabled = len(self._triangle_sampler.animation_names) > 0
        self._animation_time = 0.0
        self._animation_duration = self._triangle_sampler.getAnimationDuration() if self._animation_enabled else 0.0
        self._last_animation_tick = time()
        self._sync_animation_slider = False
        self._selected_animation_name = REST_POSE_LABEL if self._animation_enabled else None

        self._animation_dropdown = None
        self._animation_time_slider = None
        self._play_animation_ckbx = None
        self._loop_animation_ckbx = None
        self._playback_speed_slider = None

        self._register_gui()

    def _load_triangle_frame(self, triangle: RawTriangle):
        self._current_triangle = triangle
        self._model.load_triangle(triangle)
        self._mesh_vertices, self._mesh_faces = _triangle_to_mesh(triangle)
        for client in self._client_dict.values():
            client.set_triangle(triangle)
            client.update_mesh(triangle, self._mesh_vertices, self._mesh_faces)
            client.request_update()

    def _set_animation_controls_enabled(self, enabled: bool):
        if self._animation_time_slider is not None:
            self._animation_time_slider.disabled = not enabled
        if self._play_animation_ckbx is not None:
            self._play_animation_ckbx.disabled = not enabled
        if self._loop_animation_ckbx is not None:
            self._loop_animation_ckbx.disabled = not enabled
        if self._playback_speed_slider is not None:
            self._playback_speed_slider.disabled = not enabled

    def _set_animation_time(self, animation_time: float, force: bool = False):
        if not self._animation_enabled or self._selected_animation_name == REST_POSE_LABEL:
            return

        animation_time = float(np.clip(animation_time, 0.0, self._animation_duration))
        if not force and abs(animation_time - self._animation_time) < 1e-4:
            return

        self._animation_time = animation_time
        if self._animation_time_slider is not None and abs(self._animation_time_slider.value - animation_time) >= 1e-4:
            self._sync_animation_slider = True
            self._animation_time_slider.value = animation_time
            self._sync_animation_slider = False

        triangle = self._triangle_sampler.sample(animation_time)
        self._load_triangle_frame(triangle)

    def _set_animation_clip(self, animation_name: str):
        self._selected_animation_name = animation_name
        self._last_animation_tick = time()
        if animation_name == REST_POSE_LABEL:
            if self._play_animation_ckbx is not None:
                self._play_animation_ckbx.value = False
            self._set_animation_controls_enabled(False)
            self._animation_time = 0.0
            self._load_triangle_frame(self._rest_triangle)
            return

        self._triangle_sampler.setAnimation(animation_name)
        self._animation_duration = self._triangle_sampler.getAnimationDuration()
        self._set_animation_controls_enabled(True)
        if self._animation_time_slider is not None:
            self._sync_animation_slider = True
            self._animation_time_slider.max = max(self._animation_duration, 1e-3)
            self._animation_time_slider.value = 0.0
            self._sync_animation_slider = False
        self._set_animation_time(0.0, force=True)

    def _advance_animation(self):
        if (
            not self._animation_enabled
            or self._selected_animation_name == REST_POSE_LABEL
            or self._play_animation_ckbx is None
            or not self._play_animation_ckbx.value
        ):
            self._last_animation_tick = time()
            return

        if self._animation_duration <= 0.0:
            return

        now = time()
        delta_time = now - self._last_animation_tick
        self._last_animation_tick = now
        if delta_time <= 0.0:
            return

        playback_speed = self._playback_speed_slider.value if self._playback_speed_slider is not None else 1.0
        animation_time = self._animation_time + delta_time * playback_speed
        if self._loop_animation_ckbx is not None and self._loop_animation_ckbx.value:
            animation_time = math.fmod(animation_time, self._animation_duration)
            if animation_time < 0.0:
                animation_time += self._animation_duration
        elif animation_time >= self._animation_duration:
            animation_time = self._animation_duration
            self._play_animation_ckbx.value = False

        self._set_animation_time(animation_time)

    def _register_gui(self):
        close_viewer_btn = self._server.gui.add_button("Close Viewer")

        @close_viewer_btn.on_click
        async def _(_):
            self._close_viewer = True

        if self._animation_enabled:
            animation_names = [REST_POSE_LABEL] + self._triangle_sampler.animation_names
            self._animation_dropdown = self._server.gui.add_dropdown("Animation", animation_names, REST_POSE_LABEL)

            @self._animation_dropdown.on_update
            async def _(_):
                self._set_animation_clip(self._animation_dropdown.value)

            self._animation_time_slider = self._server.gui.add_slider("Animation Time", 0.0, max(self._animation_duration, 1e-3), 0.001, 0.0)

            @self._animation_time_slider.on_update
            async def _(_):
                if self._sync_animation_slider:
                    return
                self._last_animation_tick = time()
                self._set_animation_time(self._animation_time_slider.value)

            self._play_animation_ckbx = self._server.gui.add_checkbox("Play Animation", False)

            @self._play_animation_ckbx.on_update
            async def _(_):
                self._last_animation_tick = time()

            self._loop_animation_ckbx = self._server.gui.add_checkbox("Loop Animation", True)
            self._playback_speed_slider = self._server.gui.add_slider("Playback Speed", 0.1, 4.0, 0.1, 1.0)
            reset_animation_btn = self._server.gui.add_button("Reset Animation")

            @reset_animation_btn.on_click
            async def _(_):
                self._last_animation_tick = time()
                self._set_animation_time(0.0, force=True)

            self._set_animation_controls_enabled(False)

        @self._server.on_client_connect
        async def _(client: ClientHandle):
            vis_client = HybridVisClient(
                client=client,
                model=self._model,
                mesh_vertices=self._mesh_vertices,
                mesh_faces=self._mesh_faces,
                scene_center=self._scene_center,
                scene_extent=self._scene_extent,
                output_dir=self._output_dir,
            )
            vis_client.set_triangle(self._current_triangle)
            self._client_dict[client.client_id] = vis_client

        @self._server.on_client_disconnect
        async def _(client: ClientHandle):
            if client.client_id in self._client_dict:
                self._client_dict[client.client_id].terminate()

    def _step(self):
        try:
            self._advance_animation()
            for client_id in list(self._client_dict.keys()):
                client = self._client_dict[client_id]
                if client.need_terminate():
                    del self._client_dict[client_id]
                    continue
                if client.need_update():
                    client.update_image()
        except Exception as exc:
            print(f"[ERROR] {exc}")

    def run(self):
        while not self._close_viewer:
            self._step()
        self._server.stop()


def run_hybrid_viewer(ply_path: str, glb_path: str, device: int | None, output_dir: str):
    if not torch.cuda.is_available():
        raise RuntimeError("hybrid_viewer.py requires CUDA.")

    model_device: torch.device | int
    if device is None:
        model_device = torch.device("cuda")
    else:
        model_device = device
    
    triangle_sampler = AnimatedTriangle(glb_path=glb_path)
    triangle_sampler.center(np.array([0.0, 3.5, -0.6], dtype=np.float32))
    triangle_sampler.rotate(axis=np.array([1.0, 0.0, 0.0], dtype=np.float32), degree=90.0)
    triangle_sampler.rotate(axis=np.array([0.0, 0.0, 1.0], dtype=np.float32), degree=180.0)
    initial_triangle = triangle_sampler.getRestTriangle()

    model = HybridGTModel(device=model_device)
    model.load_gaussian(ply_path)
    model.load_triangle(initial_triangle)
    model.eval()

    scene_center, scene_extent = _compute_scene_stats(model)
    viewer = HybridVisViewer(
        model=model,
        triangle_sampler=triangle_sampler,
        initial_triangle=initial_triangle,
        scene_center=scene_center,
        scene_extent=scene_extent,
        output_dir=output_dir,
    )
    viewer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Viser viewer for HybridGTModel")
    parser.add_argument("--ply", type=str, required=True, help="Path to the gaussian .ply file")
    parser.add_argument("--glb", type=str, required=True, help="Path to the mesh .glb/.gltf file")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--output", type=str, default="outputs/hybrid_viewer", help="Directory for saved snapshots")
    args = parser.parse_args()

    run_hybrid_viewer(args.ply, args.glb, args.device, args.output)
