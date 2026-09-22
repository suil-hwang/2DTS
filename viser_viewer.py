from time import time, sleep
from pathlib import Path
import asyncio
import math
import numpy as np
import torch
import viser
from viser import ClientHandle
import trimesh
import os
import argparse
import cv2

from src.diff_recon import loadConfig, TSModel, BaseDatasetFactory
from src.diff_recon.datasets.Colmap_dataset import ColmapDatasetFactory
from src.diff_recon.datasets.NerfSynthetic_dataset import NerfSyntheticDatasetFactory
from src.diff_recon.datasets.MatrixCity_dataset import MatrixCityDatasetFactory
from src.diff_recon.utils.camera import Camera, rotmat2qvec, qvec2rotmat
from src.diff_recon.utils.vis_utils import depth_to_image, normal_to_image, alpha_to_image


class VisClient:
    def __init__(self, client: ClientHandle, dataset: BaseDatasetFactory, model_dict: dict, output_dir: str = "outputs"):
        self._client_handle: ClientHandle = client
        self._dataset = dataset
        self._model_dict = model_dict
        self._mesh_dict = {}
        self._curr_image = None
        self._aspect = None
        self._output_dir = output_dir

        client.scene.enable_default_lights(True)
        self._update = True
        self._terminate = False

        self._register_gui()
        if self._dataset.getTestDatasetSize():
            self._set_test_camera()
        elif self._dataset.getTrainDatasetSize():
            self._set_train_camera()
        self._select_model()

    def _display_mesh(self, name: str = None, mesh: trimesh.Trimesh | trimesh.Scene = None, refresh: bool = False):
        if name is not None:
            if name not in self._mesh_dict or refresh:
                if name.endswith("_simple"):
                    geometry = mesh.to_mesh() if isinstance(mesh, trimesh.Scene) else mesh
                    self._mesh_dict[name] = self._client_handle.scene.add_mesh_simple(
                        name, geometry.vertices, geometry.faces, color=(128, 128, 128), wireframe=True, visible=False
                    )
                elif isinstance(mesh, trimesh.Scene):
                    self._mesh_dict[name] = self._client_handle.scene.add_glb(name, mesh.export(file_type="glb"), visible=False)
                else:
                    self._mesh_dict[name] = self._client_handle.scene.add_mesh_trimesh(name, mesh, visible=False)
        for k, v in self._mesh_dict.items():
            v.visible = k == name

    def _is_mesh(self):
        return isinstance(self._model_dict[self._model_dropdown.value], (trimesh.Trimesh, trimesh.Scene))

    def _select_model(self):
        client = self._client_handle
        model = self._model_dict[self._model_dropdown.value]
        is_mesh = self._is_mesh()
        with client.atomic():
            self._curr_image = None
            self._render_mode_dropdown.disabled = is_mesh
            self._simple_mesh_ckbx.disabled = not is_mesh
            for control in (self._back_culling_ckbx, self._sh_degree_slider, self._gamma_slider, self._ste_slider, self._sort_level_slider):
                control.disabled = is_mesh
            if is_mesh:
                self._render_mode_dropdown.value = "image"
                name = self._model_dropdown.value
                self._display_mesh(f"{name}_simple" if self._simple_mesh_ckbx.value else name, model)
            else:
                self._display_mesh()
                self._back_culling_ckbx.value = model.back_culling
                self._sh_degree_slider.max = max(1, model.max_sh_degree)
                self._sh_degree_slider.value = model.active_sh_degree
                self._gamma_slider.max = max(50.0, float(model.gamma))
                self._gamma_slider.value = float(model.gamma)
                self._ste_slider.value = float(model.ste_threshold or 0.0)
                self._sort_level_slider.value = model.sort_level
        self._update = True

    def _register_gui(self):
        client: ClientHandle = self._client_handle

        @client.camera.on_update
        async def _(_):
            if self._render_mode_dropdown.value != "image_gt" or self._aspect != client.camera.aspect:
                self._update = True

        test_count = self._dataset.getTestDatasetSize()
        train_count = self._dataset.getTrainDatasetSize()
        self._test_view_slider = client.gui.add_slider("Test View", 0, max(1, test_count - 1), 1, 0, disabled=test_count < 2)

        @self._test_view_slider.on_update
        async def _(_):
            if self._test_view_ckbx.value:
                self._set_test_camera()
                self._update = True

        self._test_view_ckbx = client.gui.add_checkbox("Test View", False, disabled=test_count == 0)

        @self._test_view_ckbx.on_update
        async def _(_):
            if self._test_view_ckbx.value:
                self._train_view_ckbx.value = False
                self._set_test_camera()
                self._update = True
        
        self._train_view_slider = client.gui.add_slider("Train View", 0, max(1, train_count - 1), 1, 0, disabled=train_count < 2)
        @self._train_view_slider.on_update
        async def _(_):
            if self._train_view_ckbx.value:
                self._set_train_camera()
                self._update = True
        
        self._train_view_ckbx = client.gui.add_checkbox("Train View", False, disabled=train_count == 0)
        @self._train_view_ckbx.on_update
        async def _(_):
            if self._train_view_ckbx.value:
                self._test_view_ckbx.value = False
                self._set_train_camera()
                self._update = True

        render_modes = ["image", "image_gt", "depth", "normal", "alpha"] if test_count or train_count else ["image", "depth", "normal", "alpha"]
        self._render_mode_dropdown = client.gui.add_dropdown("Render Mode", render_modes, initial_value="image")

        @self._render_mode_dropdown.on_update
        async def _(_):
            if self._render_mode_dropdown.value == "image_gt":
                if not (self._test_view_ckbx.value or self._train_view_ckbx.value):
                    if test_count:
                        self._test_view_ckbx.value = True
                    else:
                        self._train_view_ckbx.value = True
                if self._test_view_ckbx.value:
                    self._set_test_camera()
                    self._train_view_ckbx.value = False
                elif self._train_view_ckbx.value:
                    self._set_train_camera()
                    self._test_view_ckbx.value = False
                self._test_view_ckbx.disabled = True
                self._train_view_ckbx.disabled = True
            else:
                self._test_view_ckbx.disabled = test_count == 0
                self._train_view_ckbx.disabled = train_count == 0
            self._update = True

        self._background_dropdown = client.gui.add_dropdown("Background", ["black", "white"], initial_value="black")

        @self._background_dropdown.on_update
        async def _(_):
            self._update = True

        self._resolution_slider = client.gui.add_slider("Resolution", 384, 4096, 1, 1024)

        @self._resolution_slider.on_update
        async def _(_):
            self._update = True

        self._back_culling_ckbx = client.gui.add_checkbox("Back Culling", False)

        @self._back_culling_ckbx.on_update
        async def _(_):
            self._update = True

        self._sh_degree_slider = client.gui.add_slider("SH Degree", 0, 3, 1, 3)

        @self._sh_degree_slider.on_update
        async def _(_):
            self._update = True

        self._gamma_slider = client.gui.add_slider("Gamma", 1.0, 50.0, 0.1, 1.0)

        @self._gamma_slider.on_update
        async def _(_):
            self._update = True

        self._ste_slider = client.gui.add_slider("STE Threshold", 0.0, 1.0, 0.01, 0.0, hint="0 uses the model's default threshold.")

        @self._ste_slider.on_update
        async def _(_):
            self._update = True
        
        self._sort_level_slider = client.gui.add_slider("Sort Level", 0, 2, 1, 0)

        @self._sort_level_slider.on_update
        async def _(_):
            self._update = True

        key_list = list(self._model_dict.keys())
        self._model_dropdown = client.gui.add_dropdown("Model", key_list, initial_value=key_list[0])

        @self._model_dropdown.on_update
        async def _(_):
            self._select_model()

        self._simple_mesh_ckbx = client.gui.add_checkbox("Wireframe", False, disabled=True)

        @self._simple_mesh_ckbx.on_update
        async def _(_):
            with client.atomic():
                name: str = self._model_dropdown.value
                if self._is_mesh():
                    self._display_mesh(f"{name}_simple" if self._simple_mesh_ckbx.value else name, self._model_dict[name])

        self._snapshot_btn = client.gui.add_button("Snapshot")

        @self._snapshot_btn.on_click
        async def _(_):
            name = self._model_dropdown.value
            render_mode = self._render_mode_dropdown.value
            image = self._curr_image
            if self._is_mesh():
                width = int(self._resolution_slider.value)
                height = max(1, round(width / client.camera.aspect))
                image = await asyncio.to_thread(client.get_render, height, width, transport_format="png", timeout=15.0)
            if image is not None:
                Path(self._output_dir).mkdir(parents=True, exist_ok=True)
                file_name = os.path.join(self._output_dir, f"{Path(name).name}_{render_mode}.jpg")
                if image.dtype != np.uint8:
                    image = (image.clip(0, 1) * 255).round().astype(np.uint8)
                cv2.imwrite(file_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                print(f"[INFO] Snapshot saved to {file_name}")

    def _get_camera(self) -> Camera:
        client: ClientHandle = self._client_handle
        fovy = client.camera.fov
        aspect = client.camera.aspect
        resolution = max(self._resolution_slider.value, 384)

        fovx = 2 * math.atan(math.tan(fovy / 2) * aspect)
        qvec = client.camera.wxyz  # camera-to-world rotation, OpenCV convention
        tvec = client.camera.position
        R = qvec2rotmat(qvec)
        T = R.T @ -tvec

        camera = Camera(
            R=R,
            T=T,
            FoVx=fovx,
            FoVy=fovy,
            image_width=int(resolution),
            image_height=max(1, int(resolution / aspect)),
            znear=client.camera.near,
            zfar=client.camera.far,
        )
        return camera

    def _set_camera(self, camera: Camera):
        client: ClientHandle = self._client_handle
        with client.atomic():
            client.camera.wxyz = rotmat2qvec(camera.R)
            client.camera.position = camera.camera_center.detach().cpu().numpy()
            client.camera.fov = camera.FoVy
            client.camera.far = camera.zfar
            client.camera.near = camera.znear

    def _set_test_camera(self):
        self._set_camera(self._dataset.getTestData(self._test_view_slider.value))
    
    def _set_train_camera(self):
        self._set_camera(self._dataset.getTrainData(self._train_view_slider.value))

    @torch.no_grad()
    def update_image(self):
        # Consume this request before rendering so newer camera/UI events survive.
        self._update = False
        self._aspect = self._client_handle.camera.aspect
        if self._is_mesh():
            color = 255 if self._background_dropdown.value == "white" else 0
            self._client_handle.scene.set_background_image(np.full((1, 1, 3), color, dtype=np.uint8), format="jpeg", jpeg_quality=85)
            return

        model: TSModel = self._model_dict[self._model_dropdown.value]

        back_culling = self._back_culling_ckbx.value
        sh_degree = self._sh_degree_slider.value
        gamma = self._gamma_slider.value
        ste_threshold = self._ste_slider.value
        if ste_threshold == 0:
            ste_threshold = None
        sort_level = self._sort_level_slider.value
        render_mode = self._render_mode_dropdown.value
        background = self._background_dropdown.value

        start_time = time()
        if render_mode == "image_gt":
            if self._test_view_ckbx.value:
                image = self._dataset.getTestData(self._test_view_slider.value).gt_image.permute(1, 2, 0).detach().cpu().numpy()
            elif self._train_view_ckbx.value:
                image = self._dataset.getTrainData(self._train_view_slider.value).gt_image.permute(1, 2, 0).detach().cpu().numpy()
            else:
                raise ValueError("Test View or Train View must be selected for ground truth image")
            # Keep vertical FOV and pixel proportions when the viewport is not square.
            width = max(1, round(image.shape[0] * self._aspect))
            padding = width - image.shape[1]
            if padding > 0:
                image = np.pad(image, ((0, 0), (padding // 2, padding - padding // 2), (0, 0)), constant_values=float(background == "white"))
            elif padding < 0:
                left = -padding // 2
                image = image[:, left:left + width]
        else:
            camera = self._get_camera().to(model.device)
            output = model.forward(
                camera,
                background,
                is_training=render_mode != "image",
                color_affine=False,
                back_culling=back_culling,
                sh_degree=sh_degree,
                gamma=gamma,
                ste_threshold=ste_threshold,
                sort_level=sort_level,
            )
            if render_mode == "image":
                image = output["render"].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
            elif render_mode == "depth":
                image = output["depth"].detach().cpu().numpy()
                print(f"depth min: {np.min(image)}, max: {np.max(image)}, mean: {np.mean(image)}, std: {np.std(image)}")
                image = depth_to_image(image)
            elif render_mode == "normal":
                image = output["normal"].permute(1, 2, 0) @ camera.world_view_transform[:3, :3].T  # Project normal to world space
                image = image.detach().cpu().numpy()
                print(f"normal min: {np.min(image)}, max: {np.max(image)}, mean: {np.mean(image)}, std: {np.std(image)}")
                image = normal_to_image(image)
            elif render_mode == "alpha":
                image = output["alpha_mask"].detach().cpu().numpy()
                print(f"alpha min: {np.min(image)}, max: {np.max(image)}, mean: {np.mean(image)}, std: {np.std(image)}")
                image = alpha_to_image(image)
            else:
                raise ValueError(f"Unknown render mode: {render_mode}")
        rendered_time = time()
        print(f"[INFO] Rendered {image.shape} {image.dtype}, time: {rendered_time-start_time:.3f}s")

        client: ClientHandle = self._client_handle
        with client.atomic():
            self._curr_image = image
            client.scene.set_background_image(image, format="jpeg", jpeg_quality=85)
        sent_time = time()
        print(f"[INFO] Sent Image {image.shape} {image.dtype}, time: {sent_time-rendered_time:.3f}s")

    def need_update(self):
        return self._update

    def terminate(self):
        self._terminate = True

    def need_terminate(self):
        return self._terminate


class VisViewer:
    def __init__(self, dataset: BaseDatasetFactory, model_dict, port: int = 8080, output_dir: str = "outputs"):
        if not model_dict:
            raise ValueError("No models found to display")
        self._server = viser.ViserServer(port=port)
        self._dataset = dataset
        self._model_dict = model_dict
        self._output_dir = output_dir

        self._client_dict: dict[int, VisClient] = {}
        self._close_viewer = False

        self._register_gui()

    def _register_gui(self):
        close_viewer_btn = self._server.gui.add_button("Close Viewer")

        @close_viewer_btn.on_click
        async def _(_):
            self._close_viewer = True

        @self._server.on_client_connect
        async def _(client: ClientHandle):
            self._client_dict[client.client_id] = VisClient(client, self._dataset, self._model_dict, output_dir=self._output_dir)

        @self._server.on_client_disconnect
        async def _(client: ClientHandle):
            connection = self._client_dict.get(client.client_id)
            if connection is not None:
                connection.terminate()

    def _step(self):
        for client_id, client in list(self._client_dict.items()):
            try:
                if client.need_terminate():
                    del self._client_dict[client_id]
                    continue
                if client.need_update():
                    client.update_image()
            except Exception as e:
                print(f"[ERROR] Client {client_id}: {e}")

    def run(self):
        try:
            while not self._close_viewer:
                self._step()
                sleep(0.01)
        finally:
            self._server.stop()


def load_mesh(file_name: str) -> trimesh.Trimesh | trimesh.Scene:
    file_type = Path(file_name).suffix.lower().lstrip(".")
    if file_type == "ply":
        return trimesh.load_mesh(file_name, file_type=file_type, process=False)
    if file_type not in ("glb", "gltf", "obj"):
        raise ValueError(f"Unknown file type: {file_type}")
    mesh = trimesh.load_scene(file_name, file_type=file_type, process=False)
    if file_type == "obj":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    return mesh


def run_VisViewer(config_path: str, dataset_path: str, scene_id: str, output_dir: str = None, port: int = 8080):
    config = loadConfig(config_path)
    if dataset_path:
        config.dataset.local_dir = dataset_path
    config.dataset.scene_id = scene_id
    config.dataset.train_target_res = 1
    config.dataset.test_target_res = 1
    config.dataset.background = config.dataset.test_background
    config.dataset.prefetch = False
    dataset_class = {"Colmap": ColmapDatasetFactory, "NerfSynthetic": NerfSyntheticDatasetFactory, "MatrixCity": MatrixCityDatasetFactory}
    dataset = dataset_class[config.dataset.type](config.dataset)
    result_dir = Path(output_dir) if output_dir is not None else Path(config.trainer.output_dir or ".output") / scene_id

    model_dict = {}
    for checkpoint in sorted((result_dir / "ckpt").glob("*.ckpt")):
        model_dict[checkpoint.stem] = TSModel(config.model).load_ckpt(str(checkpoint), load_optimizer=False)
    for ply_file in sorted((result_dir / "point_cloud").glob("*.ply")):
        if ply_file.stem not in model_dict:
            model = TSModel(config.model)
            model.setup_color_affine(dataset.getTrainDatasetSize())
            model.loadPLY(str(ply_file))
            model.active_sh_degree = model.max_sh_degree
            if ply_file.stem.isdecimal():
                model.state_update(int(ply_file.stem))
            model_dict[ply_file.stem] = model
    for glb_file in sorted((result_dir / "glb").glob("*.glb")):
        model_dict[f"{glb_file.stem}_mesh"] = load_mesh(str(glb_file))

    viewer = VisViewer(dataset, model_dict, port=port, output_dir=str(result_dir))
    viewer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Viser Viewer")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--dataset", type=str, help="Path to the dataset", default="")
    parser.add_argument("--scene", type=str, required=True, help="Scene ID")
    parser.add_argument("--output-dir", type=str, help="Scene results directory containing ckpt, point_cloud, and glb")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    run_VisViewer(args.config, args.dataset, args.scene, output_dir=args.output_dir, port=args.port)
