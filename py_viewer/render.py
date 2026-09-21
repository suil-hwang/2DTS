"""OpenGL resources have one owner and are released before the context closes."""
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from .scene import RenderSettings


SHADER_DIR = Path(__file__).resolve().parent / "shaders"


class RenderTarget:
    def __init__(self, ctx, size):
        self.ctx = ctx
        self.size = size
        self.texture = ctx.texture(size, 4)
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth = ctx.depth_renderbuffer(size)
        self.fbo = ctx.framebuffer([self.texture], self.depth)

    def read(self):
        data = self.fbo.read(components=3, alignment=1)
        return np.frombuffer(data, np.uint8).reshape(self.size[1], self.size[0], 3)[::-1].copy()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.read()).save(path)

    def close(self):
        self.fbo.release()
        self.depth.release()
        self.texture.release()


class ImageRenderer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.texture = None
        self.program = ctx.program(
            vertex_shader=(SHADER_DIR / "image.vert").read_text(encoding="utf-8"),
            fragment_shader=(SHADER_DIR / "image.frag").read_text(encoding="utf-8"),
        )
        self.vao = ctx.vertex_array(self.program, [])

    def render(self, image: np.ndarray, target: RenderTarget):
        image = np.ascontiguousarray(image, dtype=np.float32)
        size = (image.shape[1], image.shape[0])
        if self.texture is None or self.texture.size != size:
            if self.texture is not None:
                self.texture.release()
            self.texture = self.ctx.texture(size, 3, dtype="f4")
            self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.texture.write(image.tobytes(), alignment=1)
        target.fbo.use()
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.CULL_FACE)
        self.ctx.wireframe = False
        self.texture.use(0)
        self.program["image"].value = 0
        self.vao.render(vertices=3)

    def close(self):
        if self.texture is not None:
            self.texture.release()
        self.vao.release()
        self.program.release()


class MeshRenderer:
    def __init__(self, ctx, parts):
        self.ctx = ctx
        self.resources = []
        self.triangle_count = sum(len(part.faces) for part in parts)
        self.program = ctx.program(
            vertex_shader=(SHADER_DIR / "mesh.vert").read_text(encoding="utf-8"),
            fragment_shader=(SHADER_DIR / "mesh.frag").read_text(encoding="utf-8"),
        )
        for part in parts:
            triangles = part.vertices[part.faces].astype(np.float32)
            normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            lengths = np.linalg.norm(normal, axis=1, keepdims=True)
            normal = np.divide(normal, lengths, out=np.zeros_like(normal), where=lengths > 0)
            vertices = np.concatenate([triangles, part.colors,
                                       np.repeat(normal[:, None, :], 3, axis=1), part.uv[part.faces]], axis=-1)
            buffer = ctx.buffer(vertices.astype(np.float32).tobytes())
            vao = ctx.vertex_array(self.program, [(buffer, "3f 3f 3f 2f", "position", "vertex_color", "normal", "texcoord")])
            rgba = np.ascontiguousarray(part.texture[::-1])
            texture = ctx.texture((rgba.shape[1], rgba.shape[0]), 4, rgba.tobytes(), alignment=1)
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.resources.append((vao, buffer, texture))

    def render(self, camera, target: RenderTarget, settings):
        target.fbo.use()
        background = float(settings.white_background)
        target.fbo.clear(background, background, background, 1)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        (self.ctx.enable if settings.cull_backfaces else self.ctx.disable)(moderngl.CULL_FACE)
        self.ctx.wireframe = settings.wireframe
        self.program["mvp"].write(camera.matrix(target.size[0] / target.size[1]).T.astype("f4").tobytes())
        self.program["mode"].value = int(settings.mode == "Normal")
        self.program["wire"].value = settings.wireframe
        self.program["base_color"].value = 0
        for vao, _, texture in self.resources:
            texture.use(0)
            vao.render()
        self.ctx.wireframe = False

    def close(self):
        for vao, buffer, texture in self.resources:
            vao.release()
            buffer.release()
            texture.release()
        self.program.release()


class TSBackend:
    def __init__(self, path: Path, config_path: Path | None, device: int = 0):
        import torch
        from src.diff_recon import Config, RawTriangle, TSModel, loadConfig
        from src.diff_recon.utils.scheduler import exponential_scheduler, exponential_step_scheduler, linear_scheduler

        config = loadConfig(str(config_path)).model if config_path else Config()
        updates = config.model_update
        config.optimizer = None
        config.model_update = None
        if path.suffix.lower() == ".ckpt":
            if config_path is None:
                raise ValueError("A 2DTS checkpoint requires its YAML config (--config).")
            self.model = TSModel(config, device=device).load_ckpt(str(path), load_optimizer=False)
            self.note = "Checkpoint runtime state restored; view-specific color affine is disabled."
        else:
            raw = RawTriangle(ply_path=str(path))
            config.max_sh_degree = int(np.sqrt(raw.shs.shape[-1] // 3)) - 1
            config.use_vertex_color = raw.shs.ndim == 3
            self.model = TSModel(config, device=device).fromRawTriangle(raw)
            self.model.active_sh_degree = self.model.max_sh_degree
            iteration = int(path.stem) if path.stem.isdecimal() else None
            if updates is not None and iteration is not None:
                schedule = updates.gamma_schedule
                if schedule is not None:
                    kwargs = dict(v_init=schedule.gamma_init, v_final=schedule.gamma_final,
                                  max_steps=schedule.end_iter - schedule.start_iter)
                    function = exponential_scheduler
                    if schedule.step_scheduler:
                        function = exponential_step_scheduler
                        kwargs["n_stage"] = schedule.n_stage
                    self.model.gamma = float(function(**kwargs)(iteration - schedule.start_iter))
                schedule = updates.opacity_scheduler
                if schedule is not None:
                    self.model.opacity_floor = float(linear_scheduler(
                        schedule.opacity_init, schedule.opacity_final, schedule.end_iter - schedule.start_iter
                    )(iteration - schedule.start_iter))
                if updates.sh_schedule is not None:
                    self.model.active_sh_degree = min(self.model.max_sh_degree,
                        sum(iteration > step for step in updates.sh_schedule.one_up_iters))
            self.note = "Exported PLY; runtime settings inferred from config/filename. Check settings for other exports."
        self.model.eval()
        self.model.requires_grad_(False)
        vertices = self.model.get_vertex.detach()
        if vertices.numel():
            self.bounds = torch.stack([vertices.amin(dim=(0, 1)), vertices.amax(dim=(0, 1))]).cpu().numpy()
        else:
            self.bounds = np.array([[-1, -1, -1], [1, 1, 1]])
        self.triangle_count = len(vertices)

    def configure(self, settings: RenderSettings):
        model = self.model
        settings.gamma = model.gamma
        settings.sh_degree = model.active_sh_degree
        settings.gamma_rescale = model.gamma_rescale
        settings.sort_level = model.sort_level
        settings.supersampling = model.render_spp or 1
        settings.cull_backfaces = model.back_culling
        settings.threshold_enabled = model.ste_threshold is not None
        if settings.threshold_enabled:
            settings.threshold = model.ste_threshold

    def render(self, camera, size: tuple[int, int], settings: RenderSettings) -> np.ndarray:
        import torch

        with torch.inference_mode():
            if self.triangle_count == 0:
                value = float(settings.white_background) if settings.mode == "RGB" else 0.5 if settings.mode == "Normal" else 0.0
                return np.full((size[1], size[0], 3), value, dtype=np.float32)
            model = self.model
            model.gamma_rescale = settings.gamma_rescale
            model.render_spp = settings.supersampling
            model.ste_threshold = settings.threshold if settings.threshold_enabled else None
            cam = camera.ts_camera(*size, model.device)
            output = model.forward(cam, "white" if settings.white_background else "black",
                                   is_training=settings.mode != "RGB", color_affine=False,
                                   back_culling=settings.cull_backfaces, gamma=settings.gamma,
                                   sh_degree=settings.sh_degree, sort_level=settings.sort_level)
            if settings.mode == "RGB":
                image = output["render"].permute(1, 2, 0)
            elif settings.mode == "Alpha":
                image = output["alpha_mask"][..., None].expand(-1, -1, 3)
            elif settings.mode == "Normal":
                normal = output["normal"].permute(1, 2, 0) @ cam.world_view_transform[:3, :3].T
                image = (normal + 1) * 0.5
            else:
                depth = output["depth"]
                image = ((depth - depth.min()) / (depth.max() - depth.min()).clamp_min(1e-8))[..., None].expand(-1, -1, 3)
            return image.clamp(0, 1).contiguous().cpu().numpy()

    def close(self):
        # Dropping the model releases its tensors; the CUDA allocator may retain a reusable pool.
        self.model = None
