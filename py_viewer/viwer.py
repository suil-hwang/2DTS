"""Native viewer: python -m py_viewer.viwer [--model ...]."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time


def parse_args():
    parser = argparse.ArgumentParser(description="ModernGL viewer for 2DTS and ordinary meshes")
    parser.add_argument("--model", type=Path, nargs="+", default=[], help="PLY/CKPT/GLB/GLTF/OBJ files")
    parser.add_argument("--config", type=Path, help="2DTS YAML; auto-detected next to baseline exports")
    parser.add_argument("--cameras", type=Path, help="NeRF transforms JSON or scene directory")
    parser.add_argument("--device", type=int, default=0, help="CUDA device for 2DTS")
    parser.add_argument("--resolution", type=int, default=800, help="Longest rendered image dimension")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--output", type=Path, default=Path("outputs/viewer"))
    parser.add_argument("--headless", action="store_true", help="Hidden window for bounded rendering tests")
    parser.add_argument("--frames", type=int, default=0, help="Exit after N UI frames; default: interactive")
    parser.add_argument("--screenshot", type=Path, help="Save final viewport PNG")
    parser.add_argument("--report", type=Path, help="Save final rendering metadata as JSON")
    args = parser.parse_args()
    if args.resolution < 1 or args.width < 64 or args.height < 64 or args.frames < 0:
        parser.error("Resolution must be positive, window dimensions >= 64, and frames >= 0.")
    return args


# Handle --help and invalid arguments before importing optional graphics libraries.
if __name__ == "__main__":
    args = parse_args()

import glfw  # Select the Python GLFW binding before importing imgui_bundle.
import moderngl
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

from .render import ImageRenderer, MeshRenderer, RenderTarget, TSBackend
from .scene import OrbitCamera, RenderSettings, camera_path, find_config, load_mesh, load_views, model_kind


class ViewerApp:
    def __init__(self, *, config=None, cameras=None, hidden=False, width=1280, height=800,
                 resolution=800, device=0, output=Path("outputs/viewer")):
        views = load_views(Path(cameras)) if cameras is not None else {}
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize the display.")
        glfw.window_hint(glfw.VISIBLE, not hidden)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(width, height, "2DTS Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("An OpenGL 3.3 window could not be created.")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0 if hidden else 1)
        self.ctx = moderngl.create_context(require=330)
        imgui.create_context()
        imgui.get_io().set_ini_filename("")
        self.gui = GlfwRenderer(self.window)
        self.presenter = ImageRenderer(self.ctx)
        self.target = None
        self.renderer = None
        self.kind = None
        self.model_path = None
        self.paths = []
        self.config_path = config
        self.cameras_path = cameras
        self.loaded_cameras_path = cameras
        self.views = views
        self.camera_group = "Free"
        self.camera_index = 0
        self.preset = None
        self.camera = OrbitCamera()
        self.settings = RenderSettings(resolution=resolution)
        self.device = device
        self.output = Path(output)
        self.path_text = ""
        self.error = ""
        self.note = "Open or drop a PLY, CKPT, GLB, GLTF, or OBJ file."
        self.dirty = True
        self.render_count = 0
        self.render_ms = 0.0
        self.drag_button = None
        self.viewport_aspect = 1.0
        self.snapshot_requested = False
        glfw.set_drop_callback(self.window, self._drop_files)

    def open_model(self, path: Path):
        path = path.expanduser().resolve()
        kind = model_kind(path)
        config = self.config_path or find_config(path)
        cameras = self.cameras_path or camera_path(config)
        views = self.views if cameras == self.loaded_cameras_path else load_views(cameras) if cameras else {}
        if kind == "2DTS":
            renderer = TSBackend(path, config, self.device)
            bounds = renderer.bounds
            note = renderer.note
        else:
            parts, bounds = load_mesh(path)
            renderer = MeshRenderer(self.ctx, parts)
            note = "Opaque unlit base-color preview. PBR lighting and material transparency are not applied."
        first_model = self.renderer is None
        if self.renderer is not None:
            self.renderer.close()
        self.renderer, self.kind, self.model_path = renderer, kind, path
        self.bounds = bounds
        self.path_text = str(path)
        if path not in self.paths:
            self.paths.append(path)
        self.settings.mode = "RGB"
        self.settings.wireframe = False
        if kind == "2DTS":
            renderer.configure(self.settings)
        if cameras != self.loaded_cameras_path:
            self.views, self.loaded_cameras_path = views, cameras
            self.camera_group, self.camera_index, self.preset = "Free", 0, None
        if first_model:
            self.camera.fit(bounds)
            group = "test" if self.views.get("test") else next((key for key, value in self.views.items() if value), None)
            if group:
                self.select_camera(group, 0)
        self.note, self.error, self.dirty = note, "", True
        glfw.set_window_title(self.window, f"2DTS Viewer — {path.name} [{kind}]")

    def select_camera(self, group: str, index: int = 0):
        self.camera_group, self.camera_index = group, index
        self.preset = self.views[group][index] if group != "Free" else None
        if self.preset:
            self.camera.set_view(self.preset)
        elif self.settings.mode == "GT":
            self.settings.mode = "RGB"
        self.dirty = True

    def fit_camera(self):
        if self.renderer is not None:
            self.select_camera("Free")
            self.camera.fit(self.bounds)
            self.dirty = True

    def _drop_files(self, window, paths):
        try:
            self.open_model(Path(paths[0]))
        except Exception as error:
            self.error = str(error)

    def render_current(self, aspect=None):
        aspect = self.preset.aspect if self.preset else (aspect or self.viewport_aspect)
        resolution = self.settings.resolution
        size = (resolution, max(1, round(resolution / aspect))) if aspect >= 1 else (max(1, round(resolution * aspect)), resolution)
        if self.target is None or self.target.size != size:
            if self.target:
                self.target.close()
            self.target = RenderTarget(self.ctx, size)
            self.dirty = True
        if not self.dirty:
            return
        start = time.perf_counter()
        if self.settings.mode == "GT" and self.preset is not None:
            self.presenter.render(self.preset.image(float(self.settings.white_background)), self.target)
        elif self.kind == "2DTS":
            self.presenter.render(self.renderer.render(self.camera, size, self.settings), self.target)
        elif self.kind == "Mesh":
            self.renderer.render(self.camera, self.target, self.settings)
        else:
            self.target.fbo.use()
            self.target.fbo.clear(0.08, 0.09, 0.11, 1)
        self.render_ms = (time.perf_counter() - start) * 1000
        self.render_count += 1
        self.dirty = False

    def snapshot(self, path=None):
        self.render_current()
        if path is None:
            name = self.model_path.stem if self.model_path else "view"
            path = self.output / f"{name}_{self.settings.mode.lower()}_{time.time_ns()}.png"
        self.target.save(Path(path))
        self.note = f"Saved {path}"
        return Path(path)

    def _controls(self):
        changed, self.path_text = imgui.input_text("File", self.path_text)
        if imgui.button("Open"):
            try:
                self.open_model(Path(self.path_text))
            except Exception as error:
                self.error = str(error)
        if self.paths:
            current = self.paths.index(self.model_path)
            changed, index = imgui.combo("Model", current, [str(path.name) for path in self.paths])
            if changed:
                try:
                    self.open_model(self.paths[index])
                except Exception as error:
                    self.error = str(error)
        imgui.separator()
        modes = ["RGB", "Depth", "Normal", "Alpha"] if self.kind == "2DTS" else ["RGB", "Normal"]
        if any(self.views.values()):
            modes.append("GT")
        changed, index = imgui.combo("Display", modes.index(self.settings.mode), modes)
        if changed:
            self.settings.mode = modes[index]
            if self.settings.mode == "GT" and self.preset is None:
                group = "test" if self.views.get("test") else next(key for key in self.views if self.views[key])
                self.select_camera(group)
            self.dirty = True
        groups = ["Free"] + [key for key in self.views if self.views[key]]
        changed, index = imgui.combo("Camera", groups.index(self.camera_group), groups)
        if changed:
            self.select_camera(groups[index])
        if self.preset:
            changed, index = imgui.slider_int("View", self.camera_index, 0, len(self.views[self.camera_group]) - 1)
            if changed:
                self.select_camera(self.camera_group, index)
            imgui.text_wrapped(self.preset.name)
        if imgui.button("Fit model (F)"):
            self.fit_camera()
        changed, self.settings.white_background = imgui.checkbox("White background", self.settings.white_background)
        self.dirty |= changed
        changed, self.settings.resolution = imgui.slider_int("Resolution", self.settings.resolution, 256, 4096)
        self.dirty |= changed
        changed, self.settings.cull_backfaces = imgui.checkbox("Backface culling", self.settings.cull_backfaces)
        self.dirty |= changed
        if self.kind == "2DTS":
            changed, self.settings.gamma = imgui.slider_float("Gamma", self.settings.gamma, 0.1, 50.0)
            self.dirty |= changed
            changed, self.settings.sh_degree = imgui.slider_int("SH degree", self.settings.sh_degree, 0, self.renderer.model.max_sh_degree)
            self.dirty |= changed
            changed, self.settings.sort_level = imgui.slider_int("Sort level", self.settings.sort_level, 0, 2)
            self.dirty |= changed
            changed, self.settings.gamma_rescale = imgui.checkbox("Gamma rescale", self.settings.gamma_rescale)
            self.dirty |= changed
            changed, self.settings.threshold_enabled = imgui.checkbox("Opacity threshold", self.settings.threshold_enabled)
            self.dirty |= changed
            if self.settings.threshold_enabled:
                changed, self.settings.threshold = imgui.slider_float("Threshold", self.settings.threshold, 0, 1)
                self.dirty |= changed
            changed, self.settings.supersampling = imgui.slider_int("Samples per axis", self.settings.supersampling, 1, 4)
            self.dirty |= changed
        elif self.kind == "Mesh":
            changed, self.settings.wireframe = imgui.checkbox("Wireframe", self.settings.wireframe)
            self.dirty |= changed
        if imgui.button("Save viewport PNG"):
            self.snapshot_requested = True
        imgui.separator()
        if self.renderer:
            imgui.text(f"{self.kind}: {self.renderer.triangle_count:,} triangles")
        imgui.text(f"Render call: {self.render_ms:.2f} ms")
        imgui.text_wrapped("Left drag: orbit | Right/middle: pan | Wheel: zoom")
        imgui.text_wrapped(self.note)
        if self.error:
            imgui.separator()
            imgui.text_wrapped(f"Could not open: {self.error}")

    def _navigation(self, hovered, height):
        io = imgui.get_io()
        if self.renderer is None or self.settings.mode == "GT":
            return
        started = False
        if hovered:
            for button in (0, 1, 2):
                if imgui.is_mouse_clicked(button):
                    self.drag_button = button
                    started = True
            if io.mouse_wheel:
                self.select_camera("Free")
                self.camera.dolly(io.mouse_wheel)
                self.dirty = True
        if self.drag_button is not None:
            if not imgui.is_mouse_down(self.drag_button):
                self.drag_button = None
            elif not started and (io.mouse_delta.x or io.mouse_delta.y):
                self.select_camera("Free")
                if self.drag_button == 0:
                    self.camera.orbit(io.mouse_delta.x, io.mouse_delta.y)
                else:
                    self.camera.pan(io.mouse_delta.x, io.mouse_delta.y, height)
                self.dirty = True

    def frame(self, process_inputs=True):
        if process_inputs:
            glfw.poll_events()
            self.gui.process_inputs()
        io = imgui.get_io()
        imgui.new_frame()
        width, height = io.display_size
        panel = min(330, width * 0.4)
        flags = imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_collapse
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((panel, height))
        imgui.begin("2DTS Viewer", flags=flags)
        imgui.push_item_width(max(80, panel - 160))
        self._controls()
        imgui.pop_item_width()
        imgui.end()
        imgui.set_next_window_pos((panel, 0))
        imgui.set_next_window_size((width - panel, height))
        imgui.begin("Viewport", flags=flags)
        available = imgui.get_content_region_avail()
        if available.x > 0 and available.y > 0:
            self.viewport_aspect = available.x / available.y
            self.render_current()
            scale = min(available.x / self.target.size[0], available.y / self.target.size[1])
            image_size = (self.target.size[0] * scale, self.target.size[1] * scale)
            start = imgui.get_cursor_pos()
            imgui.set_cursor_pos((start.x + (available.x - image_size[0]) / 2,
                                 start.y + (available.y - image_size[1]) / 2))
            imgui.image(imgui.ImTextureRef(self.target.texture.glo), image_size, (0, 1), (1, 0))
            self._navigation(imgui.is_item_hovered(), image_size[1])
        imgui.end()
        if not io.want_text_input and imgui.is_key_pressed(imgui.Key.f):
            self.fit_camera()
        if self.snapshot_requested:
            self.snapshot()
            self.snapshot_requested = False
        imgui.render()
        self.ctx.screen.use()
        framebuffer_size = glfw.get_framebuffer_size(self.window)
        self.ctx.viewport = (0, 0, *framebuffer_size)
        self.ctx.clear(0.08, 0.09, 0.11, 1)
        self.gui.render(imgui.get_draw_data())
        glfw.swap_buffers(self.window)

    def run(self, frames=0):
        frame = 0
        while not glfw.window_should_close(self.window) and (not frames or frame < frames):
            if all(glfw.get_framebuffer_size(self.window)):
                self.frame()
                frame += 1
            if not frames and not self.dirty:
                glfw.wait_events_timeout(0.1)

    def report(self):
        return {"model": str(self.model_path), "kind": self.kind,
                "triangles": self.renderer.triangle_count if self.renderer else 0,
                "renderer": self.ctx.info["GL_RENDERER"], "gl_version": self.ctx.info["GL_VERSION"],
                "gl_error": self.ctx.error, "settings": asdict(self.settings),
                "camera_group": self.camera_group, "camera_index": self.camera_index,
                "image_size": self.target.size if self.target else None, "render_count": self.render_count}

    def close(self):
        if self.renderer:
            self.renderer.close()
        if self.target:
            self.target.close()
        self.presenter.close()
        self.gui.shutdown()
        imgui.destroy_context()
        self.ctx.release()
        glfw.destroy_window(self.window)
        glfw.terminate()


if __name__ == "__main__":
    app = ViewerApp(config=args.config, cameras=args.cameras, hidden=args.headless,
                    width=args.width, height=args.height, resolution=args.resolution,
                    device=args.device, output=args.output)
    try:
        app.paths = [path.expanduser().resolve() for path in args.model]
        if app.paths:
            app.open_model(app.paths[0])
        app.run(args.frames or (3 if args.headless else 0))
        if args.screenshot:
            app.snapshot(args.screenshot)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(app.report(), indent=2), encoding="utf-8")
    finally:
        app.close()
