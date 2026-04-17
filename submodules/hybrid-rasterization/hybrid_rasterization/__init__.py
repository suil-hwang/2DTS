from typing import NamedTuple, Callable
import torch
import torch.nn as nn

from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def debug_run(func: Callable, *args, debug: bool = False):
    if debug:
        func_name = func.__name__
        cpu_args = cpu_deep_copy_tuple(args)
        try:
            return func(*args)
        except Exception as ex:
            torch.save(cpu_args, f"snapshot_{func_name}.dump")
            print(f"\nAn error occured in {func_name}. Writing snapshot_{func_name}.dump for debugging.")
            raise ex
    return func(*args)


class TriangleRasterizationSettings(NamedTuple):
    image_width: int
    image_height: int
    tanfovx: float
    tanfovy: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    full_projmatrix: torch.Tensor
    campos: torch.Tensor
    tri_sh_degree: int
    gau_sh_degree: int
    tri_gamma: float
    gau_gamma: float
    ambient_intensity: float
    light_color: torch.Tensor
    light_dir: torch.Tensor
    light_intensity: float
    background_depth: float
    background: torch.Tensor
    back_culling: bool
    rich_info: bool
    sort_level: int
    gaussian_prefiltered: bool
    debug: bool


class _RasterizeHybrid(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vertex: torch.Tensor,
        grad_holder: torch.Tensor,
        shs: torch.Tensor,
        feature: torch.Tensor,
        tri_uv: torch.Tensor,
        tri_texture: torch.Tensor,
        opacity: torch.Tensor,
        means3D: torch.Tensor,
        gaussian_shs: torch.Tensor,
        gaussian_colors: torch.Tensor,
        gaussian_opacity: torch.Tensor,
        gaussian_scales: torch.Tensor,
        gaussian_rotations: torch.Tensor,
        gaussian_cov3D_precomp: torch.Tensor,
        raster_settings: TriangleRasterizationSettings,
    ):
        del grad_holder

        args = (
            raster_settings.image_width,
            raster_settings.image_height,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.viewmatrix.contiguous(),
            raster_settings.projmatrix.contiguous(),
            raster_settings.full_projmatrix.contiguous(),
            raster_settings.campos.contiguous(),
            raster_settings.tri_sh_degree,
            raster_settings.gau_sh_degree,
            raster_settings.tri_gamma,
            raster_settings.gau_gamma,
            raster_settings.ambient_intensity,
            raster_settings.light_color.contiguous(),
            raster_settings.light_dir.contiguous(),
            raster_settings.light_intensity,
            raster_settings.background_depth,
            raster_settings.background.contiguous(),
            vertex,
            shs,
            feature,
            tri_uv,
            tri_texture,
            opacity,
            raster_settings.back_culling,
            raster_settings.rich_info,
            raster_settings.sort_level,
            means3D,
            gaussian_shs,
            gaussian_colors,
            gaussian_opacity,
            gaussian_scales,
            gaussian_rotations,
            gaussian_cov3D_precomp,
            raster_settings.gaussian_prefiltered,
            raster_settings.debug,
        )

        (
            num_rendered,
            out_feature,
            radii,
            depth,
            normal,
            distortion,
            contrib_sum,
            contrib_max,
            n_contribs,
            final_Ts,
            geometryBuffer,
            binningBuffer,
            imageBuffer,
        ) = debug_run(_C.rasterize_hybrid, *args, debug=raster_settings.debug)

        ctx.save_for_backward(geometryBuffer, binningBuffer, imageBuffer)
        ctx.raster_settings = raster_settings

        alpha_mask = 1 - final_Ts
        if raster_settings.rich_info:
            return out_feature, radii, depth, normal, distortion, contrib_sum, contrib_max, n_contribs, alpha_mask, num_rendered
        return out_feature, radii

    @staticmethod
    def backward(ctx, *grads_out):
        raise RuntimeError("Backward is not implemented yet for hybrid rasterization")


class TriangleRasterizer(nn.Module):
    def __init__(self, raster_settings: TriangleRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(
        self,
        vertex: torch.Tensor,
        grad_holder: torch.Tensor,
        opacity: torch.Tensor,
        shs: torch.Tensor = None,
        feature: torch.Tensor = None,
        tri_uv: torch.Tensor = None,
        tri_texture: torch.Tensor = None,
        means3D: torch.Tensor = None,
        gaussian_opacity: torch.Tensor = None,
        gaussian_shs: torch.Tensor = None,
        gaussian_colors: torch.Tensor = None,
        gaussian_scales: torch.Tensor = None,
        gaussian_rotations: torch.Tensor = None,
        gaussian_cov3D_precomp: torch.Tensor = None,
    ):
        has_shs = shs is not None and shs.numel() > 0
        has_feature = feature is not None and feature.numel() > 0
        has_texture = tri_uv is not None and tri_uv.numel() > 0 and tri_texture is not None and tri_texture.numel() > 0
        if int(has_shs) + int(has_feature) + int(has_texture) != 1:
            raise Exception("Please provide exactly one of SHs, feature, or texture (tri_uv + tri_texture)!")

        has_gaussians = means3D is not None and means3D.numel() > 0
        if has_gaussians:
            has_gaussian_shs = gaussian_shs is not None and gaussian_shs.numel() > 0
            has_gaussian_colors = gaussian_colors is not None and gaussian_colors.numel() > 0
            if has_gaussian_shs == has_gaussian_colors:
                raise Exception("For gaussians, provide exactly one of gaussian_shs or gaussian_colors.")

            has_cov = gaussian_cov3D_precomp is not None and gaussian_cov3D_precomp.numel() > 0
            has_scales = gaussian_scales is not None and gaussian_scales.numel() > 0
            has_rots = gaussian_rotations is not None and gaussian_rotations.numel() > 0
            if not has_cov and not (has_scales and has_rots):
                raise Exception("For gaussians, provide gaussian_cov3D_precomp or gaussian_scales+gaussian_rotations.")

            if gaussian_opacity is None or gaussian_opacity.numel() == 0:
                raise Exception("gaussian_opacity is required when means3D is provided.")

        device = vertex.device
        dtype = vertex.dtype

        shs = torch.empty((0,), device=device, dtype=dtype) if shs is None else shs
        feature = torch.empty((0,), device=device, dtype=dtype) if feature is None else feature
        tri_uv = torch.empty((0, 3, 2), device=device, dtype=dtype) if tri_uv is None else tri_uv
        tri_texture = torch.empty((0, 0, 3), device=device, dtype=dtype) if tri_texture is None else tri_texture

        means3D = torch.empty((0, 3), device=device, dtype=dtype) if means3D is None else means3D
        gaussian_opacity = torch.empty((0,), device=device, dtype=dtype) if gaussian_opacity is None else gaussian_opacity
        gaussian_shs = torch.empty((0,), device=device, dtype=dtype) if gaussian_shs is None else gaussian_shs
        gaussian_colors = torch.empty((0,), device=device, dtype=dtype) if gaussian_colors is None else gaussian_colors
        gaussian_scales = torch.empty((0, 3), device=device, dtype=dtype) if gaussian_scales is None else gaussian_scales
        gaussian_rotations = torch.empty((0, 4), device=device, dtype=dtype) if gaussian_rotations is None else gaussian_rotations
        gaussian_cov3D_precomp = (
            torch.empty((0, 6), device=device, dtype=dtype) if gaussian_cov3D_precomp is None else gaussian_cov3D_precomp
        )

        return _RasterizeHybrid.apply(
            vertex,
            grad_holder,
            shs,
            feature,
            tri_uv,
            tri_texture,
            opacity,
            means3D,
            gaussian_shs,
            gaussian_colors,
            gaussian_opacity,
            gaussian_scales,
            gaussian_rotations,
            gaussian_cov3D_precomp,
            self.raster_settings,
        )
