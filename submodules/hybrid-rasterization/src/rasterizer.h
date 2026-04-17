#pragma once

#include <torch/extension.h>
#include <tuple>

namespace RASTERIZER
{
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forwardHybrid(
    const torch::Tensor &background,
    const torch::Tensor &tri_vertex,
    const torch::Tensor &tri_feature,
    const torch::Tensor &tri_uv,
    const torch::Tensor &tri_texture,
    const torch::Tensor &tri_sh,
    const torch::Tensor &tri_opacity,
    const torch::Tensor &gau_means3D,
    const torch::Tensor &gau_feature,
    const torch::Tensor &gau_sh,
    const torch::Tensor &gau_opacity,
    const torch::Tensor &gau_scales,
    const torch::Tensor &gau_rotations,
    const torch::Tensor &gau_cov3D_precomp,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &full_projmatrix,
    const torch::Tensor &campos,
    float tan_fovx,
    float tan_fovy,
    float focal_x,
    float focal_y,
    int image_height,
    int image_width,
    int tri_sh_degree,
    int gau_sh_degree,
    float tri_gamma,
    float gau_gamma,
    float ambient_intensity,
    const torch::Tensor &light_color,
    const torch::Tensor &light_dir,
    float light_intensity,
    float background_depth,
    bool prefiltered,
    bool debug,
    bool use_tri_sh,
    bool use_tri_texture,
    bool use_tri_vertex_color,
    bool use_gau_sh,
    bool use_gau_cov_precomp,
    bool back_culling,
    int sort_level,
    bool rich_info);
}
