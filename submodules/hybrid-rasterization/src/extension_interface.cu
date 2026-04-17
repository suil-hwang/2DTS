#include "extension_interface.h"

#include <tuple>

#include "rasterizer.h"
#include "backward.h"

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterizeHybridForward(
    const int image_width,
    const int image_height,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &full_projmatrix,
    const torch::Tensor &campos,
    const int tri_sh_degree,
    const int gau_sh_degree,
    const float tri_gamma,
    const float gau_gamma,
    const float ambient_intensity,
    const torch::Tensor &light_color,
    const torch::Tensor &light_dir,
    const float light_intensity,
    const float background_depth,
    const torch::Tensor &background,
    const torch::Tensor &vertex,
    const torch::Tensor &shs,
    const torch::Tensor &feature,
    const torch::Tensor &tri_uv,
    const torch::Tensor &tri_texture,
    const torch::Tensor &opacity,
    const bool back_culling,
    const bool rich_info,
    const int sort_level,
    const torch::Tensor &means3D,
    const torch::Tensor &gaussian_shs,
    const torch::Tensor &gaussian_colors,
    const torch::Tensor &gaussian_opacity,
    const torch::Tensor &gaussian_scales,
    const torch::Tensor &gaussian_rotations,
    const torch::Tensor &gaussian_cov3D_precomp,
    const bool gaussian_prefiltered,
    const bool debug)
{
    const bool use_tri_sh = shs.numel() > 0;
    const bool use_tri_texture = (tri_uv.numel() > 0) && (tri_texture.numel() > 0);
    const bool use_tri_vertex_color = use_tri_sh ? (shs.dim() == 4) : (feature.dim() == 3);

    const bool use_gau_sh = gaussian_shs.numel() > 0;
    const bool use_gau_cov_precomp = gaussian_cov3D_precomp.numel() > 0;

    const float focal_x = image_width / (2.0f * tan_fovx);
    const float focal_y = image_height / (2.0f * tan_fovy);

    auto out = RASTERIZER::forwardHybrid(
        background,
        vertex,
        feature,
        tri_uv,
        tri_texture,
        shs,
        opacity,
        means3D,
        gaussian_colors,
        gaussian_shs,
        gaussian_opacity,
        gaussian_scales,
        gaussian_rotations,
        gaussian_cov3D_precomp,
        viewmatrix,
        projmatrix,
        full_projmatrix,
        campos,
        tan_fovx,
        tan_fovy,
        focal_x,
        focal_y,
        image_height,
        image_width,
        tri_sh_degree,
        gau_sh_degree,
        tri_gamma,
        gau_gamma,
        ambient_intensity,
        light_color,
        light_dir,
        light_intensity,
        background_depth,
        gaussian_prefiltered,
        debug,
        use_tri_sh,
        use_tri_texture,
        use_tri_vertex_color,
        use_gau_sh,
        use_gau_cov_precomp,
        back_culling,
        sort_level,
        rich_info);

    int num_rendered = std::get<0>(out);
    torch::Tensor out_feature = std::get<1>(out);
    torch::Tensor out_depth = std::get<2>(out);
    torch::Tensor out_normal = std::get<3>(out);
    torch::Tensor out_distort = std::get<4>(out);
    torch::Tensor n_contribs = std::get<5>(out);
    torch::Tensor final_Ts = std::get<6>(out);
    torch::Tensor radii = std::get<7>(out);
    torch::Tensor contrib_sum = std::get<8>(out);
    torch::Tensor contrib_max = std::get<9>(out);

    auto buffer_opts = torch::TensorOptions().dtype(torch::kUInt8).device(background.device());
    torch::Tensor geometryBuffer = torch::empty({0}, buffer_opts);
    torch::Tensor binningBuffer = torch::empty({0}, buffer_opts);
    torch::Tensor imageBuffer = torch::empty({0}, buffer_opts);

    return std::make_tuple(
        num_rendered,
        out_feature,
        radii,
        out_depth,
        out_normal,
        out_distort,
        contrib_sum,
        contrib_max,
        n_contribs,
        final_Ts,
        geometryBuffer,
        binningBuffer,
        imageBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterizeHybridBackwardNotImplemented(
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const torch::Tensor &campos,
    const int tri_sh_degree,
    const float gamma,
    const float background_depth,
    const torch::Tensor &background,
    const torch::Tensor &vertex,
    const torch::Tensor &shs,
    const torch::Tensor &feature,
    const torch::Tensor &opacity,
    const int num_rendered,
    const torch::Tensor &radii,
    const torch::Tensor &final_feature,
    const torch::Tensor &final_depth,
    const torch::Tensor &final_normal,
    const torch::Tensor &final_distortion,
    const torch::Tensor &geometryBuffer,
    const torch::Tensor &binningBuffer,
    const torch::Tensor &imageBuffer,
    const torch::Tensor &dL_dout_feature,
    const torch::Tensor &dL_dout_depth,
    const torch::Tensor &dL_dout_normal,
    const torch::Tensor &dL_dout_distortion,
    const bool back_culling,
    const bool rich_info,
    const int sort_level,
    const bool debug)
{
    (void)tan_fovx;
    (void)tan_fovy;
    (void)viewmatrix;
    (void)projmatrix;
    (void)campos;
    (void)tri_sh_degree;
    (void)gamma;
    (void)background_depth;
    (void)background;
    (void)num_rendered;
    (void)radii;
    (void)final_feature;
    (void)final_depth;
    (void)final_normal;
    (void)final_distortion;
    (void)geometryBuffer;
    (void)binningBuffer;
    (void)imageBuffer;
    (void)dL_dout_feature;
    (void)dL_dout_depth;
    (void)dL_dout_normal;
    (void)dL_dout_distortion;
    (void)back_culling;
    (void)rich_info;
    (void)sort_level;
    (void)debug;

    auto out = BACKWARD::notImplemented(
        vertex,
        feature,
        shs,
        opacity,
        torch::empty({0, 3}, vertex.options()),
        torch::empty({0}, feature.options()),
        torch::empty({0}, shs.options()),
        torch::empty({0}, opacity.options()));

    return std::make_tuple(
        std::get<0>(out),
        torch::zeros({vertex.size(0)}, vertex.options()),
        std::get<2>(out),
        std::get<1>(out),
        std::get<3>(out));
}
