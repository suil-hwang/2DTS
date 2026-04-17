#include "backward.h"

namespace BACKWARD
{
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
notImplemented(
    const torch::Tensor &tri_vertex,
    const torch::Tensor &tri_feature,
    const torch::Tensor &tri_sh,
    const torch::Tensor &tri_opacity,
    const torch::Tensor &gau_means3D,
    const torch::Tensor &gau_feature,
    const torch::Tensor &gau_sh,
    const torch::Tensor &gau_opacity)
{
    auto dL_dtri_vertex = torch::zeros_like(tri_vertex);
    auto dL_dtri_feature = torch::zeros_like(tri_feature);
    auto dL_dtri_sh = torch::zeros_like(tri_sh);
    auto dL_dtri_opacity = torch::zeros_like(tri_opacity);
    auto dL_dgau_means3D = torch::zeros_like(gau_means3D);
    auto dL_dgau_feature = torch::zeros_like(gau_feature);
    auto dL_dgau_sh = torch::zeros_like(gau_sh);
    auto dL_dgau_opacity = torch::zeros_like(gau_opacity);
    return std::make_tuple(
        dL_dtri_vertex,
        dL_dtri_feature,
        dL_dtri_sh,
        dL_dtri_opacity,
        dL_dgau_means3D,
        dL_dgau_feature,
        dL_dgau_sh,
        dL_dgau_opacity);
}
}
