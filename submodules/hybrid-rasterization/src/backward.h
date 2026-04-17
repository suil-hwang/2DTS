#pragma once

#include <torch/extension.h>
#include <tuple>

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
    const torch::Tensor &gau_opacity);
}
