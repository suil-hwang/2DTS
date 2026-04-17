#include <vector>
#include <cmath>
#include <cfloat>

#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>

#include "rasterizer.h"
#include "forward.h"
#include "config.h"
#include "auxiliary.h"

uint32_t getHigherMsb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1)
    {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}

__forceinline__ __device__ float getEcc(
    const float3 &v1_view,
    const float3 &v2_view,
    const float3 &v3_view,
    const float3 &normal_view,
    const float3 &p_ray,
    float &depth)
{
    float p_ray_dot_n = dot(p_ray, normal_view);
    p_ray_dot_n = p_ray_dot_n > 0 ? fmaxf(p_ray_dot_n, EPS) : fminf(p_ray_dot_n, -EPS);
    depth = dot(v1_view, normal_view) / p_ray_dot_n;

    const float3 p_view = depth * p_ray;
    const float3 p_v1 = v1_view - p_view;
    const float3 p_v2 = v2_view - p_view;
    const float3 p_v3 = v3_view - p_view;

    const float inv_n_dot_n = 1.0f / dot(normal_view, normal_view);
    const float a1 = dot(cross(p_v2, p_v3), normal_view) * inv_n_dot_n;
    const float a2 = dot(cross(p_v3, p_v1), normal_view) * inv_n_dot_n;
    const float a3 = 1.0f - a1 - a2;
    return 1.0f - 3.0f * fminf(fminf(a1, a2), a3);
}

__forceinline__ __device__ void testRay(
    const float3 &v1_view,
    const float3 &v2_view,
    const float3 &v3_view,
    const float3 &normal_view,
    const float3 &p_ray,
    float &lowest_ecc,
    float &depth)
{
    float temp_depth;
    const float ecc = getEcc(v1_view, v2_view, v3_view, normal_view, p_ray, temp_depth);
    if (ecc < lowest_ecc)
    {
        lowest_ecc = ecc;
        depth = temp_depth;
    }
}

__forceinline__ __device__ bool findIntersection(
    const float2 &a1,
    const float2 &a2,
    const float2 &b1,
    const float2 &b2,
    float2 &intersection)
{
    float denom = (b2.y - b1.y) * (a2.x - a1.x) - (b2.x - b1.x) * (a2.y - a1.y);
    if (denom == 0.0f)
        return false;

    float ua = ((b2.x - b1.x) * (a1.y - b1.y) - (b2.y - b1.y) * (a1.x - b1.x)) / denom;
    float ub = ((a2.x - a1.x) * (a1.y - b1.y) - (a2.y - a1.y) * (a1.x - b1.x)) / denom;
    if (ua < 0.0f || ub < 0.0f || ub > 1.0f)
        return false;

    intersection = make_float2(a1.x + ua * (a2.x - a1.x), a1.y + ua * (a2.y - a1.y));
    return true;
}

__forceinline__ __device__ void testEdge(
    const float2 &v1_2D,
    const float2 &v2_2D,
    const float2 &v3_2D,
    const float2 &center_2D,
    const float2 &edge_start,
    const float2 &edge_end,
    const float3 &v1_view,
    const float3 &v2_view,
    const float3 &v3_view,
    const float3 &normal_view,
    int W,
    int H,
    float tan_fovx,
    float tan_fovy,
    float &lowest_ecc,
    float &depth)
{
    float2 intersection;
    float3 p_ray;
    if (findIntersection(center_2D, v1_2D, edge_start, edge_end, intersection))
    {
        p_ray = make_float3(tan_fovx * pixToProj(intersection.x, W), tan_fovy * pixToProj(intersection.y, H), 1.0f);
        testRay(v1_view, v2_view, v3_view, normal_view, p_ray, lowest_ecc, depth);
    }
    if (findIntersection(center_2D, v2_2D, edge_start, edge_end, intersection))
    {
        p_ray = make_float3(tan_fovx * pixToProj(intersection.x, W), tan_fovy * pixToProj(intersection.y, H), 1.0f);
        testRay(v1_view, v2_view, v3_view, normal_view, p_ray, lowest_ecc, depth);
    }
    if (findIntersection(center_2D, v3_2D, edge_start, edge_end, intersection))
    {
        p_ray = make_float3(tan_fovx * pixToProj(intersection.x, W), tan_fovy * pixToProj(intersection.y, H), 1.0f);
        testRay(v1_view, v2_view, v3_view, normal_view, p_ray, lowest_ecc, depth);
    }
}

__device__ float getLowestEcc(
    int W,
    int H,
    float tan_fovx,
    float tan_fovy,
    const float3 &v1_view,
    const float3 &v2_view,
    const float3 &v3_view,
    const float3 &normal_view,
    const float2 &v1_2D,
    const float2 &v2_2D,
    const float2 &v3_2D,
    const float2 &center_2D,
    const float2 &bbox_min,
    const float2 &bbox_max,
    float &depth)
{
    float lowest_ecc = FLT_MAX;
    const float2 bbox_min_view = {tan_fovx * pixToProj(bbox_min.x, W), tan_fovy * pixToProj(bbox_min.y, H)};
    const float2 bbox_max_view = {tan_fovx * pixToProj(bbox_max.x, W), tan_fovy * pixToProj(bbox_max.y, H)};

    if (center_2D.x >= bbox_min.x && center_2D.x <= bbox_max.x && center_2D.y >= bbox_min.y && center_2D.y <= bbox_max.y)
    {
        lowest_ecc = 0.0f;
    }
    else if (center_2D.x < bbox_min.x && center_2D.y < bbox_min.y)
    {
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
        testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_min.y), make_float2(bbox_min.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
        testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_min.y), make_float2(bbox_max.x, bbox_min.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
    }
    else if (center_2D.x >= bbox_min.x && center_2D.x <= bbox_max.x && center_2D.y < bbox_min.y)
    {
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
        testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_min.y), make_float2(bbox_max.x, bbox_min.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
    }
    else if (center_2D.x > bbox_max.x && center_2D.y < bbox_min.y)
    {
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
        testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_min.y), make_float2(bbox_max.x, bbox_min.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
        testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_max.x, bbox_min.y), make_float2(bbox_max.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
    }
    else if (center_2D.x < bbox_min.x && center_2D.y >= bbox_min.y && center_2D.y <= bbox_max.y)
    {
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
        testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_min.y), make_float2(bbox_min.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
    }
    else if (center_2D.x > bbox_max.x && center_2D.y >= bbox_min.y && center_2D.y <= bbox_max.y)
    {
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
        testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_max.x, bbox_min.y), make_float2(bbox_max.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
    }
    else if (center_2D.x < bbox_min.x && center_2D.y > bbox_max.y)
    {
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
        testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_min.y), make_float2(bbox_min.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
        testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_max.y), make_float2(bbox_max.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
    }
    else if (center_2D.x >= bbox_min.x && center_2D.x <= bbox_max.x && center_2D.y > bbox_max.y)
    {
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
        testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_max.y), make_float2(bbox_max.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
    }
    else if (center_2D.x > bbox_max.x && center_2D.y > bbox_max.y)
    {
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_min_view.y, 1.0f), lowest_ecc, depth);
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_min_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
        testRay(v1_view, v2_view, v3_view, normal_view, make_float3(bbox_max_view.x, bbox_max_view.y, 1.0f), lowest_ecc, depth);
        testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_max.x, bbox_min.y), make_float2(bbox_max.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
        testEdge(v1_2D, v2_2D, v3_2D, center_2D, make_float2(bbox_min.x, bbox_max.y), make_float2(bbox_max.x, bbox_max.y), v1_view, v2_view, v3_view, normal_view, W, H, tan_fovx, tan_fovy, lowest_ecc, depth);
    }

    return lowest_ecc;
}

__global__ void duplicateWithKeysTrianglesCUDA(
    int P,
    int tile_grid_x,
    int base_offset,
    const uint2 *__restrict__ rect_min,
    const uint2 *__restrict__ rect_max,
    const float *__restrict__ depth,
    const uint32_t *__restrict__ offsets,
    uint64_t *__restrict__ keys,
    uint32_t *__restrict__ values)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P)
        return;

    uint2 rmin = rect_min[idx];
    uint2 rmax = rect_max[idx];
    uint32_t off = base_offset + ((idx == 0) ? 0 : offsets[idx - 1]);
    for (int y = rmin.y; y < rmax.y; ++y)
    {
        for (int x = rmin.x; x < rmax.x; ++x)
        {
            uint64_t tile = ((uint64_t)y) * tile_grid_x + x;
            uint32_t z = *((uint32_t *)&depth[idx]);
            keys[off] = (tile << 32) | z;
            values[off] = (uint32_t)idx;
            off++;
        }
    }
}

__global__ void duplicateWithKeysTrianglesPerTileDepthCUDA(
    int P,
    int image_width,
    int image_height,
    int tile_grid_x,
    int tile_grid_y,
    int base_offset,
    float tan_fovx,
    float tan_fovy,
    float gamma,
    const uint2 *__restrict__ rect_min,
    const uint2 *__restrict__ rect_max,
    const float *__restrict__ depth,
    const uint32_t *__restrict__ offsets,
    const float3 *__restrict__ v1_view,
    const float3 *__restrict__ v2_view,
    const float3 *__restrict__ v3_view,
    const float3 *__restrict__ normal_view,
    const float *__restrict__ projmatrix,
    uint64_t *__restrict__ keys,
    uint32_t *__restrict__ values)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P)
        return;

    uint2 rmin = rect_min[idx];
    uint2 rmax = rect_max[idx];
    float cur_depth = depth[idx];
    uint32_t off = base_offset + ((idx == 0) ? 0 : offsets[idx - 1]);

    const float3 cur_v1_view = v1_view[idx];
    const float3 cur_v2_view = v2_view[idx];
    const float3 cur_v3_view = v3_view[idx];
    const float3 cur_normal_view = normal_view[idx];
    const float3 center_view = (cur_v1_view + cur_v2_view + cur_v3_view) / 3.0f;

    const float3 v1_view_clip = make_float3(cur_v1_view.x, cur_v1_view.y, fmaxf(cur_v1_view.z, 0.001f));
    const float3 v2_view_clip = make_float3(cur_v2_view.x, cur_v2_view.y, fmaxf(cur_v2_view.z, 0.001f));
    const float3 v3_view_clip = make_float3(cur_v3_view.x, cur_v3_view.y, fmaxf(cur_v3_view.z, 0.001f));
    const float3 center_view_clip = make_float3(center_view.x, center_view.y, fmaxf(center_view.z, 0.001f));

    const float3 v1_proj = projectPoint(v1_view_clip, projmatrix);
    const float3 v2_proj = projectPoint(v2_view_clip, projmatrix);
    const float3 v3_proj = projectPoint(v3_view_clip, projmatrix);
    const float3 center_proj = projectPoint(center_view_clip, projmatrix);

    const float2 v1_2D = {projToPix(v1_proj.x, image_width), projToPix(v1_proj.y, image_height)};
    const float2 v2_2D = {projToPix(v2_proj.x, image_width), projToPix(v2_proj.y, image_height)};
    const float2 v3_2D = {projToPix(v3_proj.x, image_width), projToPix(v3_proj.y, image_height)};
    const float2 center_2D = {projToPix(center_proj.x, image_width), projToPix(center_proj.y, image_height)};

    const float ecc_thres = powf(-2.0f * logf(G_THRES), 0.5f / gamma);
    const uint64_t invalid_tile = (uint64_t)(tile_grid_x * tile_grid_y);

    for (int y = rmin.y; y < rmax.y; ++y)
    {
        for (int x = rmin.x; x < rmax.x; ++x)
        {
            const float2 bbox_min = make_float2(x * BLOCK_X - 0.5f, y * BLOCK_Y - 0.5f);
            const float2 bbox_max = make_float2(
                fminf((x + 1) * BLOCK_X - 0.5f, image_width - 0.5f),
                fminf((y + 1) * BLOCK_Y - 0.5f, image_height - 0.5f));

            const float lowest_ecc = getLowestEcc(
                image_width,
                image_height,
                tan_fovx,
                tan_fovy,
                cur_v1_view,
                cur_v2_view,
                cur_v3_view,
                cur_normal_view,
                v1_2D,
                v2_2D,
                v3_2D,
                center_2D,
                bbox_min,
                bbox_max,
                cur_depth);
            cur_depth = fmaxf(cur_depth, 0.0f);

            uint64_t tile = lowest_ecc < ecc_thres ? ((uint64_t)y) * tile_grid_x + x : invalid_tile;
            uint32_t z = *((uint32_t *)&cur_depth);
            keys[off] = (tile << 32) | z;
            values[off] = (uint32_t)idx;
            off++;
        }
    }
}

__global__ void duplicateWithKeysGaussiansCUDA(
    int P,
    int tile_grid_x,
    int base_offset,
    const uint2 *__restrict__ rect_min,
    const uint2 *__restrict__ rect_max,
    const float *__restrict__ depth,
    const uint32_t *__restrict__ offsets,
    uint64_t *__restrict__ keys,
    uint32_t *__restrict__ values)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P)
        return;

    uint2 rmin = rect_min[idx];
    uint2 rmax = rect_max[idx];
    uint32_t off = base_offset + ((idx == 0) ? 0 : offsets[idx - 1]);
    for (int y = rmin.y; y < rmax.y; ++y)
    {
        for (int x = rmin.x; x < rmax.x; ++x)
        {
            uint64_t tile = ((uint64_t)y) * tile_grid_x + x;
            uint32_t z = *((uint32_t *)&depth[idx]);
            keys[off] = (tile << 32) | z;
            values[off] = ((uint32_t)idx) | 0x80000000u;
            off++;
        }
    }
}

__global__ void identifyTileRangesCUDA(int L, const uint64_t *keys, uint2 *ranges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L)
        return;

    uint64_t key = keys[idx];
    uint32_t tile = key >> 32;

    if (idx == 0)
    {
        ranges[tile].x = 0;
    }
    else
    {
        uint32_t prev_tile = keys[idx - 1] >> 32;
        if (tile != prev_tile)
        {
            ranges[prev_tile].y = idx;
            ranges[tile].x = idx;
        }
    }

    if (idx == L - 1)
    {
        ranges[tile].y = L;
    }
}

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
    bool rich_info)
{
    (void)debug;

    const auto device = tri_vertex.device();

    const int P_tri = tri_vertex.size(0);
    const int P_gau = gau_means3D.size(0);
    const int C = 3;

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((image_width + BLOCK_X - 1) / BLOCK_X, (image_height + BLOCK_Y - 1) / BLOCK_Y);

    auto opts_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto opts_uint = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto opts_bool = torch::TensorOptions().dtype(torch::kBool).device(device);

    auto tri_radii = torch::zeros({P_tri}, opts_int);
    auto tri_depth = torch::zeros({P_tri}, opts_float);
    auto tri_v1 = torch::zeros({P_tri, 3}, opts_float);
    auto tri_v2 = torch::zeros({P_tri, 3}, opts_float);
    auto tri_v3 = torch::zeros({P_tri, 3}, opts_float);
    auto tri_n = torch::zeros({P_tri, 3}, opts_float);
    auto tri_tiles = torch::zeros({P_tri}, opts_uint);
    auto tri_rect_min = torch::zeros({P_tri, 2}, opts_int);
    auto tri_rect_max = torch::zeros({P_tri, 2}, opts_int);
    auto tri_clamped = torch::zeros({(use_tri_vertex_color ? P_tri * 3 : P_tri), 3}, opts_bool);

    auto tri_rgb = use_tri_vertex_color ? torch::zeros({P_tri, 3, C}, opts_float) : torch::zeros({P_tri, C}, opts_float);
    auto tri_sh_in = use_tri_sh ? tri_sh : torch::empty({0}, opts_float);

    const int tri_M = use_tri_sh ? (use_tri_vertex_color ? tri_sh.size(2) : tri_sh.size(1)) : 0;
    FORWARD::preprocessTriangles(
        image_width,
        image_height,
        P_tri,
        tri_sh_degree,
        tri_M,
        tri_gamma,
        use_tri_sh,
        use_tri_vertex_color,
        back_culling,
        grid,
        viewmatrix.data_ptr<float>(),
        projmatrix.data_ptr<float>(),
        campos.data_ptr<float>(),
        tri_vertex.data_ptr<float>(),
        use_tri_sh ? tri_sh_in.data_ptr<float>() : nullptr,
        tri_radii.data_ptr<int>(),
        (float3 *)tri_v1.data_ptr<float>(),
        (float3 *)tri_v2.data_ptr<float>(),
        (float3 *)tri_v3.data_ptr<float>(),
        (float3 *)tri_n.data_ptr<float>(),
        tri_depth.data_ptr<float>(),
        tri_rgb.data_ptr<float>(),
        tri_clamped.data_ptr<bool>(),
        (uint32_t *)tri_tiles.data_ptr<int>(),
        (uint2 *)tri_rect_min.data_ptr<int>(),
        (uint2 *)tri_rect_max.data_ptr<int>());

    auto gau_radii = torch::zeros({P_gau}, opts_int);
    auto gau_xy = torch::zeros({P_gau, 2}, opts_float);
    auto gau_depth = torch::zeros({P_gau}, opts_float);
    auto gau_cov3d = torch::zeros({P_gau, 6}, opts_float);
    auto gau_conic = torch::zeros({P_gau, 4}, opts_float);
    auto gau_tiles = torch::zeros({P_gau}, opts_uint);
    auto gau_rect_min = torch::zeros({P_gau, 2}, opts_int);
    auto gau_rect_max = torch::zeros({P_gau, 2}, opts_int);
    auto gau_clamped = torch::zeros({P_gau, 3}, opts_bool);
    auto gau_feat = torch::zeros({P_gau, C}, opts_float);

    auto gau_sh_in = use_gau_sh ? gau_sh : torch::empty({0}, opts_float);
    auto gau_color_in = use_gau_sh ? torch::empty({0}, opts_float) : gau_feature;
    auto gau_cov_precomp_in = use_gau_cov_precomp ? gau_cov3D_precomp : torch::empty({0}, opts_float);
    auto gau_scales_in = use_gau_cov_precomp ? torch::empty({0}, opts_float) : gau_scales;
    auto gau_rot_in = use_gau_cov_precomp ? torch::empty({0}, opts_float) : gau_rotations;
    FORWARD::preprocessGaussians(
        image_width,
        image_height,
        P_gau,
        gau_sh_degree,
        use_gau_sh ? gau_sh.size(1) : 0,
        tan_fovx,
        tan_fovy,
        focal_x,
        focal_y,
        1.0f,
        prefiltered,
        use_gau_sh,
        use_gau_cov_precomp,
        grid,
        viewmatrix.data_ptr<float>(),
        projmatrix.data_ptr<float>(),
        full_projmatrix.data_ptr<float>(),
        campos.data_ptr<float>(),
        gau_means3D.data_ptr<float>(),
        use_gau_sh ? gau_sh_in.data_ptr<float>() : nullptr,
        use_gau_sh ? nullptr : gau_color_in.data_ptr<float>(),
        gau_opacity.data_ptr<float>(),
        use_gau_cov_precomp ? nullptr : gau_scales_in.data_ptr<float>(),
        use_gau_cov_precomp ? nullptr : gau_rot_in.data_ptr<float>(),
        use_gau_cov_precomp ? gau_cov_precomp_in.data_ptr<float>() : nullptr,
        gau_radii.data_ptr<int>(),
        (float2 *)gau_xy.data_ptr<float>(),
        gau_depth.data_ptr<float>(),
        gau_cov3d.data_ptr<float>(),
        (float4 *)gau_conic.data_ptr<float>(),
        gau_feat.data_ptr<float>(),
        gau_clamped.data_ptr<bool>(),
        (uint32_t *)gau_tiles.data_ptr<int>(),
        (uint2 *)gau_rect_min.data_ptr<int>(),
        (uint2 *)gau_rect_max.data_ptr<int>());

    auto tri_offsets = torch::zeros({P_tri}, opts_uint);
    auto gau_offsets = torch::zeros({P_gau}, opts_uint);
    size_t tri_scan_size = 0;
    size_t gau_scan_size = 0;
    if (P_tri > 0)
    {
        cub::DeviceScan::InclusiveSum(nullptr, tri_scan_size, (uint32_t *)tri_tiles.data_ptr<int>(), (uint32_t *)tri_offsets.data_ptr<int>(), P_tri);
    }
    if (P_gau > 0)
    {
        cub::DeviceScan::InclusiveSum(nullptr, gau_scan_size, (uint32_t *)gau_tiles.data_ptr<int>(), (uint32_t *)gau_offsets.data_ptr<int>(), P_gau);
    }

    auto tri_scan = torch::zeros({(long)tri_scan_size}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    auto gau_scan = torch::zeros({(long)gau_scan_size}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    if (P_tri > 0)
    {
        cub::DeviceScan::InclusiveSum(tri_scan.data_ptr(), tri_scan_size, (uint32_t *)tri_tiles.data_ptr<int>(), (uint32_t *)tri_offsets.data_ptr<int>(), P_tri);
    }
    if (P_gau > 0)
    {
        cub::DeviceScan::InclusiveSum(gau_scan.data_ptr(), gau_scan_size, (uint32_t *)gau_tiles.data_ptr<int>(), (uint32_t *)gau_offsets.data_ptr<int>(), P_gau);
    }

    int tri_instances = 0;
    int gau_instances = 0;
    if (P_tri > 0)
        cudaMemcpy(&tri_instances, tri_offsets.data_ptr<int>() + (P_tri - 1), sizeof(int), cudaMemcpyDeviceToHost);
    if (P_gau > 0)
        cudaMemcpy(&gau_instances, gau_offsets.data_ptr<int>() + (P_gau - 1), sizeof(int), cudaMemcpyDeviceToHost);
    int L = tri_instances + gau_instances;

    auto opts_key = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto unsorted_keys = torch::zeros({L}, opts_key);
    auto unsorted_vals = torch::zeros({L}, opts_uint);

    if (P_tri > 0 && tri_instances > 0)
    {
        if (sort_level == 0)
        {
            duplicateWithKeysTrianglesCUDA<<<(P_tri + 255) / 256, 256>>>(
                P_tri,
                (int)grid.x,
                0,
                (uint2 *)tri_rect_min.data_ptr<int>(),
                (uint2 *)tri_rect_max.data_ptr<int>(),
                tri_depth.data_ptr<float>(),
                (uint32_t *)tri_offsets.data_ptr<int>(),
                (uint64_t *)unsorted_keys.data_ptr<int64_t>(),
                (uint32_t *)unsorted_vals.data_ptr<int>());
        }
        else
        {
            duplicateWithKeysTrianglesPerTileDepthCUDA<<<(P_tri + 255) / 256, 256>>>(
                P_tri,
                image_width,
                image_height,
                (int)grid.x,
                (int)grid.y,
                0,
                tan_fovx,
                tan_fovy,
                tri_gamma,
                (uint2 *)tri_rect_min.data_ptr<int>(),
                (uint2 *)tri_rect_max.data_ptr<int>(),
                tri_depth.data_ptr<float>(),
                (uint32_t *)tri_offsets.data_ptr<int>(),
                (float3 *)tri_v1.data_ptr<float>(),
                (float3 *)tri_v2.data_ptr<float>(),
                (float3 *)tri_v3.data_ptr<float>(),
                (float3 *)tri_n.data_ptr<float>(),
                projmatrix.data_ptr<float>(),
                (uint64_t *)unsorted_keys.data_ptr<int64_t>(),
                (uint32_t *)unsorted_vals.data_ptr<int>());
        }
    }

    if (P_gau > 0 && gau_instances > 0)
    {
        duplicateWithKeysGaussiansCUDA<<<(P_gau + 255) / 256, 256>>>(
            P_gau,
            (int)grid.x,
            tri_instances,
            (uint2 *)gau_rect_min.data_ptr<int>(),
            (uint2 *)gau_rect_max.data_ptr<int>(),
            gau_depth.data_ptr<float>(),
            (uint32_t *)gau_offsets.data_ptr<int>(),
            (uint64_t *)unsorted_keys.data_ptr<int64_t>(),
            (uint32_t *)unsorted_vals.data_ptr<int>());
    }

    auto sorted_keys = torch::zeros({L}, opts_key);
    auto sorted_vals = torch::zeros({L}, opts_uint);
    size_t sort_size = 0;
    const int tile_sort_bits = getHigherMsb((uint32_t)(grid.x * grid.y));
    if (L > 0)
    {
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            sort_size,
            (uint64_t *)unsorted_keys.data_ptr<int64_t>(),
            (uint64_t *)sorted_keys.data_ptr<int64_t>(),
            (uint32_t *)unsorted_vals.data_ptr<int>(),
            (uint32_t *)sorted_vals.data_ptr<int>(),
            L,
            0,
            32 + tile_sort_bits);
    }

    auto sort_temp = torch::zeros({(long)sort_size}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    if (L > 0)
    {
        cub::DeviceRadixSort::SortPairs(
            sort_temp.data_ptr(),
            sort_size,
            (uint64_t *)unsorted_keys.data_ptr<int64_t>(),
            (uint64_t *)sorted_keys.data_ptr<int64_t>(),
            (uint32_t *)unsorted_vals.data_ptr<int>(),
            (uint32_t *)sorted_vals.data_ptr<int>(),
            L,
            0,
            32 + tile_sort_bits);
    }

    auto tile_ranges = torch::zeros({(long)(grid.x * grid.y + 1), 2}, opts_int);
    if (L > 0)
    {
        identifyTileRangesCUDA<<<(L + 255) / 256, 256>>>(
            L,
            (uint64_t *)sorted_keys.data_ptr<int64_t>(),
            (uint2 *)tile_ranges.data_ptr<int>());
    }

    auto out_feature = torch::zeros({C, image_height, image_width}, opts_float);
    auto out_depth = torch::zeros({image_height, image_width}, opts_float);
    auto out_normal = torch::zeros({3, image_height, image_width}, opts_float);
    auto out_distort = torch::zeros({image_height, image_width}, opts_float);
    auto final_Ts = torch::zeros({image_height, image_width}, opts_float);
    auto n_contribs = torch::zeros({image_height, image_width}, opts_uint);

    auto contrib_sum = torch::zeros({P_tri + P_gau}, opts_float);
    auto contrib_max = torch::zeros({P_tri + P_gau}, opts_float);

    auto tri_feat_in = use_tri_sh ? tri_rgb : tri_feature;
    int tri_tex_H = use_tri_texture ? (int)tri_texture.size(0) : 0;
    int tri_tex_W = use_tri_texture ? (int)tri_texture.size(1) : 0;

    if (sort_level <= 1)
    {
        FORWARD::renderHybrid(
            grid,
            block,
            image_width,
            image_height,
            C,
            P_tri,
            tri_gamma,
            gau_gamma,
            ambient_intensity,
            light_color.data_ptr<float>(),
            light_dir.data_ptr<float>(),
            light_intensity,
            rich_info,
            use_tri_texture,
            use_tri_vertex_color,
            tan_fovx,
            tan_fovy,
            (uint2 *)tile_ranges.data_ptr<int>(),
            (uint32_t *)sorted_vals.data_ptr<int>(),
            (float3 *)tri_v1.data_ptr<float>(),
            (float3 *)tri_v2.data_ptr<float>(),
            (float3 *)tri_v3.data_ptr<float>(),
            (float3 *)tri_n.data_ptr<float>(),
            tri_feat_in.data_ptr<float>(),
            use_tri_texture ? (float2 *)tri_uv.data_ptr<float>() : nullptr,
            use_tri_texture ? tri_texture.data_ptr<float>() : nullptr,
            tri_tex_H,
            tri_tex_W,
            tri_opacity.data_ptr<float>(),
            (float2 *)gau_xy.data_ptr<float>(),
            (float4 *)gau_conic.data_ptr<float>(),
            gau_feat.data_ptr<float>(),
            gau_depth.data_ptr<float>(),
            background.data_ptr<float>(),
            background_depth,
            final_Ts.data_ptr<float>(),
            (uint32_t *)n_contribs.data_ptr<int>(),
            out_feature.data_ptr<float>(),
            out_depth.data_ptr<float>(),
            out_normal.data_ptr<float>(),
            out_distort.data_ptr<float>(),
            contrib_sum.data_ptr<float>(),
            contrib_max.data_ptr<float>());
    }
    else
    {
        FORWARD::renderHybridResort(
            grid,
            block,
            image_width,
            image_height,
            C,
            P_tri,
            tri_gamma,
            gau_gamma,
            ambient_intensity,
            light_color.data_ptr<float>(),
            light_dir.data_ptr<float>(),
            light_intensity,
            rich_info,
            use_tri_texture,
            use_tri_vertex_color,
            tan_fovx,
            tan_fovy,
            (uint2 *)tile_ranges.data_ptr<int>(),
            (uint32_t *)sorted_vals.data_ptr<int>(),
            (float3 *)tri_v1.data_ptr<float>(),
            (float3 *)tri_v2.data_ptr<float>(),
            (float3 *)tri_v3.data_ptr<float>(),
            (float3 *)tri_n.data_ptr<float>(),
            tri_feat_in.data_ptr<float>(),
            use_tri_texture ? (float2 *)tri_uv.data_ptr<float>() : nullptr,
            use_tri_texture ? tri_texture.data_ptr<float>() : nullptr,
            tri_tex_H,
            tri_tex_W,
            tri_opacity.data_ptr<float>(),
            (float2 *)gau_xy.data_ptr<float>(),
            (float4 *)gau_conic.data_ptr<float>(),
            gau_feat.data_ptr<float>(),
            gau_depth.data_ptr<float>(),
            background.data_ptr<float>(),
            background_depth,
            final_Ts.data_ptr<float>(),
            (uint32_t *)n_contribs.data_ptr<int>(),
            out_feature.data_ptr<float>(),
            out_depth.data_ptr<float>(),
            out_normal.data_ptr<float>(),
            out_distort.data_ptr<float>(),
            contrib_sum.data_ptr<float>(),
            contrib_max.data_ptr<float>());
    }

    return {
        L,
        out_feature,
        out_depth,
        out_normal,
        out_distort,
        n_contribs,
        final_Ts,
        tri_radii,
        contrib_sum,
        contrib_max};
}
}
