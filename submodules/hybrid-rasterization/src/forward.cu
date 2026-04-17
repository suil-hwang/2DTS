#include <cooperative_groups.h>

#include <cfloat>

#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "forward.h"
#include "config.h"
#include "auxiliary.h"

namespace cg = cooperative_groups;

__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f};
__device__ const float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f};

__forceinline__ __device__ float3 computeRGBFromSH(int idx, int deg, int max_coeffs, float3 pos, float3 campos, const float *shs, bool *clamped)
{
    float3 dir = pos - campos;
    dir = dir / max(norm(dir), EPS);

    float3 *sh = ((float3 *)shs) + idx * max_coeffs;
    float3 rgb = SH_C0 * sh[0];

    if (deg > 0)
    {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        rgb = rgb - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        if (deg > 1)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            rgb = rgb +
                  SH_C2[0] * xy * sh[4] +
                  SH_C2[1] * yz * sh[5] +
                  SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                  SH_C2[3] * xz * sh[7] +
                  SH_C2[4] * (xx - yy) * sh[8];

            if (deg > 2)
            {
                rgb = rgb +
                      SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                      SH_C3[1] * xy * z * sh[10] +
                      SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                      SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                      SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                      SH_C3[5] * z * (xx - yy) * sh[14] +
                      SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }

    rgb = rgb + make_float3(0.5f, 0.5f, 0.5f);
    clamped[3 * idx + 0] = (rgb.x < 0);
    clamped[3 * idx + 1] = (rgb.y < 0);
    clamped[3 * idx + 2] = (rgb.z < 0);
    return make_float3(fmaxf(rgb.x, 0.0f), fmaxf(rgb.y, 0.0f), fmaxf(rgb.z, 0.0f));
}

__forceinline__ __device__ bool in_frustum(int idx, const float *points, const float *viewmatrix, bool prefiltered, float3 &p_view)
{
    float3 p = {points[3 * idx + 0], points[3 * idx + 1], points[3 * idx + 2]};
    p_view = transformPoint4x3(p, viewmatrix);
    if (p_view.z <= 0.2f)
    {
        if (prefiltered)
        {
            printf("Point filtered although prefiltered is true.\n");
            __trap();
        }
        return false;
    }
    return true;
}

__forceinline__ __device__ void computeCov3D(const float3 scale, float mod, const float4 rot, float *cov3D)
{
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    glm::vec4 q = {rot.x, rot.y, rot.z, rot.w};
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

    glm::mat3 M = S * R;
    glm::mat3 Sigma = glm::transpose(M) * M;

    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

__forceinline__ __device__ float3 computeCov2D(const float3 &mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float *cov3D, const float *viewmatrix)
{
    float3 t = transformPoint4x3(mean, viewmatrix);
    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);

    glm::mat3 W = glm::mat3(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    glm::mat3 T = W * J;

    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return make_float3(float(cov[0][0]), float(cov[0][1]), float(cov[1][1]));
}

__global__ void preprocessTrianglesCUDA(
    int W, int H, int P, int D, int M,
    float gamma,
    bool use_shs,
    bool use_vertex_color,
    bool back_culling,
    dim3 grid,
    const float *__restrict__ viewmatrix,
    const float *__restrict__ projmatrix,
    const float *__restrict__ campos,
    const float *__restrict__ vertex,
    const float *__restrict__ shs,
    int *__restrict__ radii,
    float3 *__restrict__ v1_view,
    float3 *__restrict__ v2_view,
    float3 *__restrict__ v3_view,
    float3 *__restrict__ normal_view,
    float *__restrict__ depth,
    float *__restrict__ tri_rgb,
    bool *__restrict__ clamped,
    uint32_t *__restrict__ tiles_touched,
    uint2 *__restrict__ rect_min,
    uint2 *__restrict__ rect_max)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    radii[idx] = 0;
    tiles_touched[idx] = 0;

    const float3 p1 = {vertex[9 * idx + 0], vertex[9 * idx + 1], vertex[9 * idx + 2]};
    const float3 p2 = {vertex[9 * idx + 3], vertex[9 * idx + 4], vertex[9 * idx + 5]};
    const float3 p3 = {vertex[9 * idx + 6], vertex[9 * idx + 7], vertex[9 * idx + 8]};

    const float3 vv1 = transformPoint4x3(p1, viewmatrix);
    const float3 vv2 = transformPoint4x3(p2, viewmatrix);
    const float3 vv3 = transformPoint4x3(p3, viewmatrix);

    const float3 center_view = (vv1 + vv2 + vv3) / 3.0f;
    const float3 n_view = cross(vv2 - vv1, vv3 - vv1);
    if (norm(n_view) < EPS)
        return;

    const float dilation = powf(-2.0f * logf(G_THRES), 0.5f / gamma);
    float3 d1 = center_view + dilation * (vv1 - center_view);
    float3 d2 = center_view + dilation * (vv2 - center_view);
    float3 d3 = center_view + dilation * (vv3 - center_view);
    float3 q1 = projectPoint(d1, projmatrix);
    float3 q2 = projectPoint(d2, projmatrix);
    float3 q3 = projectPoint(d3, projmatrix);

    if (q1.z <= 0 && q2.z <= 0 && q3.z <= 0)
        return;

	// render triangles that extend behind the camera properly
	if (d1.z <= EPS)
	{
		d1.z = 0.001f;
		q1 = projectPoint(d1, projmatrix);
	}
	if (d2.z <= EPS)
	{
		d2.z = 0.001f;
		q2 = projectPoint(d2, projmatrix);
	}
	if (d3.z <= EPS)
	{
		d3.z = 0.001f;
		q3 = projectPoint(d3, projmatrix);
	}

    if (back_culling && cross(q2 - q1, q3 - q1).z >= 0)
        return;

    float2 t1 = {projToPix(q1.x, W), projToPix(q1.y, H)};
    float2 t2 = {projToPix(q2.x, W), projToPix(q2.y, H)};
    float2 t3 = {projToPix(q3.x, W), projToPix(q3.y, H)};

    float2 vmin = min(t1, t2, t3);
    float2 vmax = max(t1, t2, t3);

    if (vmin.x >= ((float)W - 0.5f) || vmin.y >= ((float)H - 0.5f) || vmax.x < -0.5f || vmax.y < -0.5f)
        return;

    vmin = {min(max(vmin.x, -0.5f), (float)W - 0.5f), min(max(vmin.y, -0.5f), (float)H - 0.5f)};
    vmax = {min(max(vmax.x, -0.5f), (float)W - 0.5f), min(max(vmax.y, -0.5f), (float)H - 0.5f)};

    uint2 rmin = {min(grid.x, max(0, (int)((vmin.x + 0.5f) / BLOCK_X))), min(grid.y, max(0, (int)((vmin.y + 0.5f) / BLOCK_Y)))};
    uint2 rmax = {min(grid.x, max(0, (int)((vmax.x + 0.5f + BLOCK_X) / BLOCK_X))), min(grid.y, max(0, (int)((vmax.y + 0.5f + BLOCK_Y) / BLOCK_Y)))};
    if (rmax.x <= rmin.x || rmax.y <= rmin.y)
        return;

    if (use_shs)
    {
        float3 cam = *(float3 *)campos;
        if (use_vertex_color)
        {
            float3 c1 = computeRGBFromSH(idx * 3 + 0, D, M, p1, cam, shs, clamped);
            float3 c2 = computeRGBFromSH(idx * 3 + 1, D, M, p2, cam, shs, clamped);
            float3 c3 = computeRGBFromSH(idx * 3 + 2, D, M, p3, cam, shs, clamped);
            tri_rgb[(idx * 3 + 0) * 3 + 0] = c1.x;
            tri_rgb[(idx * 3 + 0) * 3 + 1] = c1.y;
            tri_rgb[(idx * 3 + 0) * 3 + 2] = c1.z;
            tri_rgb[(idx * 3 + 1) * 3 + 0] = c2.x;
            tri_rgb[(idx * 3 + 1) * 3 + 1] = c2.y;
            tri_rgb[(idx * 3 + 1) * 3 + 2] = c2.z;
            tri_rgb[(idx * 3 + 2) * 3 + 0] = c3.x;
            tri_rgb[(idx * 3 + 2) * 3 + 1] = c3.y;
            tri_rgb[(idx * 3 + 2) * 3 + 2] = c3.z;
        }
        else
        {
            float3 c = computeRGBFromSH(idx, D, M, (p1 + p2 + p3) / 3.0f, cam, shs, clamped);
            tri_rgb[idx * 3 + 0] = c.x;
            tri_rgb[idx * 3 + 1] = c.y;
            tri_rgb[idx * 3 + 2] = c.z;
        }
    }

    v1_view[idx] = vv1;
    v2_view[idx] = vv2;
    v3_view[idx] = vv3;
    normal_view[idx] = n_view;
    depth[idx] = fmaxf(center_view.z, 0.0f);
    rect_min[idx] = rmin;
    rect_max[idx] = rmax;
    tiles_touched[idx] = (rmax.x - rmin.x) * (rmax.y - rmin.y);
    radii[idx] = max((int)ceilf(vmax.x - vmin.x), (int)ceilf(vmax.y - vmin.y));
}

__global__ void preprocessGaussiansCUDA(
    int W, int H, int P, int D, int M,
    float tan_fovx, float tan_fovy,
    float focal_x, float focal_y,
    float scale_modifier,
    bool prefiltered,
    bool use_shs,
    bool use_cov_precomp,
    dim3 grid,
    const float *__restrict__ viewmatrix,
    const float *__restrict__ projmatrix,
    const float *__restrict__ full_projmatrix,
    const float *__restrict__ campos,
    const float *__restrict__ means3D,
    const float *__restrict__ gaussian_shs,
    const float *__restrict__ gaussian_colors,
    const float *__restrict__ gaussian_opacity,
    const float *__restrict__ gaussian_scales,
    const float *__restrict__ gaussian_rotations,
    const float *__restrict__ gaussian_cov3D_precomp,
    int *__restrict__ radii,
    float2 *__restrict__ means2D,
    float *__restrict__ depths,
    float *__restrict__ cov3Ds,
    float4 *__restrict__ conic_opacity,
    float *__restrict__ feature,
    bool *__restrict__ clamped,
    uint32_t *__restrict__ tiles_touched,
    uint2 *__restrict__ rect_min,
    uint2 *__restrict__ rect_max)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    radii[idx] = 0;
    tiles_touched[idx] = 0;

    float3 p_view;
    if (!in_frustum(idx, means3D, viewmatrix, prefiltered, p_view))
        return;

    float3 p_orig = {means3D[3 * idx + 0], means3D[3 * idx + 1], means3D[3 * idx + 2]};

    const float *cov3D = nullptr;
    if (use_cov_precomp)
    {
        cov3D = gaussian_cov3D_precomp + idx * 6;
    }
    else
    {
        float3 sc = {gaussian_scales[3 * idx + 0], gaussian_scales[3 * idx + 1], gaussian_scales[3 * idx + 2]};
        float4 rt = {gaussian_rotations[4 * idx + 0], gaussian_rotations[4 * idx + 1], gaussian_rotations[4 * idx + 2], gaussian_rotations[4 * idx + 3]};
        computeCov3D(sc, scale_modifier, rt, cov3Ds + idx * 6);
        cov3D = cov3Ds + idx * 6;
    }

    float4 p_hom = transformPoint4x4(p_orig, full_projmatrix);
    float inv_w = 1.0f / (p_hom.w + 1e-7f);
    float2 p_ndc = {p_hom.x * inv_w, p_hom.y * inv_w};
    float2 p_image = {ndc2Pix(p_ndc.x, W), ndc2Pix(p_ndc.y, H)};

    float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
    float det = cov.x * cov.z - cov.y * cov.y;
    if (det == 0.0f)
        return;
    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + sqrtf(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrtf(max(0.1f, mid * mid - det));
    float my_radius = ceilf(3.0f * sqrtf(max(lambda1, lambda2)));

    float det_inv = 1.f / det;
    float3 conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};
    conic_opacity[idx] = {conic.x, conic.y, conic.z, gaussian_opacity[idx]};

    uint2 rmin, rmax;
    getRect(p_image, (int)my_radius, rmin, rmax, grid);
    if ((rmax.x - rmin.x) * (rmax.y - rmin.y) == 0)
        return;

    if (use_shs)
    {
        float3 c = computeRGBFromSH(idx, D, M, p_orig, *(float3 *)campos, gaussian_shs, clamped);
        feature[idx * 3 + 0] = c.x;
        feature[idx * 3 + 1] = c.y;
        feature[idx * 3 + 2] = c.z;
    }
    else
    {
        feature[idx * 3 + 0] = gaussian_colors[idx * 3 + 0];
        feature[idx * 3 + 1] = gaussian_colors[idx * 3 + 1];
        feature[idx * 3 + 2] = gaussian_colors[idx * 3 + 2];
    }

    means2D[idx] = p_image;
    depths[idx] = fmaxf(p_view.z, 0.0f);
    radii[idx] = (int)my_radius;
    tiles_touched[idx] = (rmax.x - rmin.x) * (rmax.y - rmin.y);
    rect_min[idx] = rmin;
    rect_max[idx] = rmax;
}

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderHybridCUDA(
    int W, int H, int C, int P_tri,
    float tri_gamma, float gau_gamma,
    float ambient_intensity,
    const float *__restrict__ light_color,
    const float *__restrict__ light_dir,
    float light_intensity,
    bool rich_info,
    bool tri_use_texture,
    bool tri_use_vertex_color,
    float tan_fovx, float tan_fovy,
    const uint2 *ranges,
    const uint32_t *point_list,
    const float3 *tri_v1,
    const float3 *tri_v2,
    const float3 *tri_v3,
    const float3 *tri_n,
    const float *tri_feature,
    const float2 *tri_uv,
    const float *tri_texture,
    int tri_tex_H,
    int tri_tex_W,
    const float *tri_opacity,
    const float2 *gau_xy,
    const float4 *gau_conic,
    const float *gau_feature,
    const float *gau_depth,
    const float *background,
    float background_depth,
    float *final_Ts,
    uint32_t *n_contribs,
    float *out_feature,
    float *out_depth,
    float *out_normal,
    float *out_distort,
    float *contrib_sum,
    float *contrib_max)
{
    auto block = cg::this_thread_block();
    dim3 group_index = block.group_index();
    dim3 thread_index = block.thread_index();

    uint2 pix = {group_index.x * BLOCK_X + thread_index.x, group_index.y * BLOCK_Y + thread_index.y};
    uint32_t pix_id = W * pix.y + pix.x;
    bool inside = pix.x < W && pix.y < H;
    bool done = !inside;

    float3 p_ray = {tan_fovx * pixToProj((float)pix.x, W), tan_fovy * pixToProj((float)pix.y, H), 1.0f};
    float3 light_dir_unit = {light_dir[0], light_dir[1], light_dir[2]};
    light_dir_unit = light_dir_unit / fmaxf(norm(light_dir_unit), EPS);
    float2 pixf = {(float)pix.x, (float)pix.y};

    uint2 range = ranges[group_index.y * ((W + BLOCK_X - 1) / BLOCK_X) + group_index.x];
    int rounds = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int toDo = range.y - range.x;

    __shared__ uint32_t ids[BLOCK_SIZE];

    float T = 1.0f;
    uint32_t n_contrib = 0;
    float accum_feature[MAX_CHANNELS] = {0, 0, 0};
    float3 N = {0, 0, 0};
    float accum_depth = 0.0f;
    float accum_depth_sq = 0.0f;
    float accum_dist = 0.0f;

    auto sample_texture = [&](float u, float v, int ch)
    {
        float uu = fminf(fmaxf(u, 0.0f), 1.0f);
        float vv = fminf(fmaxf(v, 0.0f), 1.0f);
        float x = uu * (tri_tex_W - 1);
        float y = (1.0f - vv) * (tri_tex_H - 1);

        int x0 = (int)floorf(x);
        int y0 = (int)floorf(y);
        int x1 = min(x0 + 1, tri_tex_W - 1);
        int y1 = min(y0 + 1, tri_tex_H - 1);

        float tx = x - x0;
        float ty = y - y0;

        int i00 = (y0 * tri_tex_W + x0) * C + ch;
        int i10 = (y0 * tri_tex_W + x1) * C + ch;
        int i01 = (y1 * tri_tex_W + x0) * C + ch;
        int i11 = (y1 * tri_tex_W + x1) * C + ch;

        float c00 = tri_texture[i00];
        float c10 = tri_texture[i10];
        float c01 = tri_texture[i01];
        float c11 = tri_texture[i11];

        float c0 = c00 * (1.0f - tx) + c10 * tx;
        float c1 = c01 * (1.0f - tx) + c11 * tx;
        return c0 * (1.0f - ty) + c1 * ty;
    };

    for (int i = 0; i < rounds; ++i, toDo -= BLOCK_SIZE)
    {
        if (__syncthreads_count(done) == BLOCK_SIZE)
            break;

        block.sync();
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
            ids[block.thread_rank()] = point_list[range.x + progress];
        block.sync();

        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); ++j)
        {
            n_contrib++;
            uint32_t id = ids[j];
            bool is_gauss = (id >> 31) != 0;
            float alpha = 0.0f;
            float depth = 0.0f;
            float3 n = {0, 0, 0};
            float contrib = 0.0f;

            if (!is_gauss)
            {
                int tri_id = (int)id;
                float3 v1 = tri_v1[tri_id];
                float3 v2 = tri_v2[tri_id];
                float3 v3 = tri_v3[tri_id];
                n = tri_n[tri_id];

                float p_ray_dot_n = dot(p_ray, n);
                if (fabsf(p_ray_dot_n) < EPS)
                    continue;
                depth = dot(v1, n) / p_ray_dot_n;
                if (depth < 0.0f)
                    continue;

                float3 p_view = depth * p_ray;
                float3 p_v1 = v1 - p_view;
                float3 p_v2 = v2 - p_view;
                float3 p_v3 = v3 - p_view;
                float inv_n_dot_n = 1.0f / dot(n, n);
                float a1 = dot(cross(p_v2, p_v3), n) * inv_n_dot_n;
                float a2 = dot(cross(p_v3, p_v1), n) * inv_n_dot_n;
                float a3 = 1.0f - a1 - a2;
                float ecc = 1.0f - 3.0f * min(min(a1, a2), a3);
                if (ecc < 0.0f || ecc > 10.0f)
                    continue;

                float G = tri_gamma > 50.0f ? (ecc <= 1.0f ? 1.0f : 0.0f) : expf(-0.5f * powf(ecc, 2.0f * tri_gamma));
                if (G < G_THRES)
                    continue;

                alpha = min(ALPHA_THRES, tri_opacity[tri_id] * G);
                contrib = alpha * T;

                float3 n_unit = n * sqrtf(inv_n_dot_n);
                float n_dot_l = fmaxf(dot(n_unit, light_dir_unit), 0.0f);

                for (int ch = 0; ch < C; ++ch)
                {
                    float feat;
                    if (tri_use_texture)
                    {
                        float2 uv1 = tri_uv[tri_id * 3 + 0];
                        float2 uv2 = tri_uv[tri_id * 3 + 1];
                        float2 uv3 = tri_uv[tri_id * 3 + 2];
                        float u = uv1.x * a1 + uv2.x * a2 + uv3.x * a3;
                        float v = uv1.y * a1 + uv2.y * a2 + uv3.y * a3;
                        feat = sample_texture(u, v, ch);
                    }
                    else if (tri_use_vertex_color)
                    {
                        feat = tri_feature[(tri_id * 3 + 0) * C + ch] * a1 +
                               tri_feature[(tri_id * 3 + 1) * C + ch] * a2 +
                               tri_feature[(tri_id * 3 + 2) * C + ch] * a3;
                    }
                    else
                    {
                        feat = tri_feature[tri_id * C + ch];
                    }
                    float lighting = ambient_intensity + light_intensity * light_color[ch] * n_dot_l;
                    accum_feature[ch] += feat * lighting * contrib;
                }

                n = n_unit;
                if (rich_info)
                {
                    atomicAdd(&contrib_sum[tri_id], contrib);
                    atomicMaxFloat(&contrib_max[tri_id], contrib);
                }
            }
            else
            {
                int gau_id = (int)(id & 0x7fffffff);
                float2 xy = gau_xy[gau_id];
                float2 d = {xy.x - pixf.x, xy.y - pixf.y};
                float4 co = gau_conic[gau_id];
                float q = co.x * d.x * d.x + co.z * d.y * d.y + 2.0f * co.y * d.x * d.y;
                float power = gau_gamma == 1.0f ? -0.5f * q : -0.5f * powf(q, gau_gamma);
                if (power > 0.0f)
                    continue;

                alpha = min(ALPHA_THRES, co.w * expf(power));
                if (alpha < G_THRES)
                    continue;

                depth = gau_depth[gau_id];
                contrib = alpha * T;

                for (int ch = 0; ch < C; ++ch)
                {
                    accum_feature[ch] += gau_feature[gau_id * C + ch] * contrib;
                }

                if (rich_info)
                {
                    atomicAdd(&contrib_sum[P_tri + gau_id], contrib);
                    atomicMaxFloat(&contrib_max[P_tri + gau_id], contrib);
                }
            }

            float test_T = T * (1.0f - alpha);
            if (test_T < T_THRES)
            {
                done = true;
                continue;
            }

            if (rich_info)
            {
                N += n * contrib;
                accum_dist += (depth * depth * (1.0f - T) + accum_depth_sq - 2.0f * depth * accum_depth) * contrib;
                accum_depth += depth * contrib;
                accum_depth_sq += depth * depth * contrib;
            }
            T = test_T;
        }
    }

    if (inside)
    {
        final_Ts[pix_id] = T;
        n_contribs[pix_id] = n_contrib;
        for (int ch = 0; ch < C; ++ch)
            out_feature[ch * H * W + pix_id] = accum_feature[ch] + T * background[ch];

        if (rich_info)
        {
            out_depth[pix_id] = accum_depth + T * background_depth;
            out_normal[pix_id] = N.x;
            out_normal[H * W + pix_id] = N.y;
            out_normal[2 * H * W + pix_id] = N.z;
            out_distort[pix_id] = accum_dist;
        }
    }
}

struct BlendInfo
{
    uint32_t id;
    float depth;

    __device__ BlendInfo() : id(0), depth(FLT_MAX) {}
    __device__ BlendInfo(uint32_t id_, float depth_) : id(id_), depth(depth_) {}
};

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderHybridCUDAResort(
    int W, int H, int C, int P_tri,
    float tri_gamma, float gau_gamma,
    float ambient_intensity,
    const float *__restrict__ light_color,
    const float *__restrict__ light_dir,
    float light_intensity,
    bool rich_info,
    bool tri_use_texture,
    bool tri_use_vertex_color,
    float tan_fovx, float tan_fovy,
    const uint2 *ranges,
    const uint32_t *point_list,
    const float3 *tri_v1,
    const float3 *tri_v2,
    const float3 *tri_v3,
    const float3 *tri_n,
    const float *tri_feature,
    const float2 *tri_uv,
    const float *tri_texture,
    int tri_tex_H,
    int tri_tex_W,
    const float *tri_opacity,
    const float2 *gau_xy,
    const float4 *gau_conic,
    const float *gau_feature,
    const float *gau_depth,
    const float *background,
    float background_depth,
    float *final_Ts,
    uint32_t *n_contribs,
    float *out_feature,
    float *out_depth,
    float *out_normal,
    float *out_distort,
    float *contrib_sum,
    float *contrib_max)
{
    auto block = cg::this_thread_block();
    dim3 group_index = block.group_index();
    dim3 thread_index = block.thread_index();

    uint2 pix = {group_index.x * BLOCK_X + thread_index.x, group_index.y * BLOCK_Y + thread_index.y};
    uint32_t pix_id = W * pix.y + pix.x;
    bool inside = pix.x < W && pix.y < H;
    bool done = !inside;

    float3 p_ray = {tan_fovx * pixToProj((float)pix.x, W), tan_fovy * pixToProj((float)pix.y, H), 1.0f};
    float3 light_dir_unit = {light_dir[0], light_dir[1], light_dir[2]};
    light_dir_unit = light_dir_unit / fmaxf(norm(light_dir_unit), EPS);
    float2 pixf = {(float)pix.x, (float)pix.y};

    uint2 range = ranges[group_index.y * ((W + BLOCK_X - 1) / BLOCK_X) + group_index.x];

    float T = 1.0f;
    uint32_t n_contrib = 0;
    float accum_feature[MAX_CHANNELS] = {0, 0, 0};
    float3 N = {0, 0, 0};
    float accum_depth = 0.0f;
    float accum_depth_sq = 0.0f;
    float accum_dist = 0.0f;

    auto sample_texture = [&](float u, float v, int ch)
    {
        float uu = fminf(fmaxf(u, 0.0f), 1.0f);
        float vv = fminf(fmaxf(v, 0.0f), 1.0f);
        float x = uu * (tri_tex_W - 1);
        float y = (1.0f - vv) * (tri_tex_H - 1);

        int x0 = (int)floorf(x);
        int y0 = (int)floorf(y);
        int x1 = min(x0 + 1, tri_tex_W - 1);
        int y1 = min(y0 + 1, tri_tex_H - 1);

        float tx = x - x0;
        float ty = y - y0;

        int i00 = (y0 * tri_tex_W + x0) * C + ch;
        int i10 = (y0 * tri_tex_W + x1) * C + ch;
        int i01 = (y1 * tri_tex_W + x0) * C + ch;
        int i11 = (y1 * tri_tex_W + x1) * C + ch;

        float c00 = tri_texture[i00];
        float c10 = tri_texture[i10];
        float c01 = tri_texture[i01];
        float c11 = tri_texture[i11];

        float c0 = c00 * (1.0f - tx) + c10 * tx;
        float c1 = c01 * (1.0f - tx) + c11 * tx;
        return c0 * (1.0f - ty) + c1 * ty;
    };

    BlendInfo sort_buffer[SORT_WINDOW_SIZE];
    for (int i = 0; i < SORT_WINDOW_SIZE; ++i)
    {
        sort_buffer[i].depth = FLT_MAX;
    }
    int sort_num = 0;

    auto blend_one = [&]()
    {
        if (sort_num <= 0)
            return;

        uint32_t id = sort_buffer[0].id;
        float depth = sort_buffer[0].depth;
        bool is_gauss = (id >> 31) != 0;

        float alpha = 0.0f;
        float contrib = 0.0f;
        float3 n = {0, 0, 0};

        if (!is_gauss)
        {
            int tri_id = (int)id;
            float3 v1 = tri_v1[tri_id];
            float3 v2 = tri_v2[tri_id];
            float3 v3 = tri_v3[tri_id];
            n = tri_n[tri_id];

            float3 p_view = depth * p_ray;
            float3 p_v1 = v1 - p_view;
            float3 p_v2 = v2 - p_view;
            float3 p_v3 = v3 - p_view;
            float inv_n_dot_n = 1.0f / dot(n, n);
            float a1 = dot(cross(p_v2, p_v3), n) * inv_n_dot_n;
            float a2 = dot(cross(p_v3, p_v1), n) * inv_n_dot_n;
            float a3 = 1.0f - a1 - a2;
            float ecc = 1.0f - 3.0f * min(min(a1, a2), a3);

            float G = tri_gamma > 50.0f ? (ecc <= 1.0f ? 1.0f : 0.0f) : expf(-0.5f * powf(ecc, 2.0f * tri_gamma));
            alpha = min(ALPHA_THRES, tri_opacity[tri_id] * G);
            contrib = alpha * T;

            float3 n_unit = n * sqrtf(inv_n_dot_n);
            float n_dot_l = fmaxf(dot(n_unit, light_dir_unit), 0.0f);

            for (int ch = 0; ch < C; ++ch)
            {
                float feat;
                if (tri_use_texture)
                {
                    float2 uv1 = tri_uv[tri_id * 3 + 0];
                    float2 uv2 = tri_uv[tri_id * 3 + 1];
                    float2 uv3 = tri_uv[tri_id * 3 + 2];
                    float u = uv1.x * a1 + uv2.x * a2 + uv3.x * a3;
                    float v = uv1.y * a1 + uv2.y * a2 + uv3.y * a3;
                    feat = sample_texture(u, v, ch);
                }
                else if (tri_use_vertex_color)
                {
                    feat = tri_feature[(tri_id * 3 + 0) * C + ch] * a1 +
                           tri_feature[(tri_id * 3 + 1) * C + ch] * a2 +
                           tri_feature[(tri_id * 3 + 2) * C + ch] * a3;
                }
                else
                {
                    feat = tri_feature[tri_id * C + ch];
                }
                float lighting = ambient_intensity + light_intensity * light_color[ch] * n_dot_l;
                accum_feature[ch] += feat * lighting * contrib;
            }

            n = n_unit;
            if (rich_info)
            {
                atomicAdd(&contrib_sum[tri_id], contrib);
                atomicMaxFloat(&contrib_max[tri_id], contrib);
            }
        }
        else
        {
            int gau_id = (int)(id & 0x7fffffff);
            float2 xy = gau_xy[gau_id];
            float2 d = {xy.x - pixf.x, xy.y - pixf.y};
            float4 co = gau_conic[gau_id];
            float q = co.x * d.x * d.x + co.z * d.y * d.y + 2.0f * co.y * d.x * d.y;
            float power = gau_gamma == 1.0f ? -0.5f * q : -0.5f * powf(q, gau_gamma);
            alpha = min(ALPHA_THRES, co.w * expf(power));
            contrib = alpha * T;
            depth = gau_depth[gau_id];

            for (int ch = 0; ch < C; ++ch)
            {
                accum_feature[ch] += gau_feature[gau_id * C + ch] * contrib;
            }

            if (rich_info)
            {
                atomicAdd(&contrib_sum[P_tri + gau_id], contrib);
                atomicMaxFloat(&contrib_max[P_tri + gau_id], contrib);
            }
        }

        n_contrib++;

        float test_T = T * (1.0f - alpha);
        if (test_T < T_THRES)
        {
            done = true;
        }
        else
        {
            if (rich_info)
            {
                N += n * contrib;
                accum_dist += (depth * depth * (1.0f - T) + accum_depth_sq - 2.0f * depth * accum_depth) * contrib;
                accum_depth += depth * contrib;
                accum_depth_sq += depth * depth * contrib;
            }
            T = test_T;
        }

        for (int i = 1; i < sort_num; ++i)
        {
            sort_buffer[i - 1] = sort_buffer[i];
        }
        sort_buffer[sort_num - 1].depth = FLT_MAX;
        --sort_num;
    };

    for (int i = range.x; !done && i < range.y; ++i)
    {
        uint32_t id = point_list[i];
        bool is_gauss = (id >> 31) != 0;
        float depth = 0.0f;

        if (!is_gauss)
        {
            int tri_id = (int)id;
            float3 v1 = tri_v1[tri_id];
            float3 v2 = tri_v2[tri_id];
            float3 v3 = tri_v3[tri_id];
            float3 n = tri_n[tri_id];

            float p_ray_dot_n = dot(p_ray, n);
            if (fabsf(p_ray_dot_n) < EPS)
                continue;
            depth = dot(v1, n) / p_ray_dot_n;
            if (depth < 0.0f)
                continue;

            float3 p_view = depth * p_ray;
            float3 p_v1 = v1 - p_view;
            float3 p_v2 = v2 - p_view;
            float3 p_v3 = v3 - p_view;
            float inv_n_dot_n = 1.0f / dot(n, n);
            float a1 = dot(cross(p_v2, p_v3), n) * inv_n_dot_n;
            float a2 = dot(cross(p_v3, p_v1), n) * inv_n_dot_n;
            float a3 = 1.0f - a1 - a2;
            float ecc = 1.0f - 3.0f * min(min(a1, a2), a3);
            if (ecc < 0.0f || ecc > 10.0f)
                continue;

            float G = tri_gamma > 50.0f ? (ecc <= 1.0f ? 1.0f : 0.0f) : expf(-0.5f * powf(ecc, 2.0f * tri_gamma));
            if (G < G_THRES)
                continue;
        }
        else
        {
            int gau_id = (int)(id & 0x7fffffff);
            float2 xy = gau_xy[gau_id];
            float2 d = {xy.x - pixf.x, xy.y - pixf.y};
            float4 co = gau_conic[gau_id];
            float q = co.x * d.x * d.x + co.z * d.y * d.y + 2.0f * co.y * d.x * d.y;
            float power = gau_gamma == 1.0f ? -0.5f * q : -0.5f * powf(q, gau_gamma);
            if (power > 0.0f)
                continue;

            float alpha = min(ALPHA_THRES, co.w * expf(power));
            if (alpha < G_THRES)
                continue;

            depth = gau_depth[gau_id];
        }

        BlendInfo new_sample(id, depth);
        for (int s = 0; s < SORT_WINDOW_SIZE && new_sample.depth != FLT_MAX; ++s)
        {
            if (new_sample.depth < sort_buffer[s].depth)
            {
                BlendInfo tmp = sort_buffer[s];
                sort_buffer[s] = new_sample;
                new_sample = tmp;
            }
        }
        ++sort_num;

        if (sort_num == SORT_WINDOW_SIZE)
            blend_one();
    }

    while (!done && sort_num > 0)
        blend_one();

    if (inside)
    {
        final_Ts[pix_id] = T;
        n_contribs[pix_id] = n_contrib;
        for (int ch = 0; ch < C; ++ch)
            out_feature[ch * H * W + pix_id] = accum_feature[ch] + T * background[ch];

        if (rich_info)
        {
            out_depth[pix_id] = accum_depth + T * background_depth;
            out_normal[pix_id] = N.x;
            out_normal[H * W + pix_id] = N.y;
            out_normal[2 * H * W + pix_id] = N.z;
            out_distort[pix_id] = accum_dist;
        }
    }
}

namespace FORWARD
{
void preprocessTriangles(
    int W, int H, int P, int D, int M,
    float gamma,
    bool use_shs,
    bool use_vertex_color,
    bool back_culling,
    dim3 grid,
    const float *viewmatrix,
    const float *projmatrix,
    const float *campos,
    const float *vertex,
    const float *shs,
    int *radii,
    float3 *v1_view,
    float3 *v2_view,
    float3 *v3_view,
    float3 *normal_view,
    float *depth,
    float *tri_rgb,
    bool *clamped,
    uint32_t *tiles_touched,
    uint2 *rect_min,
    uint2 *rect_max)
{
    if (P <= 0)
        return;

    preprocessTrianglesCUDA<<<(P + 255) / 256, 256>>>(
        W,
        H,
        P,
        D,
        M,
        gamma,
        use_shs,
        use_vertex_color,
        back_culling,
        grid,
        viewmatrix,
        projmatrix,
        campos,
        vertex,
        shs,
        radii,
        v1_view,
        v2_view,
        v3_view,
        normal_view,
        depth,
        tri_rgb,
        clamped,
        tiles_touched,
        rect_min,
        rect_max);
}

void preprocessGaussians(
    int W, int H, int P, int D, int M,
    float tan_fovx, float tan_fovy,
    float focal_x, float focal_y,
    float scale_modifier,
    bool prefiltered,
    bool use_shs,
    bool use_cov_precomp,
    dim3 grid,
    const float *viewmatrix,
    const float *projmatrix,
    const float *full_projmatrix,
    const float *campos,
    const float *means3D,
    const float *gaussian_shs,
    const float *gaussian_colors,
    const float *gaussian_opacity,
    const float *gaussian_scales,
    const float *gaussian_rotations,
    const float *gaussian_cov3D_precomp,
    int *radii,
    float2 *means2D,
    float *depths,
    float *cov3Ds,
    float4 *conic_opacity,
    float *feature,
    bool *clamped,
    uint32_t *tiles_touched,
    uint2 *rect_min,
    uint2 *rect_max)
{
    if (P <= 0)
        return;

    preprocessGaussiansCUDA<<<(P + 255) / 256, 256>>>(
        W,
        H,
        P,
        D,
        M,
        tan_fovx,
        tan_fovy,
        focal_x,
        focal_y,
        scale_modifier,
        prefiltered,
        use_shs,
        use_cov_precomp,
        grid,
        viewmatrix,
        projmatrix,
        full_projmatrix,
        campos,
        means3D,
        gaussian_shs,
        gaussian_colors,
        gaussian_opacity,
        gaussian_scales,
        gaussian_rotations,
        gaussian_cov3D_precomp,
        radii,
        means2D,
        depths,
        cov3Ds,
        conic_opacity,
        feature,
        clamped,
        tiles_touched,
        rect_min,
        rect_max);
}

void renderHybrid(
    dim3 grid,
    dim3 block,
    int W,
    int H,
    int C,
    int P_tri,
    float tri_gamma,
    float gau_gamma,
    float ambient_intensity,
    const float *light_color,
    const float *light_dir,
    float light_intensity,
    bool rich_info,
    bool tri_use_texture,
    bool tri_use_vertex_color,
    float tan_fovx,
    float tan_fovy,
    const uint2 *ranges,
    const uint32_t *point_list,
    const float3 *tri_v1,
    const float3 *tri_v2,
    const float3 *tri_v3,
    const float3 *tri_n,
    const float *tri_feature,
    const float2 *tri_uv,
    const float *tri_texture,
    int tri_tex_H,
    int tri_tex_W,
    const float *tri_opacity,
    const float2 *gau_xy,
    const float4 *gau_conic,
    const float *gau_feature,
    const float *gau_depth,
    const float *background,
    float background_depth,
    float *final_Ts,
    uint32_t *n_contribs,
    float *out_feature,
    float *out_depth,
    float *out_normal,
    float *out_distort,
    float *contrib_sum,
    float *contrib_max)
{
    renderHybridCUDA<<<grid, block>>>(
        W,
        H,
        C,
        P_tri,
        tri_gamma,
        gau_gamma,
        ambient_intensity,
        light_color,
        light_dir,
        light_intensity,
        rich_info,
        tri_use_texture,
        tri_use_vertex_color,
        tan_fovx,
        tan_fovy,
        ranges,
        point_list,
        tri_v1,
        tri_v2,
        tri_v3,
        tri_n,
        tri_feature,
        tri_uv,
        tri_texture,
        tri_tex_H,
        tri_tex_W,
        tri_opacity,
        gau_xy,
        gau_conic,
        gau_feature,
        gau_depth,
        background,
        background_depth,
        final_Ts,
        n_contribs,
        out_feature,
        out_depth,
        out_normal,
        out_distort,
        contrib_sum,
        contrib_max);
}

void renderHybridResort(
    dim3 grid,
    dim3 block,
    int W,
    int H,
    int C,
    int P_tri,
    float tri_gamma,
    float gau_gamma,
    float ambient_intensity,
    const float *light_color,
    const float *light_dir,
    float light_intensity,
    bool rich_info,
    bool tri_use_texture,
    bool tri_use_vertex_color,
    float tan_fovx,
    float tan_fovy,
    const uint2 *ranges,
    const uint32_t *point_list,
    const float3 *tri_v1,
    const float3 *tri_v2,
    const float3 *tri_v3,
    const float3 *tri_n,
    const float *tri_feature,
    const float2 *tri_uv,
    const float *tri_texture,
    int tri_tex_H,
    int tri_tex_W,
    const float *tri_opacity,
    const float2 *gau_xy,
    const float4 *gau_conic,
    const float *gau_feature,
    const float *gau_depth,
    const float *background,
    float background_depth,
    float *final_Ts,
    uint32_t *n_contribs,
    float *out_feature,
    float *out_depth,
    float *out_normal,
    float *out_distort,
    float *contrib_sum,
    float *contrib_max)
{
    renderHybridCUDAResort<<<grid, block>>>(
        W,
        H,
        C,
        P_tri,
        tri_gamma,
        gau_gamma,
        ambient_intensity,
        light_color,
        light_dir,
        light_intensity,
        rich_info,
        tri_use_texture,
        tri_use_vertex_color,
        tan_fovx,
        tan_fovy,
        ranges,
        point_list,
        tri_v1,
        tri_v2,
        tri_v3,
        tri_n,
        tri_feature,
        tri_uv,
        tri_texture,
        tri_tex_H,
        tri_tex_W,
        tri_opacity,
        gau_xy,
        gau_conic,
        gau_feature,
        gau_depth,
        background,
        background_depth,
        final_Ts,
        n_contribs,
        out_feature,
        out_depth,
        out_normal,
        out_distort,
        contrib_sum,
        contrib_max);
}
}
