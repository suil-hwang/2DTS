#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "forward.h"
#include "auxiliary.h"

// Forward method for converting the input spherical harmonics coefficients to RGB colors.
__device__ float3 computeRGBFromSH(int idx, int deg, int max_coeffs, float3 pos, float3 campos, const float *shs, bool *clamped)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
	float3 dir = pos - campos;
	dir /= norm(dir);

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
	rgb += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (rgb.x < 0);
	clamped[3 * idx + 1] = (rgb.y < 0);
	clamped[3 * idx + 2] = (rgb.z < 0);
	return max(rgb, 0.0f);
}

__global__ void FORWARD::preprocessCUDA(
	int W, int H, int P, int D, int M, float gamma, bool rich_info,
	bool use_shs, bool use_vertex_color, bool back_culling, dim3 grid,
	const float *__restrict__ viewmatrix,
	const float *__restrict__ projmatrix,
	const float *__restrict__ campos,
	const float *__restrict__ vertex,
	const float *__restrict__ shs,
	const float *__restrict__ opacity,
	int *__restrict__ radii,
	float3 *__restrict__ s_v1_view,
	float3 *__restrict__ s_v2_view,
	float3 *__restrict__ s_v3_view,
	float3 *__restrict__ s_normal_view,
	float *__restrict__ s_depth,
	float3 *__restrict__ s_rgb,
	bool *__restrict__ s_clamped,
	uint32_t *__restrict__ s_tiles_touched,
	uint2 *__restrict__ s_rect_min,
	uint2 *__restrict__ s_rect_max,
	float2 *__restrict__ s_support_2D)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	radii[idx] = 0;
	s_tiles_touched[idx] = 0;

	const float3 v1 = {vertex[9 * idx + 0], vertex[9 * idx + 1], vertex[9 * idx + 2]};
	const float3 v2 = {vertex[9 * idx + 3], vertex[9 * idx + 4], vertex[9 * idx + 5]};
	const float3 v3 = {vertex[9 * idx + 6], vertex[9 * idx + 7], vertex[9 * idx + 8]};
	const float3 v1_view = transformPoint4x3(v1, viewmatrix);
	const float3 v2_view = transformPoint4x3(v2, viewmatrix);
	const float3 v3_view = transformPoint4x3(v3, viewmatrix);

	const float3 center_view = (v1_view + v2_view + v3_view) / 3.0f;
	float3 normal_view = cross(v2_view - v1_view, v3_view - v1_view);

	if (norm(normal_view) < EPS) // Skip degenerate triangles
		return;

	// calculate coverage of the triangle in screen space
	const float dilation = supportEcc(gamma);
	float3 v1_dilated_view = center_view + dilation * (v1_view - center_view);
	float3 v2_dilated_view = center_view + dilation * (v2_view - center_view);
	float3 v3_dilated_view = center_view + dilation * (v3_view - center_view);
	float3 v1_dilated_proj = projectPoint(v1_dilated_view, projmatrix);
	float3 v2_dilated_proj = projectPoint(v2_dilated_view, projmatrix);
	float3 v3_dilated_proj = projectPoint(v3_dilated_view, projmatrix);

	if (v1_dilated_proj.z <= 0 && v2_dilated_proj.z <= 0 && v3_dilated_proj.z <= 0) // Near culling
		return;

	// A support crossing the camera plane can have an unbounded projection.
	const bool support_crosses_camera = v1_dilated_view.z <= EPS || v2_dilated_view.z <= EPS || v3_dilated_view.z <= EPS;

	if (back_culling)
	{
		const float facing = support_crosses_camera ? dot(v1_view, normal_view)
			: cross(v2_dilated_proj - v1_dilated_proj, v3_dilated_proj - v1_dilated_proj).z;
		if (facing >= 0)
			return;
	}

	const float2 v1_dilated_2D = {projToPix(v1_dilated_proj.x, W), projToPix(v1_dilated_proj.y, H)};
	const float2 v2_dilated_2D = {projToPix(v2_dilated_proj.x, W), projToPix(v2_dilated_proj.y, H)};
	const float2 v3_dilated_2D = {projToPix(v3_dilated_proj.x, W), projToPix(v3_dilated_proj.y, H)};

	float2 v_min = support_crosses_camera ? make_float2(-0.5f, -0.5f) : min(v1_dilated_2D, v2_dilated_2D, v3_dilated_2D);
	float2 v_max = support_crosses_camera ? make_float2((float)W - 0.5f, (float)H - 0.5f) : max(v1_dilated_2D, v2_dilated_2D, v3_dilated_2D);
	if (v_min.x >= ((float)W - 0.5f) || v_min.y >= ((float)H - 0.5f) || v_max.x < -0.5f || v_max.y < -0.5f)
		return;
	v_min = {min(max(v_min.x, -0.5f), (float)W - 0.5f), min(max(v_min.y, -0.5f), (float)H - 0.5f)};
	v_max = {min(max(v_max.x, -0.5f), (float)W - 0.5f), min(max(v_max.y, -0.5f), (float)H - 0.5f)};

	const uint2 rect_min = {min(grid.x, max((int)0, (int)((v_min.x + 0.5f) / BLOCK_X))),
							min(grid.y, max((int)0, (int)((v_min.y + 0.5f) / BLOCK_Y)))};
	const uint2 rect_max = {min(grid.x, max((int)0, (int)((v_max.x + 0.5f + BLOCK_X) / BLOCK_X))),
							min(grid.y, max((int)0, (int)((v_max.y + 0.5f + BLOCK_Y) / BLOCK_Y)))};
	if (rect_max.x <= rect_min.x || rect_max.y <= rect_min.y)
		return;

	// Compute RGB colors if SH coefficients are provided.
	if (use_shs)
	{
		if (use_vertex_color)
		{
			s_rgb[idx * 3 + 0] = computeRGBFromSH(idx * 3 + 0, D, M, v1, *(float3 *)campos, shs, s_clamped);
			s_rgb[idx * 3 + 1] = computeRGBFromSH(idx * 3 + 1, D, M, v2, *(float3 *)campos, shs, s_clamped);
			s_rgb[idx * 3 + 2] = computeRGBFromSH(idx * 3 + 2, D, M, v3, *(float3 *)campos, shs, s_clamped);
		}
		else
		{
			const float3 center = (v1 + v2 + v3) / 3.0f;
			s_rgb[idx] = computeRGBFromSH(idx, D, M, center, *(float3 *)campos, shs, s_clamped);
		}
	}

	// Store data for the following steps.
	s_v1_view[idx] = v1_view;
	s_v2_view[idx] = v2_view;
	s_v3_view[idx] = v3_view;
	s_normal_view[idx] = normal_view;
	s_depth[idx] = center_view.z;
	// Bin only into tiles the support reaches; every pixel of the other tiles would reject this triangle.
	const float2 support_2D[3] = {v1_dilated_2D, v2_dilated_2D, v3_dilated_2D};
	uint32_t tiles_touched = 0;
	for (uint32_t y = rect_min.y; y < rect_max.y; y++)
		for (uint32_t x = rect_min.x; x < rect_max.x; x++)
			tiles_touched += support_crosses_camera || triangleOverlapsTile(support_2D, x, y, W, H);
	s_tiles_touched[idx] = tiles_touched;
	for (int i = 0; i < 3; i++)
		s_support_2D[3 * idx + i] = support_2D[i];
	s_rect_min[idx] = rect_min;
	s_rect_max[idx] = rect_max;
	radii[idx] = max(ceil(v_max.x - v_min.x), ceil(v_max.y - v_min.y));
}

// Main rasterization method.
// Collaboratively works on one tile per block, each thread treats one pixel.
// Alternates between fetching and rasterizing data.
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	FORWARD::renderCUDA(
		int W, int H, int C, float gamma, bool rich_info, bool use_vertex_color, bool back_culling,
		float tan_fovx, float tan_fovy,
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		const float3 *__restrict__ s_v1_view,
		const float3 *__restrict__ s_v2_view,
		const float3 *__restrict__ s_v3_view,
		const float3 *__restrict__ s_normal_view,
		const float *__restrict__ feature,
		const float *__restrict__ opacity,
		const float background_depth,
		const float *__restrict__ background,
		float *__restrict__ final_Ts,
		uint32_t *__restrict__ n_contribs,
		float *__restrict__ out_feature,
		float *__restrict__ out_depth,
		float *__restrict__ out_normal,
		float *__restrict__ out_distort,
		float *__restrict__ contrib_sum,
		float *__restrict__ contrib_max,
		float2 *__restrict__ distortion_moments)
{
	// Identify current tile and pixel.
	auto block = cg::this_thread_block();
	dim3 group_index = block.group_index();
	dim3 thread_index = block.thread_index();
	auto tid = block.thread_rank();

	const uint2 pix = {group_index.x * BLOCK_X + thread_index.x, group_index.y * BLOCK_Y + thread_index.y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float3 p_ray = {tan_fovx * pixToProj((float)pix.x, W), tan_fovy * pixToProj((float)pix.y, H), 1.0f};
	const float ecc_max = supportEcc(gamma);

	const bool inside = pix.x < W && pix.y < H; // Check if this thread is associated with a valid pixel or outside.
	bool done = !inside;						// Done threads can help with fetching, but don't rasterize

	// Load start/end range of IDs to process in bit sorted list.
	const uint2 range = ranges[group_index.y * ((W + BLOCK_X - 1) / BLOCK_X) + group_index.x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_v1_view[BLOCK_SIZE];
	__shared__ float3 collected_v2_view[BLOCK_SIZE];
	__shared__ float3 collected_v3_view[BLOCK_SIZE];
	__shared__ float3 collected_normal_view[BLOCK_SIZE];
	__shared__ float collected_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t n_contrib = 0;

	float accum_feature[MAX_CHANNELS] = {0};
	float3 accum_normal = {0, 0, 0};
	float accum_depth = 0.0f;
	float accum_weight = 0.0f;
	float accum_depth_squared = 0.0f;
	float accum_distort = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if the entire block votes that it is done rasterizing
		const int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch data from global to shared
		block.sync();
		const int progress = i * BLOCK_SIZE + tid;
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[tid] = coll_id;
			collected_v1_view[tid] = s_v1_view[coll_id];
			collected_v2_view[tid] = s_v2_view[coll_id];
			collected_v3_view[tid] = s_v3_view[coll_id];
			collected_normal_view[tid] = s_normal_view[coll_id];
			collected_opacity[tid] = opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			n_contrib++;

			TriangleSample s;
			if (!sampleTriangle(p_ray, collected_v1_view[j], collected_v2_view[j], collected_v3_view[j], collected_normal_view[j], gamma, ecc_max, s))
				continue;

			const float alpha = min(ALPHA_THRES, collected_opacity[j] * s.G);
			const float contrib = alpha * T;
			const int global_id = collected_id[j];

			for (int ch = 0; ch < C; ch++)
			{
				const float feat = use_vertex_color ? (feature[global_id * 3 * C + ch] * s.a1 + feature[global_id * 3 * C + C + ch] * s.a2 + feature[global_id * 3 * C + 2 * C + ch] * s.a3)
													: feature[global_id * C + ch];
				accum_feature[ch] += feat * contrib;
			}

			if (rich_info)
			{
				atomicAdd(&contrib_sum[global_id], contrib);
				atomicMaxFloat(&contrib_max[global_id], contrib);

				const float depth = s.depth;
				accum_normal += collected_normal_view[j] * sqrt(s.inv_n_dot_n) * contrib;
				accum_distort += (depth * depth * accum_weight + accum_depth_squared - 2.0f * depth * accum_depth) * contrib;
				accum_weight += contrib;
				accum_depth += depth * contrib;
				accum_depth_squared += depth * depth * contrib;
			}

			T *= (1.0f - alpha);
			if (T <= T_THRES)
				done = true;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_Ts[pix_id] = T;
		n_contribs[pix_id] = n_contrib;
		for (int ch = 0; ch < C; ch++)
			out_feature[ch * H * W + pix_id] = accum_feature[ch] + T * background[ch];

		if (rich_info)
		{
			// Preserve raw moments before background addition or 1-T cancellation.
			distortion_moments[pix_id] = make_float2(accum_weight, accum_depth);
			out_distort[pix_id] = accum_distort;
			out_depth[pix_id] = accum_depth + T * background_depth;
			out_normal[pix_id] = accum_normal.x;
			out_normal[H * W + pix_id] = accum_normal.y;
			out_normal[2 * H * W + pix_id] = accum_normal.z;
		}
	}
}

__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	FORWARD::renderCUDAResort(
		int W, int H, int C, float gamma, bool rich_info, bool use_vertex_color, bool back_culling,
		float tan_fovx, float tan_fovy,
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		const float3 *__restrict__ s_v1_view,
		const float3 *__restrict__ s_v2_view,
		const float3 *__restrict__ s_v3_view,
		const float3 *__restrict__ s_normal_view,
		const float *__restrict__ feature,
		const float *__restrict__ opacity,
		const float background_depth,
		const float *__restrict__ background,
		float *__restrict__ final_Ts,
		uint32_t *__restrict__ n_contribs,
		float *__restrict__ out_feature,
		float *__restrict__ out_depth,
		float *__restrict__ out_normal,
		float *__restrict__ out_distort,
		float *__restrict__ contrib_sum,
		float *__restrict__ contrib_max,
		float2 *__restrict__ distortion_moments)
{
	// Identify current tile and pixel.
	auto block = cg::this_thread_block();
	dim3 group_index = block.group_index();
	dim3 thread_index = block.thread_index();
	auto tid = block.thread_rank();

	const uint2 pix = {group_index.x * BLOCK_X + thread_index.x, group_index.y * BLOCK_Y + thread_index.y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float3 p_ray = {tan_fovx * pixToProj((float)pix.x, W), tan_fovy * pixToProj((float)pix.y, H), 1.0f};
	const float ecc_max = supportEcc(gamma);

	const bool inside = pix.x < W && pix.y < H; // Check if this thread is associated with a valid pixel or outside.
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	const uint2 range = ranges[group_index.y * ((W + BLOCK_X - 1) / BLOCK_X) + group_index.x];

	// Initialize blending variables
	float T = 1.0f;
	float accum_feature[MAX_CHANNELS] = {0};
	float3 accum_normal = {0, 0, 0};
	float accum_depth = 0.0f;
	float accum_weight = 0.0f;
	float accum_depth_squared = 0.0f;
	float accum_distort = 0.0f;
	uint32_t n_contrib = 0;

	// Blends the nearest pending hit, recomputing its sample rather than keeping it in the window.
	SortWindow window;
	auto blend_nearest = [&]()
	{
		const int global_id = window.pop();
		const float3 normal_view = s_normal_view[global_id];
		TriangleSample s;
		sampleTriangle(p_ray, s_v1_view[global_id], s_v2_view[global_id], s_v3_view[global_id], normal_view, gamma, ecc_max, s);
		const float depth = s.depth;

		const float alpha = min(ALPHA_THRES, opacity[global_id] * s.G);
		const float contrib = alpha * T;
		n_contrib++;

		for (int ch = 0; ch < C; ch++)
		{
			const float feat = use_vertex_color ? (feature[global_id * 3 * C + ch] * s.a1 + feature[global_id * 3 * C + C + ch] * s.a2 + feature[global_id * 3 * C + 2 * C + ch] * s.a3)
													: feature[global_id * C + ch];
			accum_feature[ch] += feat * contrib;
		}

		if (rich_info)
		{
			atomicAdd(&contrib_sum[global_id], contrib);
			atomicMaxFloat(&contrib_max[global_id], contrib);

			accum_normal += normal_view * sqrt(s.inv_n_dot_n) * contrib;
			accum_distort += (depth * depth * accum_weight + accum_depth_squared - 2.0f * depth * accum_depth) * contrib;
			accum_weight += contrib;
			accum_depth += depth * contrib;
			accum_depth_squared += depth * depth * contrib;
		}

		T *= (1.0f - alpha);
		if (T <= T_THRES)
			done = true;
	};

	// Iterate over batches of candidates, collectively fetched into shared memory, until done or range is complete
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_v1_view[BLOCK_SIZE];
	__shared__ float3 collected_v2_view[BLOCK_SIZE];
	__shared__ float3 collected_v3_view[BLOCK_SIZE];
	__shared__ float3 collected_normal_view[BLOCK_SIZE];
	const int rounds = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int toDo = range.y - range.x;
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		if (__syncthreads_count(done) == BLOCK_SIZE)
			break;
		const int progress = i * BLOCK_SIZE + tid;
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.x + progress];
			collected_id[tid] = coll_id;
			collected_v1_view[tid] = s_v1_view[coll_id];
			collected_v2_view[tid] = s_v2_view[coll_id];
			collected_v3_view[tid] = s_v3_view[coll_id];
			collected_normal_view[tid] = s_normal_view[coll_id];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			TriangleSample s;
			if (!sampleTriangle(p_ray, collected_v1_view[j], collected_v2_view[j], collected_v3_view[j], collected_normal_view[j], gamma, ecc_max, s))
				continue;
			window.push(collected_id[j], s.depth);
			if (window.size == SORT_WINDOW_SIZE)
				blend_nearest();
		}
	}

	// Blend remaining samples in the window
	while (!done && window.size > 0)
		blend_nearest();

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_Ts[pix_id] = T;
		n_contribs[pix_id] = n_contrib;
		for (int ch = 0; ch < C; ch++)
			out_feature[ch * H * W + pix_id] = accum_feature[ch] + T * background[ch];

		if (rich_info)
		{
			// Preserve raw moments before background addition or 1-T cancellation.
			distortion_moments[pix_id] = make_float2(accum_weight, accum_depth);
			out_distort[pix_id] = accum_distort;
			out_depth[pix_id] = accum_depth + T * background_depth;
			out_normal[pix_id] = accum_normal.x;
			out_normal[H * W + pix_id] = accum_normal.y;
			out_normal[2 * H * W + pix_id] = accum_normal.z;
		}
	}
}
