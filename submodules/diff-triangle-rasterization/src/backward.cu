#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "backward.h"
#include "auxiliary.h"

// Backward method for converting the input spherical harmonics coefficients to RGB colors.
__device__ void computeRGBFromSHBackward(int idx, int deg, int max_coeffs, float3 pos, float3 campos, const float *shs, const bool *clamped,
										 const float3 *dL_dfeature, float3 *dL_dshs, float3 &dL_dpos)
{
	// Use PyTorch rule for clamping: if clamping was applied, gradient becomes 0.
	float3 dL_dRGB = dL_dfeature[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	float3 *dL_dsh = dL_dshs + idx * max_coeffs;
	dL_dsh[0] = SH_C0 * dL_dRGB;
	// Degree zero is direction-independent, including when pos == campos.
	if (deg == 0)
		return;

	const float3 dir_orig = pos - campos;
	const float3 dir = dir_orig / norm(dir_orig);
	const float3 *sh = ((const float3 *)shs) + idx * max_coeffs;

	float3 dRGBdx = {0, 0, 0};
	float3 dRGBdy = {0, 0, 0};
	float3 dRGBdz = {0, 0, 0};
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// No tricks here, just high school-level calculus.
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (SH_C3[0] * sh[9] * 3.f * 2.f * xy +
						   SH_C3[1] * sh[10] * yz +
						   SH_C3[2] * sh[11] * -2.f * xy +
						   SH_C3[3] * sh[12] * -3.f * 2.f * xz +
						   SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
						   SH_C3[5] * sh[14] * 2.f * xz +
						   SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (SH_C3[0] * sh[9] * 3.f * (xx - yy) +
						   SH_C3[1] * sh[10] * xz +
						   SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
						   SH_C3[3] * sh[12] * -3.f * 2.f * yz +
						   SH_C3[4] * sh[13] * -2.f * xy +
						   SH_C3[5] * sh[14] * -2.f * yz +
						   SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (SH_C3[1] * sh[10] * xy +
						   SH_C3[2] * sh[11] * 4.f * 2.f * yz +
						   SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
						   SH_C3[4] * sh[13] * 4.f * 2.f * xz +
						   SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation.
	// View direction is influenced by the triangle center,
	// so SHs gradients must propagate back into 3D position.
	float3 dL_ddir = {dot(dL_dRGB, dRGBdx), dot(dL_dRGB, dRGBdy), dot(dL_dRGB, dRGBdz)};

	// Account for normalization of direction
	dL_dpos += dnormvdv(dir_orig, dL_ddir);
}

__global__ void BACKWARD::preprocessCUDA(
	int P, int D, int M, bool use_shs, bool use_vertex_color,
	const float *__restrict__ viewmatrix,
	const float *__restrict__ campos,
	const float *__restrict__ vertex,
	const float *__restrict__ shs,
	const int *__restrict__ radii,
	const bool *__restrict__ clamped,
	const float4 *__restrict__ dL_dv1_view_ptr,
	const float4 *__restrict__ dL_dv2_view_ptr,
	const float4 *__restrict__ dL_dv3_view_ptr,
	const float *__restrict__ dL_dfeature,
	float *__restrict__ dL_dvertex,
	float *__restrict__ dL_dshs)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || radii[idx] <= 0)
		return;

	// Transform view-space vertex gradients back to world space.
	const float4 g1 = dL_dv1_view_ptr[idx], g2 = dL_dv2_view_ptr[idx], g3 = dL_dv3_view_ptr[idx];
	float3 vertex_gradients[3] = {
		transformVec4x3Transpose(make_float3(g1.x, g1.y, g1.z), viewmatrix),
		transformVec4x3Transpose(make_float3(g2.x, g2.y, g2.z), viewmatrix),
		transformVec4x3Transpose(make_float3(g3.x, g3.y, g3.z), viewmatrix)};

	if (use_shs)
	{
		const float3 *vertices = reinterpret_cast<const float3 *>(vertex) + 3 * idx;
		const float3 camera = *reinterpret_cast<const float3 *>(campos);
		const float3 *feature_gradients = reinterpret_cast<const float3 *>(dL_dfeature);
		float3 *sh_gradients = reinterpret_cast<float3 *>(dL_dshs);
		if (use_vertex_color)
		{
			// Each vertex receives the gradient of its own SH color.
			for (int i = 0; i < 3; ++i)
				computeRGBFromSHBackward(3 * idx + i, D, M, vertices[i], camera, shs, clamped, feature_gradients, sh_gradients, vertex_gradients[i]);
		}
		else
		{
			// Distribute the face-centroid gradient equally among the three vertices.
			const float3 center = (vertices[0] + vertices[1] + vertices[2]) / 3.0f;
			float3 dL_dcenter_sh = {0, 0, 0};
			computeRGBFromSHBackward(idx, D, M, center, camera, shs, clamped, feature_gradients, sh_gradients, dL_dcenter_sh);
			for (int i = 0; i < 3; ++i)
				vertex_gradients[i] += dL_dcenter_sh / 3.0f;
		}
	}

	// Each thread owns one triangle, so these stores need no atomics.
	float3 *output = reinterpret_cast<float3 *>(dL_dvertex) + 3 * idx;
	for (int i = 0; i < 3; ++i)
		output[i] = vertex_gradients[i];
}

__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	BACKWARD::renderCUDA(
		int W, int H, int C, float gamma, bool rich_info, bool use_vertex_color,
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
		const float *__restrict__ final_Ts,
		const uint32_t *__restrict__ n_contribs,
		const float2 *__restrict__ distortion_moments,
		const float *__restrict__ distortion_image,
		const float *__restrict__ dL_dout_feature,
		const float *__restrict__ dL_dout_depth,
		const float *__restrict__ dL_dout_normal,
		const float *__restrict__ dL_dout_distortion,
		float4 *__restrict__ dL_dv1_view,
		float4 *__restrict__ dL_dv2_view,
		float4 *__restrict__ dL_dv3_view,
		float *__restrict__ dL_dfeature,
		float *__restrict__ dL_dopacity,
		float *__restrict__ dL_dv_norm)
{
	auto block = cg::this_thread_block();
	dim3 group_index = block.group_index();
	dim3 thread_index = block.thread_index();
	auto tid = block.thread_rank();

	const uint2 pix = {group_index.x * BLOCK_X + thread_index.x, group_index.y * BLOCK_Y + thread_index.y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float3 p_ray = {tan_fovx * pixToProj((float)pix.x, W), tan_fovy * pixToProj((float)pix.y, H), 1.0f};
	const float ecc_max = supportEcc(gamma);

	const bool inside = pix.x < W && pix.y < H;

	const uint2 range = ranges[group_index.y * ((W + BLOCK_X - 1) / BLOCK_X) + group_index.x];
	const uint32_t last_contributor = inside ? n_contribs[pix_id] : 0;
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_v1_view[BLOCK_SIZE];
	__shared__ float3 collected_v2_view[BLOCK_SIZE];
	__shared__ float3 collected_v3_view[BLOCK_SIZE];
	__shared__ float3 collected_normal_view[BLOCK_SIZE];
	__shared__ float collected_opacity[BLOCK_SIZE];

	const float final_T = inside ? final_Ts[pix_id] : 0;
	float final_weight = 0;
	float T = final_T;

	float accum_feature[MAX_CHANNELS] = {0}; // Accumulated feature from back to front
	float3 accum_normal = {0, 0, 0};		 // Accumulated normal from back to front
	float accum_depth = background_depth;	 // Accumulated depth from back to front
	float accum_distortion_grad = 0;		 // Accumulated distortion weight derivative from back to front

	float dL_dfeature_pixel[MAX_CHANNELS] = {0};
	float3 dL_dnormal_pixel = {0, 0, 0};
	float dL_ddepth_pixel = 0;
	float dL_ddistortion_pixel = 0;
	float depth_moment = 0;
	float mean_depth = 0;
	float distortion_per_weight = 0;

	if (inside)
	{
		for (int i = 0; i < C; i++)
		{
			accum_feature[i] = background[i];
			dL_dfeature_pixel[i] = dL_dout_feature[i * H * W + pix_id];
		}
		if (rich_info)
		{
			dL_dnormal_pixel = make_float3(dL_dout_normal[pix_id], dL_dout_normal[W * H + pix_id], dL_dout_normal[2 * W * H + pix_id]);
			dL_ddepth_pixel = dL_dout_depth[pix_id];
			dL_ddistortion_pixel = dL_dout_distortion[pix_id];
			const float2 moments = distortion_moments[pix_id];
			final_weight = moments.x;
			depth_moment = moments.y;
			if (final_weight > 0.0f)
			{
				mean_depth = depth_moment / final_weight;
				distortion_per_weight = distortion_image[pix_id] / final_weight;
			}
		}
	}

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		const int batch_size = min(BLOCK_SIZE, toDo);
		// Synchronize the previous batch and skip triangles no pixel visited.
		if (__syncthreads_and(toDo - batch_size >= last_contributor))
			continue;

		// Load the next batch in reverse order.
		const int progress = i * BLOCK_SIZE + tid;
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[tid] = coll_id;
			collected_v1_view[tid] = s_v1_view[coll_id];
			collected_v2_view[tid] = s_v2_view[coll_id];
			collected_v3_view[tid] = s_v3_view[coll_id];
			collected_normal_view[tid] = s_normal_view[coll_id];
			collected_opacity[tid] = opacity[coll_id];
		}
		block.sync();

		// Iterate over triangles in the batch
		for (int j = 0; inside && j < batch_size; j++)
		{
			// Skip triangles beyond this pixel's visited prefix.
			if (toDo - j > last_contributor)
				continue;

			// Compute blending values, as before.
			const float3 v1_view = collected_v1_view[j];
			const float3 v2_view = collected_v2_view[j];
			const float3 v3_view = collected_v3_view[j];
			const float3 normal_view = collected_normal_view[j];
			TriangleSample s;
			if (!sampleTriangle(p_ray, v1_view, v2_view, v3_view, normal_view, gamma, ecc_max, s))
				continue;
			const auto [p_v1, p_v2, p_v3, inv_p_ray_dot_n, inv_n_dot_n, depth, a1, a2, a3, ecc, power, G] = s;

			const float op = collected_opacity[j];
			const float alpha = min(ALPHA_THRES, op * G);

			T /= (1.0f - alpha);
			const float contrib = alpha * T;

			// Propagate gradients to per-triangle feature and alpha
			float dL_dcontrib = 0.0f;
			float3 dL_dnormal = {0, 0, 0};
			float dL_ddepth = 0.0f;
			float3 dL_da = {0, 0, 0};

			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float dL_dfeat = dL_dfeature_pixel[ch];
				float feat;
				if (use_vertex_color)
				{
					const float feat1 = feature[global_id * 3 * C + 0 * C + ch];
					const float feat2 = feature[global_id * 3 * C + 1 * C + ch];
					const float feat3 = feature[global_id * 3 * C + 2 * C + ch];
					feat = feat1 * a1 + feat2 * a2 + feat3 * a3;
					atomicAdd(&(dL_dfeature[global_id * 3 * C + 0 * C + ch]), dL_dfeat * a1 * contrib);
					atomicAdd(&(dL_dfeature[global_id * 3 * C + 1 * C + ch]), dL_dfeat * a2 * contrib);
					atomicAdd(&(dL_dfeature[global_id * 3 * C + 2 * C + ch]), dL_dfeat * a3 * contrib);
					dL_da.x += dL_dfeat * contrib * feat1;
					dL_da.y += dL_dfeat * contrib * feat2;
					dL_da.z += dL_dfeat * contrib * feat3;
				}
				else
				{
					atomicAdd(&(dL_dfeature[global_id * C + ch]), dL_dfeat * contrib);
					feat = feature[global_id * C + ch];
				}
				dL_dcontrib += dL_dfeat * (feat - accum_feature[ch]);
				accum_feature[ch] = alpha * feat + (1.0f - alpha) * accum_feature[ch];
			}

			if (rich_info)
			{
				const float3 normalized_normal_view = normal_view * sqrt(inv_n_dot_n);
				dL_dnormal += dnormvdv(normal_view, dL_dnormal_pixel * contrib);
				dL_dcontrib += dot(dL_dnormal_pixel, normalized_normal_view - accum_normal);
				accum_normal = alpha * normalized_normal_view + (1.0f - alpha) * accum_normal;

				dL_ddepth += dL_ddepth_pixel * contrib;
				dL_dcontrib += dL_ddepth_pixel * (depth - accum_depth);
				accum_depth = alpha * depth + (1.0f - alpha) * accum_depth;

				// D = S * sum(w*z*z) - sum(w*z)^2, so dD/dw = S*(z-mean)^2 + D/S.
				const float depth_delta = depth - mean_depth;
				const float distortion_weight_grad = final_weight * depth_delta * depth_delta + distortion_per_weight;
				dL_ddepth += dL_ddistortion_pixel * contrib * 2.0f * (final_weight * depth - depth_moment);
				dL_dcontrib += dL_ddistortion_pixel * (distortion_weight_grad - accum_distortion_grad);
				accum_distortion_grad = alpha * distortion_weight_grad + (1.0f - alpha) * accum_distortion_grad;
			}

			const float dL_dalpha = dL_dcontrib * T;
			const float dL_dpower = (op * G < ALPHA_THRES) ? (dL_dalpha * alpha) : 0.0f; // if not clamped by min(0.99, op * G)
			const float dL_decc = dL_dpower * 2 * gamma * power / (ecc + EPS);

			float3 decc_da = {0, 0, 0};
			if (a1 <= a2 && a1 <= a3)
			{
				decc_da.x = -3.0f;
			}
			else if (a2 <= a1 && a2 <= a3)
			{
				decc_da.y = -3.0f;
			}
			else
			{
				decc_da.z = -3.0f;
			}
			dL_da += dL_decc * decc_da;

			// a3 = 1 - a1 - a2, so combine its adjoint into the two independent coordinates.
			const float g1 = dL_da.x - dL_da.z;
			const float g2 = dL_da.y - dL_da.z;

			const float3 da1_dv2_view = cross(p_v3, normal_view) * inv_n_dot_n;
			const float3 da1_dv3_view = cross(normal_view, p_v2) * inv_n_dot_n;
			const float3 da1_dnormal_view = (cross(p_v2, p_v3) - 2.0f * a1 * normal_view) * inv_n_dot_n;
			const float da1_ddepth = dot(normal_view, cross(v3_view - v2_view, p_ray)) * inv_n_dot_n;

			const float3 da2_dv1_view = cross(normal_view, p_v3) * inv_n_dot_n;
			const float3 da2_dv3_view = cross(p_v1, normal_view) * inv_n_dot_n;
			const float3 da2_dnormal_view = (cross(p_v3, p_v1) - 2.0f * a2 * normal_view) * inv_n_dot_n;
			const float da2_ddepth = dot(normal_view, cross(v1_view - v3_view, p_ray)) * inv_n_dot_n;

			dL_ddepth += g1 * da1_ddepth + g2 * da2_ddepth;
			const float3 ddepth_dv1_view = normal_view * inv_p_ray_dot_n;
			const float3 ddepth_dnormal_view = p_v1 * inv_p_ray_dot_n;

			float3 dL_dv1_view_point = g2 * da2_dv1_view + dL_ddepth * ddepth_dv1_view;
			float3 dL_dv2_view_point = g1 * da1_dv2_view;
			float3 dL_dv3_view_point = g1 * da1_dv3_view + g2 * da2_dv3_view;
			dL_dnormal += g1 * da1_dnormal_view + g2 * da2_dnormal_view + dL_ddepth * ddepth_dnormal_view;

			dL_dv1_view_point += cross(v2_view - v3_view, dL_dnormal);
			dL_dv2_view_point += cross(v3_view - v1_view, dL_dnormal);
			dL_dv3_view_point += cross(v1_view - v2_view, dL_dnormal);

			// Update gradients w.r.t. triangle vertex positions
			atomicAddVec(&dL_dv1_view[global_id], dL_dv1_view_point);
			atomicAddVec(&dL_dv2_view[global_id], dL_dv2_view_point);
			atomicAddVec(&dL_dv3_view[global_id], dL_dv3_view_point);

			// Update gradients for densification
			const float dL_dv_norm_point = (norm(dL_dv1_view_point) * v1_view.z + norm(dL_dv2_view_point) * v2_view.z + norm(dL_dv3_view_point) * v3_view.z) / 3.0f;
			atomicAdd(&dL_dv_norm[global_id], dL_dv_norm_point);

			// Differentiate the renderer alpha cap. The upstream binary-opacity STE is separate.
			if (op * G < ALPHA_THRES)
				atomicAdd(&dL_dopacity[global_id], dL_dalpha * G);
		}
	}
}

__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	BACKWARD::renderCUDAResort(
		int W, int H, int C, float gamma, bool rich_info, bool use_vertex_color,
		float tan_fovx, float tan_fovy,
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		const float3 *__restrict__ s_v1_view,
		const float3 *__restrict__ s_v2_view,
		const float3 *__restrict__ s_v3_view,
		const float3 *__restrict__ s_normal_view,
		const float *__restrict__ feature,
		const float *__restrict__ opacity,
		const float *__restrict__ final_features,
		const float *__restrict__ final_normals,
		const float *__restrict__ final_depths,
		const float *__restrict__ final_distorts,
		const float2 *__restrict__ distortion_moments,
		const float *__restrict__ dL_dout_feature,
		const float *__restrict__ dL_dout_depth,
		const float *__restrict__ dL_dout_normal,
		const float *__restrict__ dL_dout_distortion,
		float4 *__restrict__ dL_dv1_view,
		float4 *__restrict__ dL_dv2_view,
		float4 *__restrict__ dL_dv3_view,
		float *__restrict__ dL_dfeature,
		float *__restrict__ dL_dopacity,
		float *__restrict__ dL_dv_norm)
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

	// Threads outside the image still help fetch candidates.
	const bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	const uint2 range = ranges[group_index.y * ((W + BLOCK_X - 1) / BLOCK_X) + group_index.x];

	// Blending state. rest_* is what the hits not yet blended add to each output:
	// the final output minus the front-to-back prefix blended so far.
	float T = 1.0f;
	float rest_feature[MAX_CHANNELS] = {0};
	float3 rest_normal = {0, 0, 0};
	float rest_depth = 0;
	float rest_distortion_grad = 0; // of sum(w * dD/dw) = 2 * D

	float final_weight = 0;
	float depth_moment = 0; // Final weighted depth for this pixel (without background)
	float mean_depth = 0;
	float distortion_per_weight = 0;

	// Gradients of loss w.r.t. pixel outputs
	float dL_dfeature_pixel[MAX_CHANNELS] = {0};
	float3 dL_dnormal_pixel = {0, 0, 0};
	float dL_ddepth_pixel = 0;
	float dL_ddistortion_pixel = 0;

	for (int i = 0; inside && i < C; i++)
	{
		rest_feature[i] = final_features[i * H * W + pix_id];
		dL_dfeature_pixel[i] = dL_dout_feature[i * H * W + pix_id];
	}
	if (inside && rich_info)
	{
		rest_depth = final_depths[pix_id];
		const float2 moments = distortion_moments[pix_id];
		final_weight = moments.x;
		depth_moment = moments.y;
		rest_normal = make_float3(final_normals[pix_id], final_normals[W * H + pix_id], final_normals[2 * W * H + pix_id]);
		const float final_distort = final_distorts[pix_id];
		rest_distortion_grad = 2.0f * final_distort;
		if (final_weight > 0.0f)
		{
			mean_depth = depth_moment / final_weight;
			distortion_per_weight = final_distort / final_weight;
		}
		dL_dnormal_pixel = make_float3(dL_dout_normal[pix_id], dL_dout_normal[W * H + pix_id], dL_dout_normal[2 * W * H + pix_id]);
		dL_ddepth_pixel = dL_dout_depth[pix_id];
		dL_ddistortion_pixel = dL_dout_distortion[pix_id];
	}

	// Blends the nearest pending hit, recomputing its sample rather than keeping it in the window.
	SortWindow window;
	auto blend_nearest = [&]()
	{
		const int global_id = window.pop();
		const float3 v1_view = s_v1_view[global_id];
		const float3 v2_view = s_v2_view[global_id];
		const float3 v3_view = s_v3_view[global_id];
		const float3 normal_view = s_normal_view[global_id];
		TriangleSample s;
		sampleTriangle(p_ray, v1_view, v2_view, v3_view, normal_view, gamma, ecc_max, s);
		const auto [p_v1, p_v2, p_v3, inv_p_ray_dot_n, inv_n_dot_n, depth, a1, a2, a3, ecc, power, G] = s;

		const float op = opacity[global_id];
		const float alpha = min(ALPHA_THRES, op * G);

		const float contrib = alpha * T;
		const float test_T = T * (1.0f - alpha);

		const float dalpha_dpower = (op * G < ALPHA_THRES) ? alpha : 0.0f; // if not clamped by min(0.99, op * G)
		const float dalpha_decc = dalpha_dpower * 2 * gamma * power / (ecc + EPS);

		// Propagate gradients to per-triangle parameters
		float dL_dcontrib = 0.0f;
		float3 dL_dnormal = {0, 0, 0};
		float dL_ddepth = 0.0f;
		float3 dL_da = {0, 0, 0};

		for (int ch = 0; ch < C; ch++)
		{
			const float dL_dfeat = dL_dfeature_pixel[ch];
			float feat;
			if (use_vertex_color)
			{
				const float feat1 = feature[global_id * 3 * C + 0 * C + ch];
				const float feat2 = feature[global_id * 3 * C + 1 * C + ch];
				const float feat3 = feature[global_id * 3 * C + 2 * C + ch];
				feat = feat1 * a1 + feat2 * a2 + feat3 * a3;
				atomicAdd(&(dL_dfeature[global_id * 3 * C + 0 * C + ch]), dL_dfeat * a1 * contrib);
				atomicAdd(&(dL_dfeature[global_id * 3 * C + 1 * C + ch]), dL_dfeat * a2 * contrib);
				atomicAdd(&(dL_dfeature[global_id * 3 * C + 2 * C + ch]), dL_dfeat * a3 * contrib);
				dL_da.x += dL_dfeat * contrib * feat1;
				dL_da.y += dL_dfeat * contrib * feat2;
				dL_da.z += dL_dfeat * contrib * feat3;
			}
			else
			{
				atomicAdd(&(dL_dfeature[global_id * C + ch]), dL_dfeat * contrib);
				feat = feature[global_id * C + ch];
			}

			rest_feature[ch] -= feat * contrib;
			dL_dcontrib += dL_dfeat * (feat - rest_feature[ch] / test_T);
		}

		if (rich_info)
		{
			const float3 normal = normal_view * sqrt(inv_n_dot_n); // normalized normal

			rest_normal = rest_normal - normal * contrib;
			rest_depth -= depth * contrib;
			// D = S * sum(w*z*z) - sum(w*z)^2, so dD/dw = S*(z-mean)^2 + D/S.
			const float depth_delta = depth - mean_depth;
			const float distortion_weight_grad = final_weight * depth_delta * depth_delta + distortion_per_weight;
			rest_distortion_grad -= distortion_weight_grad * contrib;

			// Divided by test_T, the rest is the blend of the hits behind this one.
			const float3 accum_normal_back = rest_normal / test_T;
			const float accum_depth_back = rest_depth / test_T;
			const float accum_distortion_grad_back = rest_distortion_grad / test_T;

			dL_dnormal += dnormvdv(normal_view, dL_dnormal_pixel * contrib); // dnormvdv expects unnormalized input
			dL_dcontrib += dot(dL_dnormal_pixel, normal - accum_normal_back);

			dL_ddepth += dL_ddepth_pixel * contrib;
			dL_dcontrib += dL_ddepth_pixel * (depth - accum_depth_back);

			dL_ddepth += dL_ddistortion_pixel * contrib * 2.0f * (final_weight * depth - depth_moment);
			dL_dcontrib += dL_ddistortion_pixel * (distortion_weight_grad - accum_distortion_grad_back);
		}

		const float dL_dalpha = dL_dcontrib * T;
		const float dL_decc = dL_dalpha * dalpha_decc;

		float3 decc_da = {0, 0, 0};
		if (a1 <= a2 && a1 <= a3)
		{
			decc_da.x = -3.0f;
		}
		else if (a2 <= a1 && a2 <= a3)
		{
			decc_da.y = -3.0f;
		}
		else
		{
			decc_da.z = -3.0f;
		}
		dL_da += dL_decc * decc_da;

		// a3 = 1 - a1 - a2, so combine its adjoint into the two independent coordinates.
		const float g1 = dL_da.x - dL_da.z;
		const float g2 = dL_da.y - dL_da.z;

		const float3 da1_dv2_view = cross(p_v3, normal_view) * inv_n_dot_n;
		const float3 da1_dv3_view = cross(normal_view, p_v2) * inv_n_dot_n;
		const float3 da1_dnormal_view = (cross(p_v2, p_v3) - 2.0f * a1 * normal_view) * inv_n_dot_n;
		const float da1_ddepth = dot(normal_view, cross(v3_view - v2_view, p_ray)) * inv_n_dot_n;

		const float3 da2_dv1_view = cross(normal_view, p_v3) * inv_n_dot_n;
		const float3 da2_dv3_view = cross(p_v1, normal_view) * inv_n_dot_n;
		const float3 da2_dnormal_view = (cross(p_v3, p_v1) - 2.0f * a2 * normal_view) * inv_n_dot_n;
		const float da2_ddepth = dot(normal_view, cross(v1_view - v3_view, p_ray)) * inv_n_dot_n;

		dL_ddepth += g1 * da1_ddepth + g2 * da2_ddepth;
		const float3 ddepth_dv1_view = normal_view * inv_p_ray_dot_n;
		const float3 ddepth_dnormal_view = p_v1 * inv_p_ray_dot_n;

		float3 dL_dv1_view_point = g2 * da2_dv1_view + dL_ddepth * ddepth_dv1_view;
		float3 dL_dv2_view_point = g1 * da1_dv2_view;
		float3 dL_dv3_view_point = g1 * da1_dv3_view + g2 * da2_dv3_view;
		dL_dnormal += g1 * da1_dnormal_view + g2 * da2_dnormal_view + dL_ddepth * ddepth_dnormal_view;

		dL_dv1_view_point += cross(v2_view - v3_view, dL_dnormal);
		dL_dv2_view_point += cross(v3_view - v1_view, dL_dnormal);
		dL_dv3_view_point += cross(v1_view - v2_view, dL_dnormal);

		// Update gradients w.r.t. triangle vertex positions
		atomicAddVec(&dL_dv1_view[global_id], dL_dv1_view_point);
		atomicAddVec(&dL_dv2_view[global_id], dL_dv2_view_point);
		atomicAddVec(&dL_dv3_view[global_id], dL_dv3_view_point);

		// Update gradients for densification
		const float dL_dv_norm_point = (norm(dL_dv1_view_point) * v1_view.z + norm(dL_dv2_view_point) * v2_view.z + norm(dL_dv3_view_point) * v3_view.z) / 3.0f;
		atomicAdd(&dL_dv_norm[global_id], dL_dv_norm_point);

		// Differentiate the renderer alpha cap. The upstream binary-opacity STE is separate.
		if (op * G < ALPHA_THRES)
			atomicAdd(&dL_dopacity[global_id], dL_dalpha * G);

		T *= (1.0f - alpha);
		if (T <= T_THRES)
			done = true;
	};

	// Replay the forward pass's candidate loop, batch by batch through shared memory
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
}
