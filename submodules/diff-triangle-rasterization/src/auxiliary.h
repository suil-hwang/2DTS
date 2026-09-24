#pragma once

#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include "stdio.h"
#include "config.h"

#define EPS (float)(1e-8)
#define FLT_INF std::numeric_limits<float>::infinity()

// Spherical harmonics coefficients
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

template <typename T>
__forceinline__ __device__ void swap(T &a, T &b)
{
	T temp = a;
	a = b;
	b = temp;
}

__forceinline__ __device__ float projToPix(float v, int S)
{
	return (v + 1.0f) * S * 0.5f - 0.5f;
}

__forceinline__ __device__ float pixToProj(float v, int S)
{
	return (2.0f * v - S + 1.0f) / (float)(S);
}

__forceinline__ __device__ float3 transformPoint4x3(const float3 &p, const float *matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3 &p, const float *matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]};
	return transformed;
}

__forceinline__ __device__ float3 transformPoint4x4Transpose(const float4 &p, const float *matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3] * p.w,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7] * p.w,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11] * p.w};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3 &p, const float *matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3 &p, const float *matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 projectPoint(const float3 &p, const float *projmatrix)
{
	float4 p_hom = transformPoint4x4(p, projmatrix);
	float p_w_inv = 1.0f / (abs(p_hom.w) + EPS); // use abs to make sure p_proj.z for points behind the camera are negative
	float3 p_proj = {p_hom.x * p_w_inv, p_hom.y * p_w_inv, p_hom.z * p_w_inv};
	return p_proj;
}

__forceinline__ __device__ float2 projectVecApprox(const float3 &p_view, const float3 &vec_view, float tan_fovx, float tan_fovy)
{
	/*
		p_view and vec_view are view-space origin and vector.
		We will use a linear expansion of the projection function at p to project the vector.

		We will use the following formulas:
		---------------------------
		p_view = (x, y, z)
		vec_view = (dx, dy, dz)
		p_proj = (x/z/tan_fovx, y/z/tan_fovy)
		vec_proj = (d(x_proj), d(y_proj))
		d(x_proj) = d(x/z/tan_fovx) = dx/z/tan_fovx - dz*x/z^2/tan_fovx = (dx - dz*x/z)/z/tan_fovx
		d(y_proj) = d(y/z/tan_fovy) = dy/z/tan_fovy - dz*y/z^2/tan_fovy = (dy - dz*y/z)/z/tan_fovy
		---------------------------

		x/z and y/z should be clipped by tan_fovx and tan_fovy to avoid large distortions.
	*/
	float2 vec_proj = {(vec_view.x - vec_view.z * p_view.x / p_view.z) / (p_view.z * tan_fovx),
					   (vec_view.y - vec_view.z * p_view.y / p_view.z) / (p_view.z * tan_fovy)};
	return vec_proj;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float2 dnormvdv(float2 v, float2 dv)
{
	float sum2 = v.x * v.x + v.y * v.y;
	float normv = sqrt(sum2);
	float invsum32 = 1.0f / (normv * normv * normv);

	float2 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float normv = sqrt(sum2);
	float invsum32 = 1.0f / (normv * normv * normv);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float normv = sqrt(sum2);
	float invsum32 = 1.0f / (normv * normv * normv);

	float4 vdv = {v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w};
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ float cross(const float2 &a, const float2 &b)
{
	return a.x * b.y - a.y * b.x;
}

__forceinline__ __device__ float3 cross(const float3 &a, const float3 &b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__forceinline__ __device__ float2 cross(const float2 &a)
{
	return make_float2(a.y, -a.x);
}

__forceinline__ __device__ float2 operator+(const float2 &a, const float2 &b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}

__forceinline__ __device__ float2 operator+(const float2 &a, float b)
{
	return make_float2(a.x + b, a.y + b);
}

__forceinline__ __device__ float2 operator-(const float2 &a, const float2 &b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}

__forceinline__ __device__ float2 operator*(const float2 &a, const float2 &b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}

__forceinline__ __device__ float2 operator*(float a, const float2 &b)
{
	return make_float2(a * b.x, a * b.y);
}

__forceinline__ __device__ float2 operator*(const float2 &a, float b)
{
	return make_float2(a.x * b, a.y * b);
}

__forceinline__ __device__ float2 operator/(const float2 &a, const float2 &b)
{
	return make_float2(a.x / b.x, a.y / b.y);
}

__forceinline__ __device__ float2 operator/(const float2 &a, float b)
{
	return make_float2(a.x / b, a.y / b);
}

__forceinline__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__ float3 operator+(const float3 &a, float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}

__forceinline__ __device__ void operator+=(float3 &a, const float3 &b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

__forceinline__ __device__ void operator+=(float3 &a, float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

__forceinline__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __device__ float3 operator-(const float3 &a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

__forceinline__ __device__ float3 operator*(const float3 &a, const float3 &b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __device__ float3 operator/(const float3 &a, const float3 &b)
{
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__forceinline__ __device__ void operator/=(float3 &a, float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}

__forceinline__ __device__ float3 operator*(float a, const float3 &b)
{
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__forceinline__ __device__ float3 operator*(const float3 &a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__forceinline__ __device__ float3 operator/(const float3 &a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}

__forceinline__ __device__ float4 operator*(float a, const float4 &b)
{
	return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

__forceinline__ __device__ float4 operator/(const float4 &a, float b)
{
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

__forceinline__ __device__ float dot(const float2 &a, const float2 &b)
{
	return a.x * b.x + a.y * b.y;
}

__forceinline__ __device__ float dot(const float3 &a, const float3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ float dot(const float4 &a, const float4 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__forceinline__ __device__ float norm(const float2 &a)
{
	return sqrt(dot(a, a));
}

__forceinline__ __device__ float norm(const float3 &a)
{
	return sqrt(dot(a, a));
}

__forceinline__ __device__ float2 min(const float2 &a, const float2 &b)
{
	return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

__forceinline__ __device__ float2 min(const float2 &a, const float2 &b, const float2 &c)
{
	return min(min(a, b), c);
}

__forceinline__ __device__ float3 min(const float3 &a, float b)
{
	return make_float3(fminf(a.x, b), fminf(a.y, b), fminf(a.z, b));
}

__forceinline__ __device__ float2 max(const float2 &a, const float2 &b)
{
	return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

__forceinline__ __device__ float2 max(const float2 &a, const float2 &b, const float2 &c)
{
	return max(max(a, b), c);
}

__forceinline__ __device__ float3 max(const float3 &a, float b)
{
	return make_float3(fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b));
}

__forceinline__ __device__ float atomicMaxFloat(float *addr, float value)
{
	float old;
	old = !signbit(value) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) : __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

	return old;
}

// Eccentricity at which the window exp(-ecc^(2 gamma) / 2) falls to G_THRES. The support (G >= G_THRES)
// is the triangle scaled by this factor about its centroid.
__forceinline__ __device__ float supportEcc(float gamma)
{
	return pow(-2.0f * log(G_THRES), 0.5f / gamma);
}

// Where pixel ray p_ray (z = 1) meets a triangle's plane, and the window value there.
struct TriangleSample
{
	float3 p_v1, p_v2, p_v3; // vertices relative to the hit point
	float inv_p_ray_dot_n;	 // 1 / (ray . normal)
	float inv_n_dot_n;		 // 1 / |normal|^2
	float depth;			 // view-space z of the hit
	float a1, a2, a3;		 // barycentric coordinates of the hit
	float ecc;				 // 1 - 3 min(a): 0 at the centroid, 1 on the edges
	float power;			 // -ecc^(2 gamma) / 2
	float G;				 // window value exp(power)
};

// Products written as explicit FMAs so every inlined copy of sampleTriangle rounds identically;
// otherwise the compiler may fuse multiply-adds differently in each kernel.
__forceinline__ __device__ float dotFma(const float3 &a, const float3 &b)
{
	return fmaf(a.x, b.x, fmaf(a.y, b.y, __fmul_rn(a.z, b.z)));
}

__forceinline__ __device__ float3 crossFma(const float3 &a, const float3 &b)
{
	return {fmaf(a.y, b.z, -__fmul_rn(a.z, b.y)), fmaf(a.z, b.x, -__fmul_rn(a.x, b.z)), fmaf(a.x, b.y, -__fmul_rn(a.y, b.x))};
}

// Every render kernel samples through this function, so the backward pass replays the forward pass's
// hits bit for bit. Returns false if the pixel skips the triangle: ray parallel to its plane, hit behind
// the camera, or outside the support (ecc > ecc_max). ecc^(2 gamma) is evaluated as exp2(2 gamma log2(ecc)).
__forceinline__ __device__ bool sampleTriangle(const float3 &p_ray, const float3 &v1, const float3 &v2, const float3 &v3, const float3 &normal, float gamma, float ecc_max, TriangleSample &s)
{
	const float p_ray_dot_n = dotFma(p_ray, normal);
	if (abs(p_ray_dot_n) < EPS)
		return false;
	s.inv_p_ray_dot_n = 1.0f / p_ray_dot_n;
	s.depth = dotFma(v1, normal) / p_ray_dot_n;
	if (s.depth < 0.0f)
		return false;

	auto relative = [&](const float3 &v) { return make_float3(fmaf(-s.depth, p_ray.x, v.x), fmaf(-s.depth, p_ray.y, v.y), fmaf(-s.depth, p_ray.z, v.z)); };
	s.p_v1 = relative(v1);
	s.p_v2 = relative(v2);
	s.p_v3 = relative(v3);
	s.inv_n_dot_n = 1.0f / dotFma(normal, normal);
	s.a1 = __fmul_rn(dotFma(crossFma(s.p_v2, s.p_v3), normal), s.inv_n_dot_n);
	s.a2 = __fmul_rn(dotFma(crossFma(s.p_v3, s.p_v1), normal), s.inv_n_dot_n);
	s.a3 = 1.0f - s.a1 - s.a2;
	s.ecc = max(fmaf(-3.0f, min(min(s.a1, s.a2), s.a3), 1.0f), 0.0f);
	if (s.ecc > ecc_max)
		return false;
	s.power = -0.5f * exp2f(2.0f * gamma * log2f(s.ecc));
	s.G = exp2f(s.power * 1.4426950408889634f);
	return true;
}

// A pixel's SORT_WINDOW_SIZE nearest pending hits in ascending depth; empty slots have depth FLT_MAX.
// Every loop runs its full trip count, so the window stays in registers instead of local memory.
struct SortWindow
{
	struct Hit
	{
		int global_id = -1;
		float depth = FLT_MAX;
	} hits[SORT_WINDOW_SIZE];
	int size = 0;

	__forceinline__ __device__ void push(int global_id, float depth)
	{
		Hit hit{global_id, depth};
#pragma unroll
		for (int k = 0; k < SORT_WINDOW_SIZE; k++)
			if (hit.depth < hits[k].depth)
				swap(hit, hits[k]);
		size++;
	}

	// Removes the nearest hit and returns its triangle.
	__forceinline__ __device__ int pop()
	{
		const int global_id = hits[0].global_id;
#pragma unroll
		for (int k = 1; k < SORT_WINDOW_SIZE; k++)
			hits[k - 1] = hits[k];
		hits[SORT_WINDOW_SIZE - 1] = Hit();
		size--;
		return global_id;
	}
};

// Adds a float3 with one 16-byte vector atomic where available (sm_90+); w is padding.
__forceinline__ __device__ void atomicAddVec(float4 *address, const float3 &value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
	atomicAdd(address, make_float4(value.x, value.y, value.z, 0.0f));
#else
	atomicAdd(&address->x, value.x);
	atomicAdd(&address->y, value.y);
	atomicAdd(&address->z, value.z);
#endif
}

// Whether triangle v (pixel coordinates) comes within TILE_MARGIN pixels of a pixel center of tile (x, y):
// a separating-axis test, exact for a triangle against a box. It compares products instead of summing
// them, so preprocess and duplicateWithKeys, compiled separately, always reach the same answer.
#define TILE_MARGIN 0.1f
__forceinline__ __device__ bool triangleOverlapsTile(const float2 *v, uint32_t x, uint32_t y, int W, int H)
{
	const float2 lo = {x * BLOCK_X - TILE_MARGIN, y * BLOCK_Y - TILE_MARGIN};
	const float2 hi = {min((int)(x + 1) * BLOCK_X, W) - 1 + TILE_MARGIN, min((int)(y + 1) * BLOCK_Y, H) - 1 + TILE_MARGIN};
	if (max(max(v[0].x, v[1].x), v[2].x) < lo.x || min(min(v[0].x, v[1].x), v[2].x) > hi.x ||
		max(max(v[0].y, v[1].y), v[2].y) < lo.y || min(min(v[0].y, v[1].y), v[2].y) > hi.y)
		return false;

	// The box is outside an edge if even its corner deepest inside the edge's half-plane is outside.
	// A zero-area triangle has winding 0 and is kept.
	const float2 e1 = v[1] - v[0], e2 = v[2] - v[0];
	const float winding = (e1.x * e2.y > e1.y * e2.x) - (e1.x * e2.y < e1.y * e2.x);
	for (int i = 0; i < 3; i++)
	{
		const float2 a = v[i], b = v[(i + 1) % 3];
		const float2 inward = winding * make_float2(a.y - b.y, b.x - a.x);
		const float2 corner = {inward.x > 0 ? hi.x : lo.x, inward.y > 0 ? hi.y : lo.y};
		if (inward.x * (corner.x - a.x) < -inward.y * (corner.y - a.y))
			return false;
	}
	return true;
}

#define CHECK_CUDA(debug)                                                                                              \
	if (debug)                                                                                                         \
	{                                                                                                                  \
		auto ret = cudaDeviceSynchronize();                                                                            \
		if (ret != cudaSuccess)                                                                                        \
		{                                                                                                              \
			std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
			throw std::runtime_error(cudaGetErrorString(ret));                                                         \
		}                                                                                                              \
	}
