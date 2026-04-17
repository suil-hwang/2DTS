#pragma once

#include <cmath>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define EPS (float)(1e-8)
#define FLT_INF std::numeric_limits<float>::infinity()

__forceinline__ __device__ float projToPix(float v, int S)
{
    return (v + 1.0f) * S * 0.5f - 0.5f;
}

__forceinline__ __device__ float pixToProj(float v, int S)
{
    return (2.0f * v - S + 1.0f) / (float)(S);
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
    return ((v + 1.0f) * S - 1.0f) * 0.5f;
}

__forceinline__ __device__ float2 operator+(const float2 &a, const float2 &b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__forceinline__ __device__ float2 operator-(const float2 &a, const float2 &b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

__forceinline__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __device__ float3 operator*(const float3 &a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__forceinline__ __device__ float3 operator*(float b, const float3 &a)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__forceinline__ __device__ float3 operator/(const float3 &a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__forceinline__ __device__ void operator+=(float3 &a, const float3 &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__forceinline__ __device__ float dot(const float3 &a, const float3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ float3 cross(const float3 &a, const float3 &b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__forceinline__ __device__ float norm(const float3 &a)
{
    return sqrtf(dot(a, a));
}

__forceinline__ __device__ float2 min(const float2 &a, const float2 &b)
{
    return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

__forceinline__ __device__ float2 min(const float2 &a, const float2 &b, const float2 &c)
{
    return min(min(a, b), c);
}

__forceinline__ __device__ float2 max(const float2 &a, const float2 &b)
{
    return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

__forceinline__ __device__ float2 max(const float2 &a, const float2 &b, const float2 &c)
{
    return max(max(a, b), c);
}

__forceinline__ __device__ float3 transformPoint4x3(const float3 &p, const float *matrix)
{
    return {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
    };
}

__forceinline__ __device__ float4 transformPoint4x4(const float3 &p, const float *matrix)
{
    return {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
        matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15],
    };
}

__forceinline__ __device__ float3 projectPoint(const float3 &p, const float *projmatrix)
{
    float4 p_hom = transformPoint4x4(p, projmatrix);
    float inv_w = 1.0f / (fabsf(p_hom.w) + EPS);
    return {p_hom.x * inv_w, p_hom.y * inv_w, p_hom.z * inv_w};
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2 &rect_min, uint2 &rect_max, dim3 grid)
{
    rect_min = {
        min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))};
    rect_max = {
        min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))};
}

__forceinline__ __device__ float atomicMaxFloat(float *addr, float value)
{
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) : __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
    return old;
}

#define CHECK_CUDA(debug)                                                                                     \
    if (debug)                                                                                                \
    {                                                                                                         \
        auto ret = cudaDeviceSynchronize();                                                                   \
        if (ret != cudaSuccess)                                                                               \
        {                                                                                                     \
            std::cerr << "\\n[CUDA ERROR] in " << __FILE__ << "\\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
            throw std::runtime_error(cudaGetErrorString(ret));                                               \
        }                                                                                                     \
    }
