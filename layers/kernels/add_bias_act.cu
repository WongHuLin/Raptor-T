#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <numeric>
#include <cooperative_groups/memcpy_async.h>
#include <thrust/extrema.h>
// #include <cub/cub.cuh>
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FLOAT(pointer) (reinterpret_cast<float*>(&(pointer)))

namespace sparse_transformers {
namespace layers {
namespace kernels {

template <typename T, ActivationType ActType>
__inline__ __device__ T ActivationOp(const T& x);


template <>
__inline__ __device__ float ActivationOp<float,ActivationType::Gelu>(const float& x){
    float cdf =
    0.5f *
    (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
    return x * cdf;
}

template <>
__inline__ __device__ float ActivationOp<float,ActivationType::Tanh>(const float& x){
    return tanhf(x);
}

template <>
__inline__ __device__ float ActivationOp<float,ActivationType::Relu>(const float& x){
    return (x > 0) ? x : 0;
}

template <typename T, ActivationType ActType>
__global__ void add_bias_act(T* bias, T* out, int dim_size){
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    
    const int start_index = tidx*4;

    float4 reg_a = FLOAT4(bias[start_index]);
    float4 reg_b = FLOAT4(out[start_index + bidx * dim_size]);
    float4 reg_c;

    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;

    reg_c.x = ActivationOp<T,ActType>(reg_c.x);
    reg_c.y = ActivationOp<T,ActType>(reg_c.y);
    reg_c.z = ActivationOp<T,ActType>(reg_c.z);
    reg_c.w = ActivationOp<T,ActType>(reg_c.w);

    FLOAT4(out[start_index + bidx * dim_size]) = reg_c;

}
void test_add_bias_act(float *bias, float* out, int total_seq_len, int dim_size){
    const int block_num = dim_size / 4;
    add_bias_act<float,ActivationType::Gelu><<<dim3(total_seq_len),dim3(block_num)>>>(bias,out,dim_size);
}
}
}
}