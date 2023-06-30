#include "kernel.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <numeric>
#include <cooperative_groups/memcpy_async.h>
#include <thrust/extrema.h>
// #include <cub/cub.cuh>
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FLOAT(pointer) (reinterpret_cast<float*>(&(pointer)))
#define HALF(pointer) (reinterpret_cast<half*>(&(pointer)))
#define HALF2(pointer) (reinterpret_cast<half2*>(&(pointer)))

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

template <>
__inline__ __device__ half ActivationOp<half,ActivationType::Gelu>(const half& x_h){
    float x = __half2float(x_h);
    float cdf =
    0.5 *
    (1.0 + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
    return __float2half(x * cdf);
}

template <>
__inline__ __device__ half ActivationOp<half,ActivationType::Tanh>(const half& x_h){
    float x = __half2float(x_h);
    return __float2half(tanhf(x));
}

template <>
__inline__ __device__ half ActivationOp<half,ActivationType::Relu>(const half& x_h){
    float x = __half2float(x_h);
    return __float2half((x > 0) ? x : 0);
}

template <ActivationType ActType>
__global__ void add_bias_act(float* bias, float* out, int dim_size){
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

    reg_c.x = ActivationOp<float,ActType>(reg_c.x);
    reg_c.y = ActivationOp<float,ActType>(reg_c.y);
    reg_c.z = ActivationOp<float,ActType>(reg_c.z);
    reg_c.w = ActivationOp<float,ActType>(reg_c.w);

    FLOAT4(out[start_index + bidx * dim_size]) = reg_c;
}

template <ActivationType ActType>
__global__ void add_bias_act_half(half* bias, half* out, int dim_size){
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    
    const int start_index = tidx*8;

    float4 reg_a = FLOAT4(bias[start_index]);
    float4 reg_b = FLOAT4(out[start_index + bidx * dim_size]);
    float4 reg_c;

    for(int i=0;i<4;i++)
    {
        HALF2(reg_c)[i] = __hadd2(HALF2(reg_a)[i] , HALF2(reg_b)[i]);
        HALF(reg_c)[i*2] = ActivationOp<half,ActType>(HALF(reg_c)[i*2]);
        HALF(reg_c)[i*2+1] = ActivationOp<half,ActType>(HALF(reg_c)[i*2+1]);
    }

    FLOAT4(out[start_index + bidx * dim_size]) = reg_c;
}



void test_add_bias_act(float *bias, float* out, int total_seq_len, int dim_size){
    const int block_num = dim_size / 4;
    add_bias_act<ActivationType::Gelu><<<dim3(total_seq_len),dim3(block_num)>>>(bias,out,dim_size);
}

void add_bias_act_kernel(const torch::Tensor &bias, torch::Tensor& out, int total_seq_len, int dim_size,torch::nn::GELU gelu,std::map<std::string,float>& info, bool kernel_fusion){
    
    auto start_time = std::chrono::system_clock::now();

    if(kernel_fusion)
    {
        const int block_num = dim_size / 8;
        add_bias_act_half<ActivationType::Gelu><<<dim3(total_seq_len),dim3(block_num)>>>(reinterpret_cast<half*>(bias.data_ptr()),reinterpret_cast<half*>(out.data_ptr()),dim_size);
    }
    else{
        out += bias;
        out = gelu(out);
    }
    
    auto end_time = std::chrono::system_clock::now();
    if(info.find("add_bias_act_kernel") != info.end())
    {    auto dura = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count();
        info["add_bias_act_kernel"] += dura;
    }
}

void test_add_bias_act(half *bias, half* out, int total_seq_len, int dim_size){
    const int block_num = dim_size / 8;
    // cudaEvent_t start,stop;
    // cudaEventCreate( &start );
    // cudaEventCreate( &stop ) ;
    // cudaEventRecord( start, 0 ) ;
    add_bias_act_half<ActivationType::Gelu><<<dim3(total_seq_len),dim3(block_num)>>>(bias,out,dim_size);
    // cudaEventRecord(stop,0);
    // float elapsedTime;
    // cudaEventSynchronize(stop);
    // cudaDeviceSynchronize();
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // printf( "test_add_bias_act   Time to generate:  %f ms\n", elapsedTime );

}
}
}
}
