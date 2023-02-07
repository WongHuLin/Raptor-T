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
// (seq_len)(32*8)
template <class DataType>
__global__ void add_bias_and_layernorm(DataType *out_data, DataType *input_data, 
    DataType *bias, int seq_len, int handle_row, int normalized_len, float eps, 
    DataType *layernorm_weight, DataType *layernorm_bias){
    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int tid = tidy*32+tidx;

    typedef cub::WarpReduce<float> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[8];

    __shared__ float input_smem[2][768];
    float mean[2],var[2];

    float bias_re[3];
    bias_re[0] = bias[tid];
    bias_re[1] = bias[tid+256];
    bias_re[2] = bias[tid+256*2];

    float layernorm_weight_re[3];
    layernorm_weight_re[0] = layernorm_weight[tid];
    layernorm_weight_re[1] = layernorm_weight[tid+256];
    layernorm_weight_re[2] = layernorm_weight[tid+256*2];

    
    auto block = cooperative_groups::this_thread_block();

    int data_offset = bidx*normalized_len*handle_row;

    cooperative_groups::memcpy_async(block, input_smem[0], input_data+data_offset, sizeof(float)*768*2);

    cooperative_groups::wait(block);
    {
        float sum[2] = {0.0f, 0.0f};
        float sum_[2] = {0.0f, 0.0f};
        __shared__ float sum_temp_smem[4][8];
        __shared__ float sum_temp_smem_[2][8];

        for(int i=0;i<3;i++){
            input_smem[0][256*i+tid] += bias_re[i];
            input_smem[1][256*i+tid] += bias_re[i];
            sum[0] += input_smem[0][256*i+tid];
            sum[1] += input_smem[1][256*i+tid];
            sum_[0] += input_smem[0][256*i+tid]*input_smem[0][256*i+tid];
            sum_[1] += input_smem[1][256*i+tid]*input_smem[1][256*i+tid];
        }
        sum_temp_smem[0][tidy] = WarpReduce(temp_storage[tidy]).Sum(sum[0]);
        sum_temp_smem[1][tidy] = WarpReduce(temp_storage[tidy]).Sum(sum[1]);
        sum_temp_smem[2][tidy] = WarpReduce(temp_storage[tidy]).Sum(sum_[0]);
        sum_temp_smem[3][tidy] = WarpReduce(temp_storage[tidy]).Sum(sum_[1]);

        __syncthreads();
        if(tidy < 4){
            float temp = 0.0f;
            if(tidx < 8)
                temp = sum_temp_smem[tidy][tidx];
            sum_temp_smem[0][tidy] = WarpReduce(temp_storage[tidy]).Sum(temp);
        }
        __syncthreads();
        mean[0] = sum_temp_smem[0][0]/normalized_len;
        mean[1] = sum_temp_smem[0][1]/normalized_len;
        var[0] = sqrt(sum_temp_smem[0][2]/normalized_len-mean[0]*mean[0]+eps);
        var[1] = sqrt(sum_temp_smem[0][3]/normalized_len-mean[1]*mean[1]+eps);
        
    }
    for(int i=0;i<3;i++){
        input_smem[0][256*i+tid] = (input_smem[0][256*i+tid] - mean[0])/var[0];
        input_smem[1][256*i+tid] = (input_smem[1][256*i+tid] - mean[1])/var[1];
    }

    __syncthreads();

    cooperative_groups::memcpy_async(block, out_data+data_offset, input_smem[0],sizeof(float)*768*2);

}

// 残差 相加
template <class DataType>
__global__ void add_bias_and_layernorm_1(DataType *out_data, DataType *input_data, 
    DataType *bias, int seq_len, int handle_row, int normalized_len, float eps, 
    DataType *layernorm_weight, DataType *layernorm_bias){
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tidx_dim = blockDim.x;
    const int tid = tidy*tidx_dim + tidx;

    const int tidy_dim = blockDim.y;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    typedef cub::WarpReduce<float> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[6];

    // 加载数据
    // 原始数据
    float4 data_1 = FLOAT4(out_data[bidx*handle_row*normalized_len + tid*4]);
    // 加载残差数据
    float4 input_data_1 = FLOAT4(input_data[bidx*handle_row*normalized_len + tid*4]);
    float4 data_2 = FLOAT4(out_data[(bidx*handle_row+1)*normalized_len + tid*4]);
    float4 input_data_2 = FLOAT4(input_data[(bidx*handle_row+1)*normalized_len + tid*4]);
    float4 bias_ = FLOAT4(bias[tid*4]);
    float4 layer_norm_weight_ = FLOAT4(layernorm_weight[tid*4]);
    float4 layer_norm_bias_ = FLOAT4(layernorm_bias[tid*4]);

    // if(bidx == 0 && bidy == 0 && tidx == 0 && tidy ==0 )
    //     printf("%f %f %f \n",FLOAT(data_1)[1],FLOAT(input_data_1)[1],FLOAT(bias_)[1]);

    float mean[2],var[2];

    {
        float sum[2] = {0.0f, 0.0f};
        float sum_[2] = {0.0f, 0.0f};

        for(int i=0;i<4;i++){
            FLOAT(data_1)[i] = FLOAT(data_1)[i] + FLOAT(bias_)[i] + FLOAT(input_data_1)[i];
            FLOAT(data_2)[i] = FLOAT(data_2)[i] + FLOAT(bias_)[i] + FLOAT(input_data_2)[i];
            sum[0] += FLOAT(data_1)[i];
            sum[1] += FLOAT(data_2)[i];
            sum_[0] += FLOAT(data_1)[i]*FLOAT(data_1)[i];
            sum_[1] += FLOAT(data_2)[i]*FLOAT(data_2)[i];
        }
        // if(bidx == 0 && bidy == 0)
        //     printf("%f %f \n",sum[0],sum[1]);

        __shared__ float sum_temp_smem[4][6];

        sum_temp_smem[0][tidy] = WarpReduce(temp_storage[tidy]).Sum(sum[0]);
        sum_temp_smem[1][tidy] = WarpReduce(temp_storage[tidy]).Sum(sum[1]);
        sum_temp_smem[2][tidy] = WarpReduce(temp_storage[tidy]).Sum(sum_[0]);
        sum_temp_smem[3][tidy] = WarpReduce(temp_storage[tidy]).Sum(sum_[1]);

        __syncthreads();
        if(tidy < 4){
            float temp = 0.0f;
            if(tidx < 6)
                temp = sum_temp_smem[tidy][tidx];
            
            sum_temp_smem[0][tidy] = WarpReduce(temp_storage[tidy]).Sum(temp);
        }

        // if(tidy < 4){
        //     float temp = 0.0f;
        //     if(tidx < 6)
        //         temp = sum_temp_smem[tidy][tidx];
            
        //         sum_temp_smem[0][tidy] = WarpReduce(temp_storage[tidy]).Sum(temp);
        // }
        __syncthreads();
        mean[0] = sum_temp_smem[0][0]/normalized_len;
        mean[1] = sum_temp_smem[0][1]/normalized_len;
        var[0] = sqrt(sum_temp_smem[0][2]/normalized_len-mean[0]*mean[0]+eps);
        var[1] = sqrt(sum_temp_smem[0][3]/normalized_len-mean[1]*mean[1]+eps);

        // if(bidx == 0 && bidy == 0 && tidx == 0 && tidy ==0 )
        //     printf("%f %f %f %f\n",mean[0],mean[1],var[0],var[1]);
        
    }
    for(int i=0;i<4;i++){
        FLOAT(data_1)[i] = (FLOAT(data_1)[i] - mean[0])/var[0]*FLOAT(layer_norm_weight_)[i] + FLOAT(layer_norm_bias_)[i];

        FLOAT(data_2)[i] = (FLOAT(data_2)[i] - mean[1])/var[1]*FLOAT(layer_norm_weight_)[i] + FLOAT(layer_norm_bias_)[i];
    }

    __syncthreads();

    FLOAT4(out_data[bidx*handle_row*normalized_len + tid*4]) = data_1;
    FLOAT4(out_data[(bidx*handle_row+1)*normalized_len + tid*4]) = data_2;

}
void test_add_bias_and_layernorm(float *out,float *input_data, float *bias,int seq_len, int handle_row, int normalized_len, float eps,float* layernorm_weight,float* layernorm_bias){
    cudaEvent_t start,stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop ) ;
    cudaEventRecord( start, 0 ) ;
    add_bias_and_layernorm_1<float><<<dim3(seq_len/handle_row),dim3(32,6)>>>(out,input_data,bias,seq_len,handle_row,normalized_len,eps,layernorm_weight,layernorm_bias);
    cudaEventRecord(stop,0);
    float elapsedTime;
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time to generate:  %f ms\n", elapsedTime );

}
}
}
}