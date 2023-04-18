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
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FLOAT(pointer) (reinterpret_cast<float*>(&(pointer)))
#define HALF(pointer) (reinterpret_cast<half*>(&(pointer)))
#define HALF2(pointer) (reinterpret_cast<half2*>(&(pointer)))


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

    const int bidx = blockIdx.x;

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

__global__ void add_bias_and_layernorm_half(half *out_data, half *input_data, 
    half *bias, int seq_len, int handle_row, int normalized_len, float eps, 
    half *layernorm_weight, half *layernorm_bias){
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tidx_dim = blockDim.x;

    const int bidx = blockIdx.x;

    typedef cub::WarpReduce<float> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[4];

    float4 data[3];
    float4 input_data_[3];
    float4 bias_[3];
    float4 layer_norm_weight_[3];
    float4 layer_norm_bias_[3];


    for(int i=0;i<3;i++){
        // 原始数据
        data[i] = FLOAT4(out_data[bidx*handle_row*normalized_len + tidy*normalized_len + tidx*24 + i*8]);
        // 加载残差数据
        input_data_[i] = FLOAT4(input_data[bidx*handle_row*normalized_len + tidy*normalized_len + tidx*24 + i*8]);
        // bias
        bias_[i] = FLOAT4(bias[tidx*24 + i*8]);
        layer_norm_weight_[i] = FLOAT4(layernorm_weight[tidx*24 + i*8]);
        layer_norm_bias_[i] = FLOAT4(layernorm_bias[tidx*24 + i*8]);
    }

    half2 mean,var;

    {
        half2 temp_h2,temp1_h2;
        float temp_f = 0.0f;
        float temp1_f = 0.0f;

        for(int i=0;i<4;i++){
            HALF2(data[0])[i] = __hadd2(__hadd2(HALF2(data[0])[i], HALF2(bias_[0])[i]),HALF2(input_data_[0])[i]);
            HALF2(data[1])[i] = __hadd2(__hadd2(HALF2(data[1])[i], HALF2(bias_[1])[i]),HALF2(input_data_[1])[i]);
            HALF2(data[2])[i] = __hadd2(__hadd2(HALF2(data[2])[i], HALF2(bias_[2])[i]),HALF2(input_data_[2])[i]);

            temp_h2 = __hadd2(__hadd2(HALF2(data[0])[i],HALF2(data[1])[i]),HALF2(data[2])[i]);
            temp1_h2 = __hadd2(__hadd2(__hmul2(HALF2(data[0])[i],HALF2(data[0])[i]), __hmul2(HALF2(data[1])[i],HALF2(data[1])[i])),__hmul2(HALF2(data[2])[i],HALF2(data[2])[i]));

            temp_f += __half2float(__hadd(HALF(temp_h2)[0], HALF(temp_h2)[1]));
            temp1_f += __half2float(__hadd(HALF(temp1_h2)[0], HALF(temp1_h2)[1]));

        }

        __shared__ float sum_temp_smem[2][4];

        sum_temp_smem[0][tidy] = WarpReduce(temp_storage[tidy]).Sum(temp_f);
        sum_temp_smem[1][tidy] = WarpReduce(temp_storage[tidy]).Sum(temp1_f);

        __syncthreads();
        temp_f = sum_temp_smem[0][tidy]/normalized_len;
        temp1_f = sum_temp_smem[1][tidy]/normalized_len-temp_f*temp_f+eps;

        mean = __half2half2(__float2half(temp_f));
        var = __half2half2(__float2half(sqrt(temp1_f)));

    }
    for(int i=0;i<4;i++){

        HALF2(data[0])[i] = __hadd2(__hmul2(__h2div(__hsub2(HALF2(data[0])[i],mean),var),HALF2(layer_norm_weight_[0])[i]),HALF2(layer_norm_bias_[0])[i]);
        HALF2(data[1])[i] = __hadd2(__hmul2(__h2div(__hsub2(HALF2(data[1])[i],mean),var),HALF2(layer_norm_weight_[1])[i]),HALF2(layer_norm_bias_[1])[i]);
        HALF2(data[2])[i] = __hadd2(__hmul2(__h2div(__hsub2(HALF2(data[2])[i],mean),var),HALF2(layer_norm_weight_[2])[i]),HALF2(layer_norm_bias_[2])[i]);

    }

    for(int i=0;i<3;i++){
        FLOAT4(out_data[bidx*handle_row*normalized_len + tidy*normalized_len + tidx*24 + i*8]) = data[i];
    }

}

void add_bias_and_layernorm_kernel(torch::Tensor &out,const torch::Tensor &input_data,const torch::Tensor &bias,int seq_len, int handle_row, int normalized_len, float eps,const torch::Tensor &layernorm_weight,const torch::Tensor &layernorm_bias,torch::nn::LayerNorm& layernorm,std::map<std::string,float> &info, bool kernel_fusion){

    auto start_time = std::chrono::system_clock::now();

    if(kernel_fusion)
    {
        add_bias_and_layernorm_half<<<dim3(seq_len/4),dim3(32,4)>>>(reinterpret_cast<half*>(out.data_ptr()),reinterpret_cast<half*>(input_data.data_ptr()),reinterpret_cast<half*>(bias.data_ptr()),seq_len,4,normalized_len,eps,reinterpret_cast<half*>(layernorm_weight.data_ptr()),reinterpret_cast<half*>(layernorm_bias.data_ptr()));
    }
    else{
        out += input_data + bias;
        out = layernorm(out);
    }
    auto end_time = std::chrono::system_clock::now();
    if(info.find("add_bias_and_layernorm_kernel") != info.end())
    {auto dura = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count();
        info["add_bias_and_layernorm_kernel"] += dura;
    }
}

void test_add_bias_and_layernorm(float *out,float *input_data, float *bias,int seq_len, int handle_row, int normalized_len, float eps,float* layernorm_weight,float* layernorm_bias){
    // cudaEvent_t start,stop;
    // cudaEventCreate( &start );
    // cudaEventCreate( &stop ) ;
    // cudaEventRecord( start, 0 ) ;
    add_bias_and_layernorm_1<float><<<dim3(seq_len/handle_row),dim3(32,6)>>>(out,input_data,bias,seq_len,handle_row,normalized_len,eps,layernorm_weight,layernorm_bias);
    // cudaEventRecord(stop,0);
    // float elapsedTime;
    // cudaEventSynchronize(stop);
    // cudaDeviceSynchronize();
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // printf( "Time to generate:  %f ms\n", elapsedTime );

}

void test_add_bias_and_layernorm(half *out,half *input_data, half *bias,int seq_len, int handle_row, int normalized_len, float eps,half* layernorm_weight,half* layernorm_bias){
    // cudaEvent_t start,stop;
    // cudaEventCreate( &start );
    // cudaEventCreate( &stop ) ;
    // cudaEventRecord( start, 0 ) ;
    // normalized_len = 768, handle_row = 4
    add_bias_and_layernorm_half<<<dim3(seq_len/4),dim3(32,4)>>>(out,input_data,bias,seq_len,handle_row,normalized_len,eps,layernorm_weight,layernorm_bias);
    // cudaEventRecord(stop,0);
    // float elapsedTime;
    // cudaEventSynchronize(stop);
    // cudaDeviceSynchronize();
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // printf( "test_add_bias_and_layernorm  Time to generate:  %f ms\n", elapsedTime );

}
}
}
}
