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
    
//(12,64)(32*8)
//input_data: seq_len * all_head_size * 3   bias: all_head_size  q,k,v: seq * all_head_size
template <class DataType>
__global__ void add_bias_and_transpose(DataType *input_data, DataType *bias, 
    DataType *q, DataType *k, DataType *v, int q_offset, int k_offset, 
    int v_offset,int batch_size, int seq_len, int head_num, int block_size, 
    int head_size, int block_num){
    // For K and V: batch_size * seq_len * all_head_size -> batch_size * head_num * block_num * blcok_size * head_size
    // For Q: batch_size * seq_len * all_head_size -> batch_size * head_num * block_num * head_size* blcok_size
    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int all_head_size = head_num * head_size;
    const int start_read_data_index = bidy * block_size * head_size * 3 * head_num + bidx * head_size;
    const int start_write_data_index = bidx * block_num * block_size * head_size + bidy * block_size * head_size;
    __shared__ float q_bias[64],k_bias[64],v_bias[64];

    // load bias
    auto block = cooperative_groups::this_thread_block();
    cooperative_groups::memcpy_async(block, q_bias, bias+bidx*head_size+q_offset, sizeof(float)*64);
    cooperative_groups::memcpy_async(block, k_bias, bias+bidx*head_size+all_head_size, sizeof(float)*64);
    cooperative_groups::memcpy_async(block, v_bias, bias+bidx*head_size+all_head_size*2, sizeof(float)*64);
    
    //smem_q 为32*33 是为了转置时，避免bank conflict
    __shared__ float smem_k[16*64],smem_v[16*64],smem_q[32][33];
    int smem_index = (tidy*32+tidx)*4;
    for(int block_row=0;block_row<block_size;block_row+=32){
        for(int block_col=0;block_col<head_size;block_col+=32){

            int smem_row_index = smem_index / 32;
            int smem_col_index = smem_index % 32;
            int read_index = start_read_data_index + head_num*head_size*3*(block_row+smem_row_index) + block_col + smem_col_index;
            // load q k v

            FLOAT4(smem_k[smem_index]) = FLOAT4(input_data[read_index+k_offset]);
            FLOAT4(smem_v[smem_index]) = FLOAT4(input_data[read_index+v_offset]);
            smem_q[smem_row_index][smem_col_index] = input_data[read_index+q_offset];
            smem_q[smem_row_index][smem_col_index+1] = input_data[read_index+q_offset+1];
            smem_q[smem_row_index][smem_col_index+2] = input_data[read_index+q_offset+2];
            smem_q[smem_row_index][smem_col_index+3] = input_data[read_index+q_offset+3];

            cooperative_groups::wait(block);
            for(int i=0;i<4;i++){
                smem_k[(tidy*4+i)*32+tidx] += k_bias[tidx+block_col];
                smem_v[(tidy*4+i)*32+tidx] += v_bias[tidx+block_col];
                smem_q[(tidy*4+i)][tidx] += q_bias[tidx+block_col];
            }

            int write_index = start_write_data_index + (block_row+smem_row_index)*head_size + block_col + smem_col_index;
            FLOAT4(k[write_index]) = FLOAT4(smem_k[smem_index]);
            FLOAT4(v[write_index]) = FLOAT4(smem_v[smem_index]);

            __syncthreads();
            for(int i=0;i<4;i++){
                q[start_write_data_index + (tidy*4+i+block_col)*64 + tidx + block_row] = smem_q[tidx][(tidy*4+i)];
            }
            __syncthreads();
        }
    }
}
  
//(12,64)(32*8)
template <class DataType>
__global__ void sparse_attention(DataType *a,  DataType *b,  DataType *c, 
    DataType *out,const int *select_index,int *select_index_position, 
    const int block_size,const int head_size,const int select_block_num){


    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int b_dimx = 8;
    const int g_dimy = gridDim.y;

    // 计算Q的起始位置
    const int from_block_id = bidy;
    const int data_offset_a = (bidx*g_dimy + from_block_id) * block_size * head_size;

    const int A_BM = 32;
    const int A_BK = 64;
    const int B_BK = 32;
    const int B_BN = 4;
    const int C_BK = 32;
    const int C_BN = 4;


    __shared__ float smem_q[64][32],smem_k[32][64],temp_score[32][32],smem_v[32][64];

    __shared__ float out_temp[32][64],global_sum_scores[32],temp_smem[16][32],pre_max_score[32],max_score[32];

    float zero4[4] = {0.0f,0.0f,0.0f,0.0f};

    auto block_k = cooperative_groups::this_thread_block();
    auto block_v = cooperative_groups::this_thread_block();

    for(int a_bm = 0; a_bm< block_size/A_BM; a_bm++){

        const int to_block_start = select_index_position[from_block_id];
        const int to_block_end = select_index_position[from_block_id+1];

        int to_block_id = select_index[to_block_start];

        int data_b_start = (bidx*g_dimy+to_block_id) * block_size * head_size;
        
        cooperative_groups::memcpy_async(block_k, smem_k[0], b+data_b_start, sizeof(float)*32*64);
        cooperative_groups::memcpy_async(block_v, smem_v[0], c+data_b_start, sizeof(float)*32*64);


        const int smem_index =  32*8*tidy + tidx*8; // warp_size * 8
        const int global_a_index_i = (smem_index / 32 );
        const int global_a_index_j = (smem_index % 32 + a_bm*A_BM);

        // 加载Q的部分数据,32*64
        #pragma unroll
        for(int i=0;i<8;i+=4){
            FLOAT4(smem_q[smem_index/32][smem_index % 32+i]) = FLOAT4(a[data_offset_a + global_a_index_i*head_size+global_a_index_j+i]); 
            FLOAT4(out_temp[smem_index/64][smem_index % 64+i]) = FLOAT4(zero4[0]);
        }

        // 初始化sharedmem
        if(tidy == 0){
            max_score[tidx] = 0.0f;
            // sum_score_max[tidx] = 0.0f;
            pre_max_score[tidx] = 0.0f;
            global_sum_scores[tidx] = 0.0f;
        }

        __syncthreads();

        // 遍历K、V的每一个Block进行计算
        for(int block_id=to_block_start;block_id<to_block_end;block_id++)
        {

            // if(bidx == 0 && bidy == 4 && tidx == 0 && tidy == 0)
            //     printf("%d %d %d %d\n",to_block_start,to_block_end,block_id,select_index[block_id]);
            // 计算KV块的起始位置
            
            // KV按照 32*64 的大小进行加载计算
            for(int b_bn=0;b_bn<block_size/32;b_bn++){
                
                // 计算Q*K
                cooperative_groups::wait(block_k);
                for(int i=0;i<4;i++){
                    float temp = 0.0f;
                    for(int j=0;j<64;j++){
                        temp += smem_q[j][tidx] * smem_k[tidy*4+i][j];
                    }
                    temp_score[tidy*4+i][tidx] = temp;
                }

                //加载下一次使用的数据
                const int next_block_id = b_bn == 1 ? block_id+1:block_id;
                const int next_bn = (b_bn + 1) & 1;
                to_block_id = select_index[next_block_id];
                const int data_b_start = (bidx*g_dimy+to_block_id) * block_size * head_size;
                __syncthreads();
                if(block_id != to_block_end - 1 || b_bn != 1)
                {
                    cooperative_groups::memcpy_async(block_k, smem_k[0], b+data_b_start+next_bn*32*head_size, sizeof(float)*32*64);
                }
            
                //计算最大值 rowmax
                {
                    int num = 16;
                    while(num >= 1)
                    {
                        if(num == 16)
                        {
                            float value1 = temp_score[tidy][tidx];
                            float value2 = temp_score[tidy+16][tidx];
                            float value3 = temp_score[tidy+8][tidx];
                            float value4 = temp_score[tidy+24][tidx];

                            temp_smem[tidy][tidx] = value1>value2?value1:value2;
                            temp_smem[tidy+8][tidx] = value3>value4?value3:value4;
                        }
                        else if(tidy < num){
                            float value1 = temp_smem[tidy][tidx];
                            float value2 = temp_smem[tidy+num][tidx];
                            temp_smem[tidy][tidx] = value1>value2?value1:value2;
                        }
                        num = num >> 1;
                        __syncthreads();
                    }
                    if(tidy == 0)
                    {
                        pre_max_score[tidx] = max_score[tidx];
                        max_score[tidx] = max_score[tidx]>temp_smem[0][tidx]?max_score[tidx]:temp_smem[0][tidx];
                    }
                }

                //计算差值

                {
                    __syncthreads();
                    for(int i=0;i<4;i++)
                    {
                        float temp =  exp(temp_score[(i+tidy*4)][tidx] - max_score[tidx]);
                        temp_score[(i+tidy*4)][tidx] = temp;
                    }

                    #pragma unroll
                    for(int i=0;i<4;i++){
                        float diff = (pre_max_score[tidy*4+i] - max_score[tidy*4+i]);
                        if(diff != 0){
                            diff = exp(diff);
                            out_temp[tidy*4+i][tidx] *= diff;
                            out_temp[tidy*4+i][tidx+32] *= diff;
                        }
                    }
                    __syncthreads();
                    if(tidy == 0){
                        float diff = exp(pre_max_score[tidx] - max_score[tidx]);
                        global_sum_scores[tidx] *= diff;
                    }
                    

                }
                {
                    int num = 16;
                    while(num >= 1)
                    {
                        if(num == 16)
                        {
                            int value1 = temp_score[tidy][tidx];
                            int value2 = temp_score[tidy+16][tidx];
                            int value3 = temp_score[tidy+8][tidx];
                            int value4 = temp_score[tidy+24][tidx];
                            temp_smem[tidy][tidx] = value1 + value2;
                            temp_smem[tidy+8][tidx] = value3 + value4;
                        }
                        else if(tidy < num){
                            int value1 = temp_smem[tidy][tidx];
                            int value2 = temp_smem[tidy+num][tidx];
                            temp_smem[tidy][tidx] = value1 + value2;
                        }
                        num = num >> 1;
                        __syncthreads();
                    }
                    if(tidy == 0)
                        global_sum_scores[tidx] += temp_smem[0][tidx];
                }

                //计算S*V
                cooperative_groups::wait(block_v);

                #pragma unroll
                for(int i = 0;i<4; i++){
                    for(int j=0;j<32;j++){
                        out_temp[tidy*4+i][tidx] += temp_score[j][tidy*4+i]*smem_v[j][tidx];
                        out_temp[tidy*4+i][tidx+32] += temp_score[j][tidy*4+i]*smem_v[j][tidx+32];
                    }
                }
                __syncthreads();

                //加载下一次使用的数据
                if(block_id != to_block_end - 1 || b_bn != 1)
                {
                    cooperative_groups::memcpy_async(block_v, smem_v[0], c+data_b_start+next_bn*32*head_size, sizeof(float)*32*64);
                }
            }
        }

        const int index_x = (tidy%4)*8;
        const int index_y = tidx + (tidy/4)*32;
        // 结果写入global mem
        #pragma unroll
        for(int i=0;i<8;i+=1){
            out[data_offset_a+(a_bm*A_BM+index_x+i)*head_size+index_y] = out_temp[(index_x+i)][index_y] / global_sum_scores[index_x+i];
        }

        __syncthreads();
        // printf("111\n");
    }
}

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


void test_add_bias_and_transpose(float *bias,float *input_data,float *q, float *k,float *v, int q_offset, int k_offset, int v_offset,int batch_size, int seq_len, int head_num, int block_size,int block_num, int head_size){
    cudaEvent_t start,stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop ) ;
    cudaEventRecord( start, 0 ) ;
    add_bias_and_transpose<float><<<dim3(head_num,block_num),dim3(32,8)>>>(input_data,bias,q,k,v,q_offset,k_offset,v_offset,batch_size,seq_len,head_num,block_size,head_size,block_num);
    cudaEventRecord(stop,0);
    float elapsedTime;
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time to generate:  %f ms\n", elapsedTime );

}


void test_gemm_(float *a, float *b,float *c, float *out,int *select_index,int *select_index_position, int block_num, int head_num,int block_size,int head_size)
{
    // std::cout<<*a<<std::endl;
    // sparse_attention<float><<<dim3(block_num,head_num),dim3(11,32)>>>(a,b,c,out,select_index,64,64);

    cudaEvent_t start,stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop ) ;
    // test_gemm<float><<<1,dim3(11,32)>>>(a,b,c,m,n,k,64);

    cudaEventRecord( start, 0 ) ;
    sparse_attention<float><<<dim3(head_num,block_num),dim3(32,8)>>>(a,b,c,out,select_index,select_index_position,64,64,11);

    // test_gpu<<<1,1>>>();
    // test_cpu();
    cudaEventRecord(stop,0);
    float elapsedTime;
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time to generate:  %f ms\n", elapsedTime );
    // printf("%f\n",*a);
}

void test_add_bias_act(float *bias, float* out, int total_seq_len, int dim_size){
    const int block_num = dim_size / 4;
    add_bias_act<float,ActivationType::Gelu><<<dim3(total_seq_len),dim3(block_num)>>>(bias,out,dim_size);
}
}
}
}