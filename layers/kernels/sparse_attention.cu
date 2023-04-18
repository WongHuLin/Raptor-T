#include "kernel.h"
#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <numeric>
#include <cooperative_groups/memcpy_async.h>
#include <thrust/extrema.h>
#include <mma.h>
// #include <cub/cub.cuh>
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FLOAT(pointer) (reinterpret_cast<float*>(&(pointer)))
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer)))
#define HALF(pointer) (reinterpret_cast<half*>(&(pointer)))
#define HALF2(pointer) (reinterpret_cast<half2*>(&(pointer)))
using namespace nvcuda;
namespace sparse_transformers {
namespace layers {
namespace kernels {

// a is not transpose
template <class DataType>
__global__ void sparse_attention_with_tensor_core(DataType *a,  DataType *b,  DataType *c, DataType *out, const int *to_select_index,const int *to_select_index_position,  const int block_size,const int head_size,const int select_block_num){

    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int b_dimx = 8;
    const int g_dimy = gridDim.y;

    // 计算Q的起始位置
    const int from_block_id = bidy;
    const int data_offset_q = (bidx*g_dimy + from_block_id) * block_size * head_size;

    const int A_BM = 32;
    const int A_BK = 64;
    const int B_BK = 32;
    const int B_BN = 4;
    const int C_BK = 32;
    const int C_BN = 4;


    __shared__ half smem_q[32 * 64],smem_k[32 * 64],smem_v[32 * 64],smem_temp_half[64][32];
    __shared__ float temp_score[32][32],out_temp[32][64],max_score_diff[32];
    __shared__ half temp_score_half[32][32];

    __shared__ float global_sum_scores[32],pre_max_score[32],max_score[32],temp_smem[32];

    typedef cub::WarpReduce<float> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[8];

    float zero4[4] = {0.0f,0.0f,0.0f,0.0f};
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_q[2];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_k[2];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_s_out;

    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_s[2];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_v[2];
    wmma::fragment<wmma::accumulator, 8, 32, 16, float> frag_out;

    auto block_k = cooperative_groups::this_thread_block();
    auto block_v = cooperative_groups::this_thread_block();

    for(int a_bm = 0; a_bm< block_size/A_BM; a_bm++){

        const int to_block_start = to_select_index_position[from_block_id];
        const int to_block_end = to_select_index_position[from_block_id+1];

        

        int to_block_id = to_select_index[to_block_start];

        int data_b_start = (bidx*g_dimy+to_block_id) * block_size * head_size;
        const int smem_index =  32*8*tidy + tidx*8; // warp_size * 8

        // 加载Q的部分数据,32*64
        // Q  tidy
        //      --------------------
        //          0     |     1
        //      --------------------
        //          2     |     3
        //      --------------------
        //          4     |     5
        //      --------------------
        //          6     |     7
        //      --------------------
        const int smem_index_q = 32*8*tidy + (tidx/16)*16*8 + (tidx%16)*8;
        const int global_q_index_i = (a_bm*A_BM + (tidx%16)/2 + (tidy/2)*8);
        const int global_q_index_j = ((tidx%2)*8 + (tidx/16)*16 + (tidy%2)*32);
        FLOAT4(smem_q[smem_index_q]) = FLOAT4(a[data_offset_q+global_q_index_i*head_size+global_q_index_j]); 

        FLOAT4(out_temp[global_q_index_i - a_bm*A_BM][global_q_index_j]) = FLOAT4(zero4[0]);
        // wmma::fill_fragment(frag_s_out, half(0.0));
        wmma::fill_fragment(frag_s_out, __float2half(0.0));
        wmma::fill_fragment(frag_out, 0.0);


        // KT 32*16  tidy  Q*K 8*16*32
        //      -----------------------------------------
        //          0     |     2   |   4     |     6   
        //      -----------------------------------------
        //          1     |     3   |   5     |     7   
        //      -----------------------------------------

        const int global_k_index_i = tidx/2 + (tidy%2)*16;
        const int global_k_index_j = (tidx%2)*8 + (tidy/2)*16;
        FLOAT4(smem_k[smem_index]) = FLOAT4(b[data_b_start + global_k_index_i*head_size+global_k_index_j]); 


        // V 16*32  tidy  S*K 8*16*32
        //      --------------------
        //          0     |     4
        //      --------------------
        //          1     |     5
        //      --------------------
        //          2     |     6
        //      --------------------
        //          3     |     7
        //      --------------------
        const int global_v_index_i = (tidx/4 + ((tidy/2)%2)*16 + (tidy%2)*8);
        const int global_v_index_j = ((tidx*8)%32 + ((tidy/2)/2)*32);
        FLOAT4(smem_v[smem_index]) = FLOAT4(c[data_b_start + global_v_index_i*head_size+global_v_index_j]); 

        wmma::load_matrix_sync(frag_q[0], &smem_q[tidy*2*8*16], 16);
        wmma::load_matrix_sync(frag_q[1], &smem_q[(tidy*2+1)*8*16], 16);

        // 初始化sharedmem
        if(tidy == 0){
            max_score[tidx] = 0.0f;
            // sum_score_max[tidx] = 0.0f;
            pre_max_score[tidx] = 0.0f;
            global_sum_scores[tidx] = 0.0f;
        }


        // 遍历K、V的每一个Block进行计算
        for(int block_id=to_block_start;block_id<to_block_end;block_id++)
        {
            // KV按照 32*64 的大小进行加载计算
            for(int b_bn=0;b_bn<block_size/32;b_bn++){
                
                // 计算Q*K
                // cooperative_groups::wait(block_k);
                wmma::fill_fragment(frag_s_out, __float2half(0.0));

                if(tidy % 2 == 0){
                    wmma::load_matrix_sync(frag_k[0], &smem_k[0], 16);
                    wmma::load_matrix_sync(frag_k[1], &smem_k[16*32], 16);
                }
                else{
                    wmma::load_matrix_sync(frag_k[0], &smem_k[16*32*2], 16);
                    wmma::load_matrix_sync(frag_k[1], &smem_k[16*32*3], 16);
                }
                wmma::mma_sync(frag_s_out, frag_q[0], frag_k[0], frag_s_out);
                wmma::mma_sync(frag_s_out, frag_q[1], frag_k[1], frag_s_out);
                wmma::store_matrix_sync(&smem_temp_half[(tidy/2)*8 + (tidy % 2)*32][0], frag_s_out, 32, wmma::mem_row_major);
                
                __syncthreads();

                for(int i=0;i<4;i++){
                    // temp_score[i*8+tidy][tidx] = __half2float(smem_temp_half[i*8+tidy][tidx] + smem_temp_half[i*8+tidy+32][tidx]);
                    temp_score[i*8+tidy][tidx] = __half2float(smem_temp_half[i*8+tidy][tidx]) + __half2float(smem_temp_half[i*8+tidy+32][tidx]);
                }

                //加载下一次使用的数据
                const int next_block_id = b_bn == 1 ? block_id+1:block_id;
                const int next_bn = (b_bn + 1) & 1;
                to_block_id = to_select_index[next_block_id];
                data_b_start = (bidx*g_dimy+to_block_id) * block_size * head_size;
                if(block_id != to_block_end - 1 || b_bn != 1)
                {
                    FLOAT4(smem_k[smem_index]) = FLOAT4(b[data_b_start + next_bn*32*head_size + global_k_index_i*head_size+global_k_index_j]);
                }
            
                //计算最大值 rowmax
                {
                    float value1 = temp_score[tidy][tidx];
                    float value2 = temp_score[tidy+16][tidx];
                    float value3 = temp_score[tidy+8][tidx];
                    float value4 = temp_score[tidy+24][tidx];
                    
                    temp_smem[tidy] = WarpReduce(temp_storage[tidy]).Reduce(value1, cub::Max());
                    temp_smem[tidy+16] = WarpReduce(temp_storage[tidy]).Reduce(value2, cub::Max());
                    temp_smem[tidy+8] = WarpReduce(temp_storage[tidy]).Reduce(value3, cub::Max());
                    temp_smem[tidy+24] = WarpReduce(temp_storage[tidy]).Reduce(value4, cub::Max());

                    __syncthreads();
                    if(tidy == 0)
                    {
                        pre_max_score[tidx] = max_score[tidx];
                        max_score[tidx] = max_score[tidx]>temp_smem[tidx]?max_score[tidx]:temp_smem[tidx];
                        max_score_diff[tidx] = exp(pre_max_score[tidx] - max_score[tidx]);
                    }
                }

                //计算差值
                {
                    __syncthreads();
                    for(int i=0;i<4;i++)
                    {
                        float temp =  exp(temp_score[(i+tidy*4)][tidx] - max_score[tidy*4+i]);
                        temp_score[(i+tidy*4)][tidx] = temp;
                        temp_score_half[(i+tidy*4)][tidx] = __float2half(temp);
                    }

                    const int t = (tidy/2)*8+(tidx%4)*2;
                    for(int i=0;i<4;i++){
                        frag_out.x[i*2] *= max_score_diff[t];
                        frag_out.x[i*2+1] *= max_score_diff[t+1];
                    }
                    __syncthreads();  
                }

                {
                    float value1 = temp_score[tidy][tidx];
                    float value2 = temp_score[tidy+16][tidx];
                    float value3 = temp_score[tidy+8][tidx];
                    float value4 = temp_score[tidy+24][tidx];
                    
                    temp_smem[tidy] = WarpReduce(temp_storage[tidy]).Sum(value1);
                    temp_smem[tidy+16] = WarpReduce(temp_storage[tidy]).Sum(value2);
                    temp_smem[tidy+8] = WarpReduce(temp_storage[tidy]).Sum(value3);
                    temp_smem[tidy+24] = WarpReduce(temp_storage[tidy]).Sum(value4);
                    
                    __syncthreads();

                    if(tidy == 0)
                    {
                        global_sum_scores[tidx] *= max_score_diff[tidx];
                        global_sum_scores[tidx] += temp_smem[tidx];
                    }
                }

                // //计算S*V
                wmma::load_matrix_sync(frag_s[0], &temp_score_half[(tidy/2)*8][0], 32);
                wmma::load_matrix_sync(frag_s[1], &temp_score_half[(tidy/2)*8][16], 32);
                if(tidy % 2 == 0){
                    wmma::load_matrix_sync(frag_v[0], &smem_v[0], 32);
                    wmma::load_matrix_sync(frag_v[1], &smem_v[32*16], 32);
                }
                else{
                    wmma::load_matrix_sync(frag_v[0], &smem_v[32*16*2], 32);
                    wmma::load_matrix_sync(frag_v[1], &smem_v[32*16*3], 32);
                }

                wmma::mma_sync(frag_out, frag_s[0], frag_v[0], frag_out);
                wmma::mma_sync(frag_out, frag_s[1], frag_v[1], frag_out);
                
                //加载下一次使用的数据
                if(block_id != to_block_end - 1 || b_bn != 1)
                {
                    FLOAT4(smem_v[smem_index]) = FLOAT4(c[data_b_start + next_bn*32*head_size + global_v_index_i*head_size+global_v_index_j]); 
                }

            }
        }
        // __syncthreads();

        wmma::store_matrix_sync(&out_temp[(tidy/2)*8][(tidy % 2)*32], frag_out, 64, wmma::mem_row_major);

        __syncthreads();

        const int index_x = (tidy%4)*8;
        const int index_y = tidx + (tidy/4)*32;
        // 结果写入global mem
        #pragma unroll
        for(int i=0;i<8;i+=1){
            out[data_offset_q+(a_bm*A_BM+index_x+i)*head_size+index_y] = __float2half(out_temp[(index_x+i)][index_y] / global_sum_scores[index_x+i]);
        }
        __syncthreads();
    }
}

template <class DataType>
__global__ void sparse_attention(DataType *a,  DataType *b,  DataType *c, 
    DataType *out, const int *to_select_index,const int *to_select_index_position, 
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

        const int to_block_start = to_select_index_position[from_block_id];
        const int to_block_end = to_select_index_position[from_block_id+1];

        int to_block_id = to_select_index[to_block_start];

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
            //     printf("%d %d %d %d\n",to_block_start,to_block_end,block_id,to_select_index[block_id]);
            // 计算KV块的起始位置
            
            // KV按照 32*64 的大小进行加载计算
            for(int b_bn=0;b_bn<block_size/32;b_bn++){
                
                // 计算Q*K
                cooperative_groups::wait(block_k);

                //32*64 64*32

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
                to_block_id = to_select_index[next_block_id];
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
                            float value1 = temp_score[tidy][tidx];
                            float value2 = temp_score[tidy+16][tidx];
                            float value3 = temp_score[tidy+8][tidx];
                            float value4 = temp_score[tidy+24][tidx];
                            temp_smem[tidy][tidx] = value1 + value2;
                            temp_smem[tidy+8][tidx] = value3 + value4;
                        }
                        else if(tidy < num){
                            float value1 = temp_smem[tidy][tidx];
                            float value2 = temp_smem[tidy+num][tidx];
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
    //(12,64)(32*8)  a is transpose 

//32*8
template <class DataType>
__global__ void sparse_attention_with_var(half *a,  half *b,  half *c, 
    half *out,const int *seq_len_info,const int *from_block_index, const int *from_block_index_position, const int *to_select_index,const int *to_select_index_position, const int batch_size,
    const int block_size,const int head_size,const int select_block_num){


    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    const int A_BM = 32;
    const int A_BK = 64;
    const int B_BK = 32;
    const int B_BN = 4;
    const int C_BK = 32;
    const int C_BN = 4;


    __shared__ half smem_q[32 * 64],smem_k[32 * 64],smem_v[32 * 64],smem_temp_half[64][32];
    __shared__ float temp_score[32][32],out_temp[32][64],max_score_diff[32];
    __shared__ half temp_score_half[32][32];

    __shared__ float global_sum_scores[32],pre_max_score[32],max_score[32],temp_smem[32];

    typedef cub::WarpReduce<float> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[8];
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_q[2];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_k[2];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_s_out;

    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_s[2];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_v[2];
    wmma::fragment<wmma::accumulator, 8, 32, 16, float> frag_out;


    float zero4[4] = {0.0f,0.0f,0.0f,0.0f};

    auto block_k = cooperative_groups::this_thread_block();
    auto block_v = cooperative_groups::this_thread_block();


    // 计算Q的起始位置
    const int compute_block_start = from_block_index_position[bidx];
    const int compute_block_end = from_block_index_position[bidx + 1];
    const int compute_block_num = compute_block_end - compute_block_start;
    for(int from_block_id_index = compute_block_start;from_block_id_index<compute_block_end;from_block_id_index++){
        int from_block_id = from_block_index[from_block_id_index];
        int seq_start_block_index = 0;
        int seq_block_len = 0;
        for(int i = 1;i<batch_size+1;i++){
            if(from_block_id >= seq_len_info[i]*12)
                continue;
            else{
                seq_start_block_index = seq_len_info[i-1];
                seq_block_len = seq_len_info[i] - seq_len_info[i-1];
                break;
            }
        }
        const int head_num = (from_block_id - 12 * seq_start_block_index)/seq_block_len;
        from_block_id = (from_block_id - 12 * seq_start_block_index)%seq_block_len;

        const int seq_start_index = 12 * seq_start_block_index * block_size * head_size + head_num * seq_block_len * block_size * head_size;
        const int data_offset_q = seq_start_index + from_block_id*block_size * head_size;


        // if(tidx == 0 && tidy == 0 ){
        //     printf("%d  %d  %d  %d  %d  %d\n",bidx,head_num,from_block_id,seq_start_block_index,seq_start_index,data_offset_a);
        // }
        
        for(int a_bm = 0; a_bm< block_size/A_BM; a_bm++){
            const int to_block_start = to_select_index_position[from_block_id + seq_start_block_index];
            const int to_block_end = to_select_index_position[from_block_id + seq_start_block_index + 1];

            int to_block_id = to_select_index[to_block_start];
            int data_b_start = seq_start_index + to_block_id * block_size * head_size; 

            const int smem_index =  32*8*tidy + tidx*8; // warp_size * 8

            // 加载Q的部分数据,32*64
            // Q  tidy
            //      --------------------
            //          0     |     1
            //      --------------------
            //          2     |     3
            //      --------------------
            //          4     |     5
            //      --------------------
            //          6     |     7
            //      --------------------
            const int smem_index_q = 32*8*tidy + (tidx/16)*16*8 + (tidx%16)*8;
            const int global_q_index_i = (a_bm*A_BM + (tidx%16)/2 + (tidy/2)*8);
            const int global_q_index_j = ((tidx%2)*8 + (tidx/16)*16 + (tidy%2)*32);
            FLOAT4(smem_q[smem_index_q]) = FLOAT4(a[data_offset_q+global_q_index_i*head_size+global_q_index_j]); 
    
            FLOAT4(out_temp[global_q_index_i - a_bm*A_BM][global_q_index_j]) = FLOAT4(zero4[0]);
            // wmma::fill_fragment(frag_s_out, 0.0);
            wmma::fill_fragment(frag_s_out, __float2half(0.0));

            wmma::fill_fragment(frag_out, 0.0);
    
    
            // KT 32*16  tidy  Q*K 8*16*32
            //      -----------------------------------------
            //          0     |     2   |   4     |     6   
            //      -----------------------------------------
            //          1     |     3   |   5     |     7   
            //      -----------------------------------------
    
            const int global_k_index_i = tidx/2 + (tidy%2)*16;
            const int global_k_index_j = (tidx%2)*8 + (tidy/2)*16;
            FLOAT4(smem_k[smem_index]) = FLOAT4(b[data_b_start + global_k_index_i*head_size+global_k_index_j]); 
    
    
            // V 16*32  tidy  S*K 8*16*32
            //      --------------------
            //          0     |     4
            //      --------------------
            //          1     |     5
            //      --------------------
            //          2     |     6
            //      --------------------
            //          3     |     7
            //      --------------------
            const int global_v_index_i = (tidx/4 + ((tidy/2)%2)*16 + (tidy%2)*8);
            const int global_v_index_j = ((tidx*8)%32 + ((tidy/2)/2)*32);
            FLOAT4(smem_v[smem_index]) = FLOAT4(c[data_b_start + global_v_index_i*head_size+global_v_index_j]); 
    
            wmma::load_matrix_sync(frag_q[0], &smem_q[tidy*2*8*16], 16);
            wmma::load_matrix_sync(frag_q[1], &smem_q[(tidy*2+1)*8*16], 16);
    
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
                // KV按照 32*64 的大小进行加载计算
                for(int b_bn=0;b_bn<block_size/32;b_bn++){
                    // 计算Q*K
                    // cooperative_groups::wait(block_k);
                    wmma::fill_fragment(frag_s_out, __float2half(0.0));
                    if(tidy % 2 == 0){
                        wmma::load_matrix_sync(frag_k[0], &smem_k[0], 16);
                        wmma::load_matrix_sync(frag_k[1], &smem_k[16*32], 16);
                    }
                    else{
                        wmma::load_matrix_sync(frag_k[0], &smem_k[16*32*2], 16);
                        wmma::load_matrix_sync(frag_k[1], &smem_k[16*32*3], 16);
                    }
                    wmma::mma_sync(frag_s_out, frag_q[0], frag_k[0], frag_s_out);
                    wmma::mma_sync(frag_s_out, frag_q[1], frag_k[1], frag_s_out);
                    wmma::store_matrix_sync(&smem_temp_half[(tidy/2)*8 + (tidy % 2)*32][0], frag_s_out, 32, wmma::mem_row_major);

                    __syncthreads();

                    for(int i=0;i<4;i++){
                        temp_score[i*8+tidy][tidx] = __half2float(smem_temp_half[i*8+tidy][tidx]) + __half2float(smem_temp_half[i*8+tidy+32][tidx]);
                    }

                    //加载下一次使用的数据
                    const int next_block_id = b_bn == 1 ? block_id+1:block_id;
                    const int next_bn = (b_bn + 1) & 1;
                    to_block_id = to_select_index[next_block_id];
                    int data_b_start = seq_start_index + to_block_id * block_size * head_size;

                    if(block_id != to_block_end - 1 || b_bn != 1)
                    {
                        FLOAT4(smem_k[smem_index]) = FLOAT4(b[data_b_start + next_bn*32*head_size + global_k_index_i*head_size+global_k_index_j]);

                    }

                    //计算最大值 rowmax
                    {
                        float value1 = temp_score[tidy][tidx];
                        float value2 = temp_score[tidy+16][tidx];
                        float value3 = temp_score[tidy+8][tidx];
                        float value4 = temp_score[tidy+24][tidx];

                        temp_smem[tidy] = WarpReduce(temp_storage[tidy]).Reduce(value1, cub::Max());
                        temp_smem[tidy+16] = WarpReduce(temp_storage[tidy]).Reduce(value2, cub::Max());
                        temp_smem[tidy+8] = WarpReduce(temp_storage[tidy]).Reduce(value3, cub::Max());
                        temp_smem[tidy+24] = WarpReduce(temp_storage[tidy]).Reduce(value4, cub::Max());

                        __syncthreads();
                        if(tidy == 0)
                        {
                            pre_max_score[tidx] = max_score[tidx];
                            max_score[tidx] = max_score[tidx]>temp_smem[tidx]?max_score[tidx]:temp_smem[tidx];
                            max_score_diff[tidx] = exp(pre_max_score[tidx] - max_score[tidx]);
                        }
                    }
                    
                    //计算差值
                    {
                        __syncthreads();
                        for(int i=0;i<4;i++)
                        {
                            float temp =  exp(temp_score[(i+tidy*4)][tidx] - max_score[tidy*4+i]);
                            temp_score[(i+tidy*4)][tidx] = temp;
                            temp_score_half[(i+tidy*4)][tidx] = __float2half(temp);
                        }

                        const int t = (tidy/2)*8+(tidx%4)*2;
                        for(int i=0;i<4;i++){
                            frag_out.x[i*2] *= max_score_diff[t];
                            frag_out.x[i*2+1] *= max_score_diff[t+1];
                        }
                        __syncthreads();  
                    }
                    {
                        float value1 = temp_score[tidy][tidx];
                        float value2 = temp_score[tidy+16][tidx];
                        float value3 = temp_score[tidy+8][tidx];
                        float value4 = temp_score[tidy+24][tidx];
                        
                        temp_smem[tidy] = WarpReduce(temp_storage[tidy]).Sum(value1);
                        temp_smem[tidy+16] = WarpReduce(temp_storage[tidy]).Sum(value2);
                        temp_smem[tidy+8] = WarpReduce(temp_storage[tidy]).Sum(value3);
                        temp_smem[tidy+24] = WarpReduce(temp_storage[tidy]).Sum(value4);
                        
                        __syncthreads();

                        if(tidy == 0)
                        {
                            global_sum_scores[tidx] *= max_score_diff[tidx];
                            global_sum_scores[tidx] += temp_smem[tidx];
                        }
                    }


                    // //计算S*V
                    wmma::load_matrix_sync(frag_s[0], &temp_score_half[(tidy/2)*8][0], 32);
                    wmma::load_matrix_sync(frag_s[1], &temp_score_half[(tidy/2)*8][16], 32);
                    if(tidy % 2 == 0){
                        wmma::load_matrix_sync(frag_v[0], &smem_v[0], 32);
                        wmma::load_matrix_sync(frag_v[1], &smem_v[32*16], 32);
                    }
                    else{
                        wmma::load_matrix_sync(frag_v[0], &smem_v[32*16*2], 32);
                        wmma::load_matrix_sync(frag_v[1], &smem_v[32*16*3], 32);
                    }

                    wmma::mma_sync(frag_out, frag_s[0], frag_v[0], frag_out);
                    wmma::mma_sync(frag_out, frag_s[1], frag_v[1], frag_out);
                    
                    __syncthreads();

                    //加载下一次使用的数据
                    if(block_id != to_block_end - 1 || b_bn != 1)
                    {
                        FLOAT4(smem_v[smem_index]) = FLOAT4(c[data_b_start + next_bn*32*head_size + global_v_index_i*head_size+global_v_index_j]); 
                    }

                }
            }

            wmma::store_matrix_sync(&out_temp[(tidy/2)*8][(tidy % 2)*32], frag_out, 64, wmma::mem_row_major);

            __syncthreads();
    
            const int index_x = (tidy%4)*8;
            const int index_y = tidx + (tidy/4)*32;
            // 结果写入global mem
            #pragma unroll
            for(int i=0;i<8;i+=1){
                out[data_offset_q+(a_bm*A_BM+index_x+i)*head_size+index_y] = __float2half(out_temp[(index_x+i)][index_y] / global_sum_scores[index_x+i]);
            }
            __syncthreads();
        }
    }
}

template <class DataType>
__global__ void sparse_attention_without_bank(half *a,  half *b,  half *c, 
    DataType *out,const int *seq_len_info,const int *from_block_index, const int *from_block_index_position, const int *to_select_index,const int *to_select_index_position, const int batch_size,
    const int block_size,const int head_size,const int select_block_num){


    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;


    const int A_BM = 32;
    const int A_BK = 64;
    const int B_BK = 32;
    const int B_BN = 4;
    const int C_BK = 32;
    const int C_BN = 4;
    const int pad = 8;

    __shared__ half smem_q[32][64+pad],smem_k[32][64+pad],smem_v[32][64+pad],smem_temp_half[64][32+pad];
    // __shared__ half smem_q[32][64+pad],smem_k[32 * 64],smem_v[32 * 64],smem_temp_half[64][32+pad];

    __shared__ float temp_score[32][32],out_temp[32][64+4],max_score_diff[32];
    __shared__ half temp_score_half[32][32+8];
    

    __shared__ float global_sum_scores[32],pre_max_score[32],max_score[32],temp_smem[32];

    typedef cub::WarpReduce<float> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[8];
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_q[2];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_k[2];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_s_out;

    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_s[2];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_v[2];
    wmma::fragment<wmma::accumulator, 8, 32, 16, float> frag_out;


    float4 zero4 = {0.0f,0.0f,0.0f,0.0f};

    auto block_k = cooperative_groups::this_thread_block();
    auto block_v = cooperative_groups::this_thread_block();


    // 计算Q的起始位置
    const int compute_block_start = from_block_index_position[bidx];
    const int compute_block_end = from_block_index_position[bidx + 1];
    const int compute_block_num = compute_block_end - compute_block_start;
    for(int from_block_id_index = compute_block_start;from_block_id_index<compute_block_end;from_block_id_index++){
        int from_block_id = from_block_index[from_block_id_index];
        int seq_start_block_index = 0;
        int seq_block_len = 0;
        for(int i = 1;i<batch_size+1;i++){
            if(from_block_id >= seq_len_info[i]*12)
                continue;
            else{
                seq_start_block_index = seq_len_info[i-1];
                seq_block_len = seq_len_info[i] - seq_len_info[i-1];
                break;
            }
        }
        const int head_num = (from_block_id - 12 * seq_start_block_index)/seq_block_len;
        from_block_id = (from_block_id - 12 * seq_start_block_index)%seq_block_len;

        const int seq_start_index = 12 * seq_start_block_index * block_size * head_size + head_num * seq_block_len * block_size * head_size;
        const int data_offset_q = seq_start_index + from_block_id*block_size * head_size;


        // if(tidx == 0 && tidy == 0 ){
        //     printf("%d  %d  %d  %d  %d  %d\n",bidx,head_num,from_block_id,seq_start_block_index,seq_start_index,data_offset_a);
        // }
        
        for(int a_bm = 0; a_bm< block_size/A_BM; a_bm++){
            const int to_block_start = to_select_index_position[from_block_id + seq_start_block_index];
            const int to_block_end = to_select_index_position[from_block_id + seq_start_block_index + 1];

            int to_block_id = to_select_index[to_block_start];
            int data_b_start = seq_start_index + to_block_id * block_size * head_size; 

            const int smem_index =  32*8*tidy + tidx*8; // warp_size * 8

            // 加载Q的部分数据,32*64
            // Q  tidy
            //      --------------------
            //          0     |     1
            //      --------------------
            //          2     |     3
            //      --------------------
            //          4     |     5
            //      --------------------
            //          6     |     7
            //      --------------------
            const int global_q_index_i =  tidy*4+tidx/8;
            const int global_q_index_j = (tidx%8)*8;
            FLOAT4(smem_q[global_q_index_i][global_q_index_j]) = FLOAT4(a[data_offset_q+(global_q_index_i+a_bm*A_BM)*head_size+global_q_index_j]);

            // const int smem_index_q = 32*8*tidy + (tidx/16)*16*8 + (tidx%16)*8;
            // const int global_q_index_i = (a_bm*A_BM + (tidx%16)/2 + (tidy/2)*8);
            // const int global_q_index_j = ((tidx%2)*8 + (tidx/16)*16 + (tidy%2)*32);
            // FLOAT4(smem_q[smem_index_q]) = FLOAT4(a[data_offset_q+global_q_index_i*head_size+global_q_index_j]); 
    
            // FLOAT4(out_temp[global_q_index_i][global_q_index_j]) = zero4;
            // FLOAT4(out_temp[global_q_index_i][global_q_index_j+4]) = zero4;

            // wmma::fill_fragment(frag_s_out, 0.0);
            wmma::fill_fragment(frag_s_out, __float2half(0.0));

            wmma::fill_fragment(frag_out, 0.0);
    
    
            // KT 32*16  tidy  Q*K 8*16*32
            //      -----------------------------------------
            //          0     |     2   |   4     |     6   
            //      -----------------------------------------
            //          1     |     3   |   5     |     7   
            //      -----------------------------------------

            // const int global_q_index_i = a_bm*A_BM + tidy*4+tidx/8;
            // const int global_q_index_j = (tidx%8)*8;
            FLOAT4(smem_k[global_q_index_i][global_q_index_j]) = FLOAT4(b[data_b_start+global_q_index_i*head_size+global_q_index_j]);
            // const int global_k_index_i = tidx/2 + (tidy%2)*16;
            // const int global_k_index_j = (tidx%2)*8 + (tidy/2)*16;
            // FLOAT4(smem_k[smem_index]) = FLOAT4(b[data_b_start + global_k_index_i*head_size+global_k_index_j]); 
    
    
            // V 16*32  tidy  S*K 8*16*32
            //      --------------------
            //          0     |     4
            //      --------------------
            //          1     |     5
            //      --------------------
            //          2     |     6
            //      --------------------
            //          3     |     7
            //      --------------------
            FLOAT4(smem_v[global_q_index_i][global_q_index_j]) = FLOAT4(c[data_b_start+global_q_index_i*head_size+global_q_index_j]);
            // const int global_v_index_i = (tidx/4 + ((tidy/2)%2)*16 + (tidy%2)*8);
            // const int global_v_index_j = ((tidx*8)%32 + ((tidy/2)/2)*32);
            // FLOAT4(smem_v[smem_index]) = FLOAT4(c[data_b_start + global_v_index_i*head_size+global_v_index_j]); 

            __syncthreads();
    
            wmma::load_matrix_sync(frag_q[0], &smem_q[(tidy/2)*8][(tidy%2)*32], 64+pad);
            wmma::load_matrix_sync(frag_q[1], &smem_q[(tidy/2)*8][(tidy%2)*32+16], 64+pad);
            // wmma::load_matrix_sync(frag_q[0], &smem_q[tidy*2*8*16], 16);
            // wmma::load_matrix_sync(frag_q[1], &smem_q[(tidy*2+1)*8*16], 16);
    
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
                // KV按照 32*64 的大小进行加载计算
                for(int b_bn=0;b_bn<block_size/32;b_bn++){
                    // 计算Q*K
                    // cooperative_groups::wait(block_k);
                    wmma::fill_fragment(frag_s_out, __float2half(0.0));
                    if(tidy % 2 == 0){
                        wmma::load_matrix_sync(frag_k[0], &smem_k[0][0], 64+pad);
                        wmma::load_matrix_sync(frag_k[1], &smem_k[0][16], 64+pad);
                    }
                    else{
                        wmma::load_matrix_sync(frag_k[0], &smem_k[0][32], 64+pad);
                        wmma::load_matrix_sync(frag_k[1], &smem_k[0][48], 64+pad);
                    }
                    // if(tidy % 2 == 0){
                    //     wmma::load_matrix_sync(frag_k[0], &smem_k[0], 16);
                    //     wmma::load_matrix_sync(frag_k[1], &smem_k[16*32], 16);
                    // }
                    // else{
                    //     wmma::load_matrix_sync(frag_k[0], &smem_k[16*32*2], 16);
                    //     wmma::load_matrix_sync(frag_k[1], &smem_k[16*32*3], 16);
                    // }
                    wmma::mma_sync(frag_s_out, frag_q[0], frag_k[0], frag_s_out);
                    wmma::mma_sync(frag_s_out, frag_q[1], frag_k[1], frag_s_out);
                    wmma::store_matrix_sync(&smem_temp_half[(tidy/2)*8 + (tidy % 2)*32][0], frag_s_out, 32+pad, wmma::mem_row_major);

                    __syncthreads();

                    for(int i=0;i<4;i++){
                        temp_score[i*8+tidy][tidx] = __half2float(smem_temp_half[i*8+tidy][tidx]) + __half2float(smem_temp_half[i*8+tidy+32][tidx]);
                    }

                    //加载下一次使用的数据
                    const int next_block_id = b_bn == 1 ? block_id+1:block_id;
                    const int next_bn = (b_bn + 1) & 1;
                    to_block_id = to_select_index[next_block_id];
                    int data_b_start = seq_start_index + to_block_id * block_size * head_size;

                    if(block_id != to_block_end - 1 || b_bn != 1)
                    {
                        // FLOAT4(smem_k[smem_index]) = FLOAT4(b[data_b_start + next_bn*32*head_size + global_k_index_i*head_size+global_k_index_j]);

                        FLOAT4(smem_k[global_q_index_i][global_q_index_j]) = FLOAT4(b[data_b_start+next_bn*32*head_size+global_q_index_i*head_size+global_q_index_j]);
                    }

                    //计算最大值 rowmax
                    {
                        float value1 = temp_score[tidy][tidx];
                        float value2 = temp_score[tidy+16][tidx];
                        float value3 = temp_score[tidy+8][tidx];
                        float value4 = temp_score[tidy+24][tidx];

                        temp_smem[tidy] = WarpReduce(temp_storage[tidy]).Reduce(value1, cub::Max());
                        temp_smem[tidy+16] = WarpReduce(temp_storage[tidy]).Reduce(value2, cub::Max());
                        temp_smem[tidy+8] = WarpReduce(temp_storage[tidy]).Reduce(value3, cub::Max());
                        temp_smem[tidy+24] = WarpReduce(temp_storage[tidy]).Reduce(value4, cub::Max());

                        __syncthreads();
                        if(tidy == 0)
                        {
                            pre_max_score[tidx] = max_score[tidx];
                            max_score[tidx] = max_score[tidx]>temp_smem[tidx]?max_score[tidx]:temp_smem[tidx];
                            max_score_diff[tidx] = exp(pre_max_score[tidx] - max_score[tidx]);
                        }
                    }
                    
                    //计算差值
                    {
                        __syncthreads();
                        for(int i=0;i<4;i++)
                        {
                            float temp =  exp(temp_score[(i+tidy*4)][tidx] - max_score[tidy*4+i]);
                            temp_score[(i+tidy*4)][tidx] = temp;
                            temp_score_half[(i+tidy*4)][tidx] = __float2half(temp);
                        }

                        const int t = (tidy/2)*8+(tidx%4)*2;
                        for(int i=0;i<4;i++){
                            frag_out.x[i*2] *= max_score_diff[t];
                            frag_out.x[i*2+1] *= max_score_diff[t+1];
                        }
                        // if(tidy == 0 && bidx == 0 && bidy == 0 && a_bm == 0 && block_id==to_block_start && b_bn == 1)
                        // {
                        //     for(int i=0;i<8;i++){
                        //         printf("%d %d %f \n",tidy,tidx,frag_out.x[i]);
                        //     }
                        // }
                       
                        // if(tidy == 0)
                        // wmma::store_matrix_sync(&out_temp[(tidy/2)*8][(tidy % 2)*32], frag_out, 64, wmma::mem_row_major);

                        // __syncthreads();

                        // if(tidx == 0 && tidy == 0 && bidx == 0 && bidy == 0 && a_bm == 0 && block_id==to_block_start && b_bn == 1)
                        // {
                        //     for(int i=0;i<32;i++)
                        //     {
                        //         for(int j=0;j<64;j++)
                        //             printf("%f ",out_temp[i][j]);
                        //         printf("\n");
                        //     }
                        // }
                        __syncthreads();  
                    }
                    {
                        float value1 = temp_score[tidy][tidx];
                        float value2 = temp_score[tidy+16][tidx];
                        float value3 = temp_score[tidy+8][tidx];
                        float value4 = temp_score[tidy+24][tidx];
                        
                        temp_smem[tidy] = WarpReduce(temp_storage[tidy]).Sum(value1);
                        temp_smem[tidy+16] = WarpReduce(temp_storage[tidy]).Sum(value2);
                        temp_smem[tidy+8] = WarpReduce(temp_storage[tidy]).Sum(value3);
                        temp_smem[tidy+24] = WarpReduce(temp_storage[tidy]).Sum(value4);
                        
                        __syncthreads();

                        if(tidy == 0)
                        {
                            global_sum_scores[tidx] *= max_score_diff[tidx];
                            global_sum_scores[tidx] += temp_smem[tidx];
                        }
                    }


                    // //计算S*V
                    wmma::load_matrix_sync(frag_s[0], &temp_score_half[(tidy/2)*8][0], 32+8);
                    wmma::load_matrix_sync(frag_s[1], &temp_score_half[(tidy/2)*8][16], 32+8);
                    if(tidy % 2 == 0){
                        wmma::load_matrix_sync(frag_v[0], &smem_v[0][0], 64+pad);
                        wmma::load_matrix_sync(frag_v[1], &smem_v[16][0], 64+pad);
                    }
                    else{
                        wmma::load_matrix_sync(frag_v[0], &smem_v[0][32], 64+pad);
                        wmma::load_matrix_sync(frag_v[1], &smem_v[16][32], 64+pad);
                    }
                    // if(tidy % 2 == 0){
                    //     wmma::load_matrix_sync(frag_v[0], &smem_v[0], 32);
                    //     wmma::load_matrix_sync(frag_v[1], &smem_v[32*16], 32);
                    // }
                    // else{
                    //     wmma::load_matrix_sync(frag_v[0], &smem_v[32*16*2], 32);
                    //     wmma::load_matrix_sync(frag_v[1], &smem_v[32*16*3], 32);
                    // }

                    wmma::mma_sync(frag_out, frag_s[0], frag_v[0], frag_out);
                    wmma::mma_sync(frag_out, frag_s[1], frag_v[1], frag_out);
                    
                    __syncthreads();

                    //加载下一次使用的数据
                    if(block_id != to_block_end - 1 || b_bn != 1)
                    {
                        // FLOAT4(smem_v[smem_index]) = FLOAT4(c[data_b_start + next_bn*32*head_size + global_v_index_i*head_size+global_v_index_j]); 
                        FLOAT4(smem_v[global_q_index_i][global_q_index_j]) = FLOAT4(c[data_b_start+next_bn*32*head_size+global_q_index_i*head_size+global_q_index_j]);
                    }

                }
            }

            wmma::store_matrix_sync(&out_temp[(tidy/2)*8][(tidy % 2)*32], frag_out, 64+4, wmma::mem_row_major);

            __syncthreads();
    
            const int index_x = (tidy%4)*8;
            const int index_y = tidx + (tidy/4)*32;
            // 结果写入global mem
            #pragma unroll
            for(int i=0;i<8;i+=1){
                out[data_offset_q+(a_bm*A_BM+index_x+i)*head_size+index_y] = out_temp[(index_x+i)][index_y] / global_sum_scores[index_x+i];
            }
            __syncthreads();
        }
    }
}

// template <class DataType>
// __global__ void sparse_attention_(half *a,  half *b,  half *c, 
//     half *out,const int *seq_len_info,const int *from_block_index, const int *from_block_index_position, const int *to_select_index,const int *to_select_index_position, const int batch_size,
//     const int block_size,const int head_size){

//     const int tidy = threadIdx.y;
//     const int tidx = threadIdx.x;
//     const int bidx = blockIdx.x;
//     const int bidy = blockIdx.y;

//     const int A_BM = 32;
//     const int A_BK = 64;
//     const int B_BK = 32;
//     const int B_BN = 4;
//     const int C_BK = 32;
//     const int C_BN = 4;
//     const int pad = 8;

//     typedef cub::WarpReduce<float> WarpReduce;
//     __shared__  typename WarpReduce::TempStorage temp_storage[4];
//     __shared__  float global_sum_scores[64],pre_max_score[64],max_score[64];  
//     __shared__  float temp_smem[16],max_score_diff[16];
//     __shared__  half smem_q[2][16][64+pad],smem_k[64][64+pad],smem_v[64][64+pad],smem_temp_score[16][64+pad],out_temp[64][64+pad];


//     wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_q[4];
//     wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_k[4];
//     wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_s_out;

//     wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_s[4];
//     wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_v[4];
//     wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_out;


//     float4 zero4 = {0.0f,0.0f,0.0f,0.0f};

//     // 计算Q的起始位置
//     const int compute_block_start = from_block_index_position[bidx];
//     const int compute_block_end = from_block_index_position[bidx + 1];
//     const int compute_block_num = compute_block_end - compute_block_start;
//     for(int from_block_id_index = compute_block_start;from_block_id_index<compute_block_end;from_block_id_index++){
//         int from_block_id = from_block_index[from_block_id_index];
//         int seq_start_block_index = 0;
//         int seq_block_len = 0;
//         for(int i = 1;i<batch_size+1;i++){
//             if(from_block_id >= seq_len_info[i]*12)
//                 continue;
//             else{
//                 seq_start_block_index = seq_len_info[i-1];
//                 seq_block_len = seq_len_info[i] - seq_len_info[i-1];
//                 break;
//             }
//         }
//         const int head_num = (from_block_id - 12 * seq_start_block_index)/seq_block_len;
//         from_block_id = (from_block_id - 12 * seq_start_block_index)%seq_block_len;

//         const int seq_start_index = 12 * seq_start_block_index * block_size * head_size + head_num * seq_block_len * block_size * head_size;
//         const int data_offset_q = seq_start_index + from_block_id*block_size * head_size;

//         const int to_block_start = to_select_index_position[from_block_id + seq_start_block_index];
//         const int to_block_end = to_select_index_position[from_block_id + seq_start_block_index + 1];
        
//         const int smem_index_i =  tidy*4+tidx/8;
//         const int smem_index_j = (tidx%8)*8;
        

//         FLOAT4(smem_q[0][smem_index_i][smem_index_j]) = FLOAT4(a[data_offset_q+ smem_index_i*head_size+smem_index_j]);

//         int to_block_id = to_select_index[0];
//         int data_k_start = seq_start_index + to_block_id * block_size * head_size;
//         for(int i=0;i<4;i++){
//             FLOAT4(smem_k[i*16 + smem_index_i][smem_index_j]) = FLOAT4(b[data_k_start+(i*16 + smem_index_i)*head_size+smem_index_j]);
//             FLOAT4(smem_v[i*16 + smem_index_i][smem_index_j]) = FLOAT4(c[data_k_start+(i*16 + smem_index_i)*head_size+smem_index_j]);
//         }

//         for(int i=0;i<4;i++){
//             FLOAT4(out_temp[i*16 + smem_index_i][smem_index_j]) = FLOAT4(zero4);
//         }

//         wmma::fill_fragment(frag_out, __float2half(0.0));
//         if(tidy < 2){
//             global_sum_scores[tidy*32+tidx] = 0.0;
//             pre_max_score[tidy*32+tidx] = -1000.0f;
//             max_score[tidy*32+tidx] = -1000.0f;
//         }

//         if(tidy == 0 && tidx < 16)
//         {
//             temp_smem[tidx] = 0.0;
//             max_score_diff[tidx] = 1;
//         }

//         // 遍历K、V的每一个Block进行计算
//         for(int block_id_index=to_block_start;block_id_index<to_block_end;block_id_index++){

//             __syncthreads();

//             for(int i=0;i<4;i++){
//                 wmma::load_matrix_sync(frag_k[i], &smem_k[(tidy%2)*32][i*16], 64+pad);
//                 wmma::load_matrix_sync(frag_v[i], &smem_v[i*16][(tidy%2)*32], 64+pad);
//             }

//             if(block_id_index != to_block_end - 1){
//                 to_block_id = to_select_index[block_id_index+1];
//                 data_k_start = seq_start_index + to_block_id * block_size * head_size;

//                 for(int i=0;i<4;i++){
//                     FLOAT4(smem_k[i*16 + smem_index_i][smem_index_j]) = FLOAT4(b[data_k_start+(i*16 + smem_index_i)*head_size+smem_index_j]);
//                     FLOAT4(smem_v[i*16 + smem_index_i][smem_index_j]) = FLOAT4(c[data_k_start+(i*16 + smem_index_i)*head_size+smem_index_j]);
//                 }
//             }

//             for(int from_block_part_index = 0;from_block_part_index<block_size;from_block_part_index+=16){
//                 wmma::fill_fragment(frag_s_out, __float2half(0.0));

//                 for(int i=0;i<4;i++)
//                 {
//                     wmma::load_matrix_sync(frag_q[i], &smem_q[(from_block_part_index/16)&1][(tidy/2)*8][i*16], 64+pad);
//                 }
//                 for(int i=0;i<4;i++){
//                     wmma::mma_sync(frag_s_out, frag_q[i], frag_k[i], frag_s_out);
//                 }

//                 if(block_id_index != to_block_end - 1 || from_block_part_index != 48)
//                 FLOAT4(smem_q[(from_block_part_index/16 + 1)&1][smem_index_i][smem_index_j]) = FLOAT4(a[data_offset_q+((from_block_part_index+16)%64+ smem_index_i)*head_size+smem_index_j]);

//                 //load next data

//                 wmma::store_matrix_sync(&smem_temp_score[(tidy/2)*8][(tidy%2)*32], frag_s_out, 64+pad, wmma::mem_row_major);
                
//                 __syncthreads();

//                 // 计算最大值 rowmax
//                 {
//                     float value_h2[8];
//                     float score_value[4];
//                     for(int i=0;i<4;i++)
//                     {
//                         value_h2[i*2] = __half2float(smem_temp_score[tidy+i*4][tidx*2]);
//                         value_h2[i*2+1] = __half2float(smem_temp_score[tidy+i*4][tidx*2+1]);
//                     }

//                     for(int i=0;i<4;i++)
//                     {
//                         score_value[i] = value_h2[i*2] > value_h2[i*2+1] ? value_h2[i*2] : value_h2[i*2+1];
//                     }

//                     for(int i=0;i<4;i++)
//                     {
//                         float t = float(score_value[i]);
//                         temp_smem[tidy + i*4] = WarpReduce(temp_storage[tidy]).Reduce(t, cub::Max());
//                     }

//                     __syncthreads();
//                     if(tidy == 0 && tidx < 16)
//                     {
//                         int idx = tidx+from_block_part_index;
//                         pre_max_score[idx] = max_score[idx];
//                         max_score[idx] = max_score[idx] > temp_smem[tidx]?max_score[idx]:temp_smem[tidx];
//                         max_score_diff[tidx] = exp(pre_max_score[idx]-max_score[idx]);
//                     }
                    
//                     float out_temp_value[8];
//                     for(int i=0;i<4;i++)
//                     {
//                         out_temp_value[i*2] =  __half2float(out_temp[from_block_part_index+i*4+tidy][tidx*2]);
//                         out_temp_value[i*2+1] =  __half2float(out_temp[from_block_part_index+i*4+tidy][tidx*2 + 1]);
//                     }
//                     __syncthreads();

//                     float value_after_exp[8];
//                     for(int i=0;i<4;i++){
//                         float max_value_h = max_score[tidy+i*4 + from_block_part_index];
//                         float max_score_diff_h2 = max_score_diff[tidy+i*4];
//                         value_after_exp[i*2] = exp(value_h2[i*2]-max_value_h);
//                         value_after_exp[i*2 + 1] = exp(value_h2[i*2 + 1]-max_value_h);
//                         smem_temp_score[tidy+i*4][tidx*2] = __float2half( value_after_exp[i*2]);
//                         smem_temp_score[tidy+i*4][tidx*2 + 1] = __float2half( value_after_exp[i*2 + 1]);

//                         out_temp[from_block_part_index+i*4+tidy][tidx*2] = __float2half(out_temp_value[i*2] * max_score_diff_h2);
//                         out_temp[from_block_part_index+i*4+tidy][tidx*2 + 1] = __float2half(out_temp_value[i*2 + 1] * max_score_diff_h2);
//                     }

//                     for(int i=0;i<4;i++){
//                         float sum_temp = value_after_exp[i*2] + value_after_exp[i*2+1];
//                         temp_smem[tidy + i*4] = WarpReduce(temp_storage[tidy]).Sum(sum_temp);
//                     }

//                     __syncthreads();

//                     if(tidy == 0 && tidx < 16)
//                     {
//                         int idx = tidx+from_block_part_index;
//                         global_sum_scores[idx] *= max_score_diff[tidx];
//                         global_sum_scores[idx] += temp_smem[tidx];
//                     }

//                 }

//                 wmma::load_matrix_sync(frag_out,&out_temp[(tidy/2)*8 + from_block_part_index][(tidy%2)*32],64+pad,wmma::mem_row_major);
//                 for(int i=0;i<4;i++)
//                 {
//                     wmma::load_matrix_sync(frag_s[i], &smem_temp_score[(tidy/2)*8][i*16], 64+pad);
//                 }

//                 for(int i=0;i<4;i++){
//                     wmma::mma_sync(frag_out, frag_s[i], frag_v[i], frag_out);
//                 }

//                 wmma::store_matrix_sync(&out_temp[(tidy/2)*8 + from_block_part_index][(tidy%2)*32],frag_out,64+pad,wmma::mem_row_major);

                
//             }
//         }
//         __syncthreads();
//         for(int i=0;i<16;i++){
//             float sum_score_value = global_sum_scores[tidy*16+i];
//             float2 out_temp_value = __half22float2(HALF2(out_temp[tidy*16+i][tidx*2]));

//             out_temp[tidy*16+i][tidx*2] = __float2half(out_temp_value.x/sum_score_value);
//             out_temp[tidy*16+i][tidx*2 + 1] = __float2half(out_temp_value.y/sum_score_value);
//         }
//         __syncthreads();
//         for(int i=0;i<4;i++)
//             FLOAT4(out[data_offset_q+(i*16 + smem_index_i)*head_size+smem_index_j]) = FLOAT4(out_temp[i*16 + smem_index_i][smem_index_j]);

//     }
// }

//32*4
template <class DataType>
__global__ void sparse_attention_with_q_double_buffer(half *a,  half *b,  half *c, 
    half *out,const int *seq_len_info,const int *from_block_index, const int *from_block_index_position, const int *to_select_index,const int *to_select_index_position, const int batch_size,
    const int block_size,const int head_size,const int select_block_num){

    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const int A_BM = 32;
    const int A_BK = 64;
    const int B_BK = 32;
    const int B_BN = 4;
    const int C_BK = 32;
    const int C_BN = 4;
    const int pad = 8;

    typedef cub::WarpReduce<float> WarpReduce;
    __shared__  typename WarpReduce::TempStorage temp_storage[4];
    __shared__  float global_sum_scores[64],pre_max_score[64],max_score[64];  
    __shared__  float temp_smem[16],max_score_diff[16];
    __shared__  half smem_q[2][16][64+pad],smem_k[64][64+pad],smem_v[64][64+pad],smem_temp_score[16][64+pad],out_temp[64][64+pad];


    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_q[4];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_k[4];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_s_out;

    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_s[4];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_v[4];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_out;


    float4 zero4 = {0.0f,0.0f,0.0f,0.0f};

    // 计算Q的起始位置
    const int compute_block_start = from_block_index_position[bidx];
    const int compute_block_end = from_block_index_position[bidx + 1];
    const int compute_block_num = compute_block_end - compute_block_start;
    for(int from_block_id_index = compute_block_start;from_block_id_index<compute_block_end;from_block_id_index++){
        int from_block_id = from_block_index[from_block_id_index];
        int seq_start_block_index = 0;
        int seq_block_len = 0;
        for(int i = 1;i<batch_size+1;i++){
            if(from_block_id >= seq_len_info[i]*12)
                continue;
            else{
                seq_start_block_index = seq_len_info[i-1];
                seq_block_len = seq_len_info[i] - seq_len_info[i-1];
                break;
            }
        }
        const int head_num = (from_block_id - 12 * seq_start_block_index)/seq_block_len;
        from_block_id = (from_block_id - 12 * seq_start_block_index)%seq_block_len;

        const int seq_start_index = 12 * seq_start_block_index * block_size * head_size + head_num * seq_block_len * block_size * head_size;
        const int data_offset_q = seq_start_index + from_block_id*block_size * head_size;

        const int to_block_start = to_select_index_position[from_block_id + seq_start_block_index];
        const int to_block_end = to_select_index_position[from_block_id + seq_start_block_index + 1];
        
        const int smem_index_i =  tidy*4+tidx/8;
        const int smem_index_j = (tidx%8)*8;
        

        FLOAT4(smem_q[0][smem_index_i][smem_index_j]) = FLOAT4(a[data_offset_q+ smem_index_i*head_size+smem_index_j]);

        int to_block_id = to_select_index[0];
        int data_k_start = seq_start_index + to_block_id * block_size * head_size;
        for(int i=0;i<4;i++){
            FLOAT4(smem_k[i*16 + smem_index_i][smem_index_j]) = FLOAT4(b[data_k_start+(i*16 + smem_index_i)*head_size+smem_index_j]);
            FLOAT4(smem_v[i*16 + smem_index_i][smem_index_j]) = FLOAT4(c[data_k_start+(i*16 + smem_index_i)*head_size+smem_index_j]);
        }

        for(int i=0;i<4;i++){
            FLOAT4(out_temp[i*16 + smem_index_i][smem_index_j]) = FLOAT4(zero4);
        }

        wmma::fill_fragment(frag_out, __float2half(0.0));
        if(tidy < 2){
            global_sum_scores[tidy*32+tidx] = 0.0;
            pre_max_score[tidy*32+tidx] = -1000.0f;
            max_score[tidy*32+tidx] = -1000.0f;
        }

        if(tidy == 0 && tidx < 16)
        {
            temp_smem[tidx] = 0.0;
            max_score_diff[tidx] = 1;
        }

        // 遍历K、V的每一个Block进行计算
        for(int block_id_index=to_block_start;block_id_index<to_block_end;block_id_index++){

            __syncthreads();

            for(int i=0;i<4;i++){
                wmma::load_matrix_sync(frag_k[i], &smem_k[(tidy%2)*32][i*16], 64+pad);
                wmma::load_matrix_sync(frag_v[i], &smem_v[i*16][(tidy%2)*32], 64+pad);
            }

            if(block_id_index != to_block_end - 1){
                to_block_id = to_select_index[block_id_index+1];
                data_k_start = seq_start_index + to_block_id * block_size * head_size;

                for(int i=0;i<4;i++){
                    FLOAT4(smem_k[i*16 + smem_index_i][smem_index_j]) = FLOAT4(b[data_k_start+(i*16 + smem_index_i)*head_size+smem_index_j]);
                    FLOAT4(smem_v[i*16 + smem_index_i][smem_index_j]) = FLOAT4(c[data_k_start+(i*16 + smem_index_i)*head_size+smem_index_j]);
                }
            }

            for(int from_block_part_index = 0;from_block_part_index<block_size;from_block_part_index+=16){
                wmma::fill_fragment(frag_s_out, __float2half(0.0));

                for(int i=0;i<4;i++)
                {
                    wmma::load_matrix_sync(frag_q[i], &smem_q[(from_block_part_index/16)&1][(tidy/2)*8][i*16], 64+pad);
                }
                
                for(int i=0;i<4;i++){
                    wmma::mma_sync(frag_s_out, frag_q[i], frag_k[i], frag_s_out);
                }

                if(block_id_index != to_block_end - 1 || from_block_part_index != 48)
                FLOAT4(smem_q[(from_block_part_index/16 + 1)&1][smem_index_i][smem_index_j]) = FLOAT4(a[data_offset_q+((from_block_part_index+16)%64+ smem_index_i)*head_size+smem_index_j]);

                //load next data

                wmma::store_matrix_sync(&smem_temp_score[(tidy/2)*8][(tidy%2)*32], frag_s_out, 64+pad, wmma::mem_row_major);
                
                __syncthreads();

                // 计算最大值 rowmax
                {
                    float value_h2[8];
                    float score_value[4];
                    for(int i=0;i<4;i++)
                    {
                        value_h2[i*2] = __half2float(smem_temp_score[tidy+i*4][tidx*2]);
                        value_h2[i*2+1] = __half2float(smem_temp_score[tidy+i*4][tidx*2+1]);
                    }

                    for(int i=0;i<4;i++)
                    {
                        score_value[i] = value_h2[i*2] > value_h2[i*2+1] ? value_h2[i*2] : value_h2[i*2+1];
                    }

                    for(int i=0;i<4;i++)
                    {
                        float t = float(score_value[i]);
                        temp_smem[tidy + i*4] = WarpReduce(temp_storage[tidy]).Reduce(t, cub::Max());
                    }

                    __syncthreads();
                    if(tidy == 0 && tidx < 16)
                    {
                        int idx = tidx+from_block_part_index;
                        pre_max_score[idx] = max_score[idx];
                        max_score[idx] = max_score[idx] > temp_smem[tidx]?max_score[idx]:temp_smem[tidx];
                        max_score_diff[tidx] = exp(pre_max_score[idx]-max_score[idx]);
                    }
                    
                    float out_temp_value[8];
                    for(int i=0;i<4;i++)
                    {
                        out_temp_value[i*2] =  __half2float(out_temp[from_block_part_index+i*4+tidy][tidx*2]);
                        out_temp_value[i*2+1] =  __half2float(out_temp[from_block_part_index+i*4+tidy][tidx*2 + 1]);
                    }
                    __syncthreads();

                    float value_after_exp[8];
                    for(int i=0;i<4;i++){
                        float max_value_h = max_score[tidy+i*4 + from_block_part_index];
                        float max_score_diff_h2 = max_score_diff[tidy+i*4];
                        value_after_exp[i*2] = exp(value_h2[i*2]-max_value_h);
                        value_after_exp[i*2 + 1] = exp(value_h2[i*2 + 1]-max_value_h);
                        smem_temp_score[tidy+i*4][tidx*2] = __float2half( value_after_exp[i*2]);
                        smem_temp_score[tidy+i*4][tidx*2 + 1] = __float2half( value_after_exp[i*2 + 1]);

                        
                        out_temp[from_block_part_index+i*4+tidy][tidx*2] = __float2half(out_temp_value[i*2] * max_score_diff_h2);
                        out_temp[from_block_part_index+i*4+tidy][tidx*2 + 1] = __float2half(out_temp_value[i*2 + 1] * max_score_diff_h2);
                    }

                    for(int i=0;i<4;i++){
                        float sum_temp = value_after_exp[i*2] + value_after_exp[i*2+1];
                        temp_smem[tidy + i*4] = WarpReduce(temp_storage[tidy]).Sum(sum_temp);
                    }

                    __syncthreads();

                    if(tidy == 0 && tidx < 16)
                    {
                        int idx = tidx+from_block_part_index;
                        global_sum_scores[idx] *= max_score_diff[tidx];
                        global_sum_scores[idx] += temp_smem[tidx];
                    }

                }

                wmma::load_matrix_sync(frag_out,&out_temp[(tidy/2)*8 + from_block_part_index][(tidy%2)*32],64+pad,wmma::mem_row_major);
                for(int i=0;i<4;i++)
                {
                    wmma::load_matrix_sync(frag_s[i], &smem_temp_score[(tidy/2)*8][i*16], 64+pad);
                }

                for(int i=0;i<4;i++){
                    wmma::mma_sync(frag_out, frag_s[i], frag_v[i], frag_out);
                }

                wmma::store_matrix_sync(&out_temp[(tidy/2)*8 + from_block_part_index][(tidy%2)*32],frag_out,64+pad,wmma::mem_row_major);

                
            }
        }
        __syncthreads();
        for(int i=0;i<16;i++){
            float sum_score_value = global_sum_scores[tidy*16+i];
            float2 out_temp_value = __half22float2(HALF2(out_temp[tidy*16+i][0])[tidx]);

            out_temp[tidy*16+i][tidx*2] = __float2half(out_temp_value.x/sum_score_value);
            out_temp[tidy*16+i][tidx*2 + 1] = __float2half(out_temp_value.y/sum_score_value);
        }
        __syncthreads();
        for(int i=0;i<4;i++)
            FLOAT4(out[data_offset_q+(i*16 + smem_index_i)*head_size+smem_index_j]) = FLOAT4(out_temp[i*16 + smem_index_i][smem_index_j]);

    }
}
      
void test_gemm_1(half *a, half *b,half *c, float *out,int *seq_len_info,int *from_select_index,int *from_select_index_position,int *to_select_index,int *to_select_index_position,int block_limit, int block_num, int head_num,int block_size,int head_size)
{


    // cudaEvent_t start,stop;
    // cudaEventCreate( &start );
    // cudaEventCreate( &stop ) ;
    // cudaEventRecord( start, 0 ) ;
    // 修改成最大线程块数量 80 * 2
    sparse_attention_without_bank<float><<<dim3(block_limit),dim3(32,8)>>>(a,b,c,out,seq_len_info,from_select_index,from_select_index_position,to_select_index,to_select_index_position,2,64,64,11);

    // cudaEventRecord(stop,0);
    // float elapsedTime;
    // cudaEventSynchronize(stop);
    // cudaDeviceSynchronize();
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // printf( "Time to generate:  %f ms\n", elapsedTime );
}
// template <class DataType>
// __global__ void sparse_attention_(half *a,  half *b,  half *c, 
//     half *out,const int *seq_len_info,const int *from_block_index, const int *from_block_index_position, const int *to_select_index,const int *to_select_index_position, const int batch_size,
//     const int block_size,const int head_size,const int select_block_num){

//     const int tidy = threadIdx.y;
//     const int tidx = threadIdx.x;
//     const int bidx = blockIdx.x;
//     const int bidy = blockIdx.y;

//     const int A_BM = 32;
//     const int A_BK = 64;
//     const int B_BK = 32;
//     const int B_BN = 4;
//     const int C_BK = 32;
//     const int C_BN = 4;
//     const int pad = 8;

//     typedef cub::WarpReduce<float> WarpReduce;
//     __shared__  typename WarpReduce::TempStorage temp_storage[4];
//     __shared__  float global_sum_scores[64],pre_max_score[64],max_score[64];  
//     __shared__  float temp_smem[16],max_score_diff[16];
//     __shared__  half smem_q[2][16][64+pad],smem_k[64][64+pad],smem_v[64][64+pad],smem_temp_score[16][64+pad],out_temp[64][64+pad];


//     wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_q[4];
//     wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_k[4];
//     wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_s_out;

//     wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_s[4];
//     wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_v[4];
//     wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_out;

//     const int smem_index_i =  tidy*4+tidx/8;
//     const int smem_index_j = (tidx%8)*8;

//     const unsigned long load_q_smem_addr[2] = {__cvta_generic_to_shared(smem_q[0]) + (smem_index_i*(64+pad)+smem_index_j)*2,__cvta_generic_to_shared(smem_q[1]) + (smem_index_i*(64+pad)+smem_index_j)*2};
//     const int load_k_smem_addr = __cvta_generic_to_shared(smem_k[0]) + (smem_index_i*(64+pad)+smem_index_j)*2;
//     const int load_v_smem_addr = __cvta_generic_to_shared(smem_v[0]) + (smem_index_i*(64+pad)+smem_index_j)*2;

    

//     float4 zero4 = {0.0f,0.0f,0.0f,0.0f};

//     // 计算Q的起始位置
//     const int compute_block_start = from_block_index_position[bidx];
//     const int compute_block_end = from_block_index_position[bidx + 1];
//     const int compute_block_num = compute_block_end - compute_block_start;
//     for(int from_block_id_index = compute_block_start;from_block_id_index<compute_block_end;from_block_id_index++){
//         int from_block_id = from_block_index[from_block_id_index];
//         int seq_start_block_index = 0;
//         int seq_block_len = 0;
//         for(int i = 1;i<batch_size+1;i++){
//             if(from_block_id >= seq_len_info[i]*12)
//                 continue;
//             else{
//                 seq_start_block_index = seq_len_info[i-1];
//                 seq_block_len = seq_len_info[i] - seq_len_info[i-1];
//                 break;
//             }
//         }
//         const int head_num = (from_block_id - 12 * seq_start_block_index)/seq_block_len;
//         from_block_id = (from_block_id - 12 * seq_start_block_index)%seq_block_len;

//         const int seq_start_index = 12 * seq_start_block_index * block_size * head_size + head_num * seq_block_len * block_size * head_size;
//         const int data_offset_q = seq_start_index + from_block_id*block_size * head_size;

//         const int to_block_start = to_select_index_position[from_block_id + seq_start_block_index];
//         const int to_block_end = to_select_index_position[from_block_id + seq_start_block_index + 1];
        
//         int load_q_smem_addr_now = load_q_smem_addr[0];

//         int load_q_gmem_addr = data_offset_q+ smem_index_i*head_size+smem_index_j;
//         asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_q_smem_addr_now), "l"(&a[load_q_gmem_addr]));

//         int to_block_id = to_select_index[0];
//         int data_k_start = seq_start_index + to_block_id * block_size * head_size;
//         int load_k_gmem_addr = data_k_start+ smem_index_i*head_size+smem_index_j;
//         for(int i=0;i<64;i+=16){
//             asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_k_smem_addr + i*(64+pad)*2), "l"(&b[load_k_gmem_addr+i*head_size]));
//             asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_v_smem_addr + i*(64+pad)*2), "l"(&c[load_k_gmem_addr+i*head_size]));
//         }

//         for(int i=0;i<4;i++){
//             FLOAT4(out_temp[i*16 + smem_index_i][smem_index_j]) = FLOAT4(zero4);
            
//         }
//         wmma::fill_fragment(frag_out, __float2half(0.0));

//         if(tidy < 2){
//             global_sum_scores[tidy*32+tidx] = 0.0;
//             pre_max_score[tidy*32+tidx] = 0.0f;
//             max_score[tidy*32+tidx] = 0.0f;
//         }

//         if(tidy == 0 && tidx < 16)
//         {
//             temp_smem[tidx] = 0.0;
//             max_score_diff[tidx] = 1;
//         }

//         // 遍历K、V的每一个Block进行计算
//         for(int block_id_index=to_block_start;block_id_index<to_block_end;block_id_index++){

//             __syncthreads();
//             asm ("cp.async.commit_group;\n" ::);
//             asm ("cp.async.wait_group 0;\n" ::);

//             for(int i=0;i<4;i++){
//                 wmma::load_matrix_sync(frag_k[i], &smem_k[(tidy%2)*32][i*16], 64+pad);
//                 wmma::load_matrix_sync(frag_v[i], &smem_v[i*16][(tidy%2)*32], 64+pad);
//             }

//             if(block_id_index != to_block_end - 1){
//                 to_block_id = to_select_index[block_id_index+1];
//                 data_k_start = seq_start_index + to_block_id * block_size * head_size;
//                 load_k_gmem_addr = data_k_start+ smem_index_i*head_size+smem_index_j;
//                 for(int i=0;i<4;i++){
//                 asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_k_smem_addr + i*(64+pad)*2), "l"(&b[load_k_gmem_addr+i*head_size]));
//                 asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_v_smem_addr + i*(64+pad)*2), "l"(&c[load_k_gmem_addr+i*head_size]));
//                 }
//             }

//             for(int from_block_part_index = 0;from_block_part_index<block_size;from_block_part_index+=16){
//                 wmma::fill_fragment(frag_s_out, __float2half(0.0));

//                 for(int i=0;i<4;i++)
//                 {
//                     wmma::load_matrix_sync(frag_q[i], &smem_q[(from_block_part_index/16)&1][(tidy/2)*8][i*16], 64+pad);
//                 }
//                 for(int i=0;i<4;i++){
//                     wmma::mma_sync(frag_s_out, frag_q[i], frag_k[i], frag_s_out);
//                 }

//                 if(block_id_index != to_block_end - 1 || from_block_part_index != 48)
//                 {
//                     int load_q_smem_addr_now = load_q_smem_addr[(from_block_part_index/16 + 1)&1];
//                     asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_q_smem_addr_now), "l"(&a[load_q_gmem_addr + ((from_block_part_index+16)%64)*head_size]));
//                 }
//                 //load next data

//                 wmma::store_matrix_sync(&smem_temp_score[(tidy/2)*8][(tidy%2)*32], frag_s_out, 64+pad, wmma::mem_row_major);
                
//                 __syncthreads();

//                 // 计算最大值 rowmax
//                 {
//                     float value_h2[8];
//                     float score_value[4];
//                     for(int i=0;i<4;i++)
//                     {
//                         value_h2[i*2] = __half2float(smem_temp_score[tidy+i*4][tidx*2]);
//                         value_h2[i*2+1] = __half2float(smem_temp_score[tidy+i*4][tidx*2+1]);
//                     }

//                     for(int i=0;i<4;i++)
//                     {
//                         score_value[i] = value_h2[i*2] > value_h2[i*2+1] ? value_h2[i*2] : value_h2[i*2+1];
//                     }

//                     for(int i=0;i<4;i++)
//                     {
//                         float t = float(score_value[i]);
//                         temp_smem[tidy + i*4] = WarpReduce(temp_storage[tidy]).Reduce(t, cub::Max());
//                     }

//                     float max_values_f[4];
                    
//                     for(int i=0;i<4;i++)
//                         max_values_f[4] = max_score[tidy+i*4 + from_block_part_index];

//                     __syncthreads();
//                     if(tidy == 0 && tidx < 16)
//                     {
//                         int idx = tidx+from_block_part_index;
//                         pre_max_score[idx] = max_score[idx];
//                         max_score[idx] = max_score[idx] > temp_smem[tidx]?max_score[idx]:temp_smem[tidx];
//                         max_score_diff[tidx] = exp(pre_max_score[idx]-max_score[idx]);
//                     }

//                     float out_temp_value[8];
//                     for(int i=0;i<4;i++)
//                     {
//                         out_temp_value[i*2] =  __half2float(out_temp[from_block_part_index+i*4+tidy][tidx*2]);
//                         out_temp_value[i*2+1] =  __half2float(out_temp[from_block_part_index+i*4+tidy][tidx*2 + 1]);
//                     }
                    
//                     __syncthreads();
//                     const int t = (tidy/2)*4+(tidx%4)*2;
//                     float value_after_exp[8];
//                     for(int i=0;i<4;i++){
//                         float max_value_h = max_score[tidy+i*4 + from_block_part_index];
//                         float max_score_diff_h2 = max_score_diff[tidy+i*4];
//                         value_after_exp[i*2] = exp(value_h2[i*2]-max_value_h);
//                         value_after_exp[i*2 + 1] = exp(value_h2[i*2 + 1]-max_value_h);
//                         smem_temp_score[tidy+i*4][tidx*2] = __float2half( value_after_exp[i*2]);
//                         smem_temp_score[tidy+i*4][tidx*2 + 1] = __float2half( value_after_exp[i*2 + 1]);

//                         out_temp[from_block_part_index+i*4+tidy][tidx*2] = __float2half(out_temp_value[i*2] * max_score_diff_h2);
//                         out_temp[from_block_part_index+i*4+tidy][tidx*2 + 1] = __float2half(out_temp_value[i*2 + 1] * max_score_diff_h2);

//                         // frag_out[from_block_part_index/16].x[i*2] *= max_score_diff[t];
//                         // frag_out[from_block_part_index/16].x[i*2+1] *= max_score_diff[t+1];
//                     }
//                     // for(int i=0;i<4;i++){
//                     //     float max_score_diff_h2 = max_score_diff[tidy+i*4];
    
//                     //     smem_temp_score[tidy+i*4][tidx*2] = __float2half( value_after_exp[i*2] * max_score_diff_h2);
//                     //     smem_temp_score[tidy+i*4][tidx*2 + 1] = __float2half( value_after_exp[i*2 + 1] * max_score_diff_h2);

                        
//                     //     out_temp[from_block_part_index+i*4+tidy][tidx*2] = out_temp_value[i*2] * max_score_diff_h2;
//                     //     out_temp[from_block_part_index+i*4+tidy][tidx*2 + 1] = out_temp_value[i*2 + 1] * max_score_diff_h2;
//                     // }

//                     for(int i=0;i<4;i++){
//                         float sum_temp = value_after_exp[i*2] + value_after_exp[i*2+1];
//                         temp_smem[tidy + i*4] = WarpReduce(temp_storage[tidy]).Sum(sum_temp);
//                     }

//                     __syncthreads();

//                     if(tidy == 0 && tidx < 16)
//                     {
//                         int idx = tidx+from_block_part_index;
//                         global_sum_scores[idx] *= max_score_diff[tidx];
//                         global_sum_scores[idx] += temp_smem[tidx];
//                     }

//                 }

//                 wmma::load_matrix_sync(frag_out,&out_temp[(tidy/2)*8 + from_block_part_index][(tidy%2)*32],64+pad,wmma::mem_row_major);
//                 for(int i=0;i<4;i++)
//                 {
//                     wmma::load_matrix_sync(frag_s[i], &smem_temp_score[(tidy/2)*8][i*16], 64+pad);
//                 }

//                 for(int i=0;i<4;i++){
//                     wmma::mma_sync(frag_out, frag_s[i], frag_v[i], frag_out);
//                 }

//                 wmma::store_matrix_sync(&out_temp[(tidy/2)*8 + from_block_part_index][(tidy%2)*32],frag_out,64+pad,wmma::mem_row_major);

//                 asm ("cp.async.commit_group;\n" ::);
//                 asm ("cp.async.wait_group 0;\n" ::);
//             }
//         }

//         __syncthreads();
//         for(int i=0;i<16;i++){
//             float sum_score_value = global_sum_scores[tidy*16+i];
//             float2 out_temp_value = __half22float2(HALF2(out_temp[tidy*16+i][0])[tidx]);

//             out_temp[tidy*16+i][tidx*2] = __float2half(out_temp_value.x/sum_score_value);
//             out_temp[tidy*16+i][tidx*2 + 1] = __float2half(out_temp_value.y/sum_score_value);
//         }
//         __syncthreads();
//         for(int i=0;i<4;i++)
//             FLOAT4(out[data_offset_q+(i*16 + smem_index_i)*head_size+smem_index_j]) = FLOAT4(out_temp[i*16 + smem_index_i][smem_index_j]);

//     }
// }

template <class DataType>
__global__ void sparse_attention_banlanced(half *a,  half *b,  half *c, 
    half *out,const int *seq_len_info,const int *from_block_index, const int *from_block_index_position, const int *to_select_index,const int *to_select_index_position, const int batch_size,
    const int block_size,const int head_size){
    
    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const int A_BM = 32;
    const int A_BK = 64;
    const int B_BK = 32;
    const int B_BN = 4;
    const int C_BK = 32;
    const int C_BN = 4;
    const int pad = 8;

    typedef cub::WarpReduce<float> WarpReduce;
    __shared__  typename WarpReduce::TempStorage temp_storage[4];
    __shared__  float global_sum_scores[64],pre_max_score[64],max_score[64];  
    __shared__  float temp_smem[32],max_score_diff[16],sum_temp[64];
    __shared__  half smem_q[2][16][64+pad],smem_k[64][64+pad],smem_v[64][64+pad],smem_temp_score[16][64+pad],out_temp[64][64+pad];


    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_q[4];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_k[4];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_s_out;

    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_s[4];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_v[4];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_out;

    const int smem_index_i =  tidy*4+tidx/8;
    const int smem_index_j = (tidx%8)*8;

    const int load_k_smem_addr = __cvta_generic_to_shared(smem_k[0]) + (smem_index_i*(64+pad)+smem_index_j)*2;
    const int load_v_smem_addr = __cvta_generic_to_shared(smem_v[0]) + (smem_index_i*(64+pad)+smem_index_j)*2;
    const unsigned long load_q_smem_addr[2] = {__cvta_generic_to_shared(smem_q[0]) + (smem_index_i*(64+pad)+smem_index_j)*2,__cvta_generic_to_shared(smem_q[1]) + (smem_index_i*(64+pad)+smem_index_j)*2};

    float4 zero4 = {0.0f,0.0f,0.0f,0.0f};

    const int compute_block_start = from_block_index_position[bidx];
    const int compute_block_end = from_block_index_position[bidx + 1];
    const int compute_block_num = compute_block_end - compute_block_start;
    for(int from_block_id_index = compute_block_start;from_block_id_index<compute_block_end;from_block_id_index++){
        int from_block_id = from_block_index[from_block_id_index];
        int seq_start_block_index = 0;
        int seq_block_len = 0;
        for(int i=1;i<batch_size+1;i++){
            if(from_block_id >= seq_len_info[i]*12)
                continue;
            else{
                seq_start_block_index = seq_len_info[i-1]; //开始的block id
                seq_block_len = seq_len_info[i] - seq_len_info[i-1]; // seq 拥有的block id
                break;
            }
        }

        //还原原始的headnum和blockid
        const int head_num = (from_block_id - 12*seq_start_block_index)/seq_block_len;
        from_block_id = (from_block_id - 12*seq_start_block_index)%seq_block_len;

        const int seq_start_index = 12*seq_start_block_index*block_size*head_size + head_num*seq_block_len*block_size*head_size;
        const int data_offset_q = seq_start_index + from_block_id*block_size*head_size;

        const int to_block_start = to_select_index_position[from_block_id+seq_start_block_index];
        const int to_block_end = to_select_index_position[from_block_id+seq_start_block_index + 1];

        int load_q_smem_addr_now = load_q_smem_addr[0];

        int load_q_gmem_addr = data_offset_q+ smem_index_i*head_size+smem_index_j;
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_q_smem_addr_now), "l"(&a[load_q_gmem_addr]));

        
        for(int i=0;i<4;i++){
            FLOAT4(out_temp[i*16 + smem_index_i][smem_index_j]) = FLOAT4(zero4);
        }

        wmma::fill_fragment(frag_out, __float2half(0.0));

        int to_block_id = to_select_index[0];
        int data_k_start = seq_start_index + to_block_id * block_size * head_size;
        int load_k_gmem_addr = data_k_start+ smem_index_i*head_size+smem_index_j;
        for(int i=0;i<64;i+=16){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_k_smem_addr + i*(64+pad)*2), "l"(&b[load_k_gmem_addr+i*head_size]));
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_v_smem_addr + i*(64+pad)*2), "l"(&c[load_k_gmem_addr+i*head_size]));
        }

        if(tidy == 0 && tidx < 16)
        {
            max_score_diff[tidx] = 1.0f;
        }

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);


        for(int block_id_index=to_block_start;block_id_index<to_block_end;block_id_index++){
            #pragma unroll
            for(int i=0;i<4;i++){
                wmma::load_matrix_sync(frag_k[i], &smem_k[(tidy%2)*32][i*16], 64+pad);
                wmma::load_matrix_sync(frag_v[i], &smem_v[i*16][(tidy%2)*32], 64+pad);
            }

            __syncthreads();
            if(block_id_index != to_block_end - 1){
                to_block_id = to_select_index[block_id_index];
                data_k_start = seq_start_index + to_block_id * block_size * head_size;
                
                int load_k_gmem_addr = data_k_start+ smem_index_i*head_size+smem_index_j;
                for(int i=0;i<64;i+=16){
                    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_k_smem_addr + i*(64+pad)*2), "l"(&b[load_k_gmem_addr+i*head_size]));
                    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_v_smem_addr + i*(64+pad)*2), "l"(&c[load_k_gmem_addr+i*head_size]));
                }
            }


            for(int from_block_part_index = 0;from_block_part_index<block_size;from_block_part_index+=16){
                
                // FLOAT4(smem_q[smem_index_i][smem_index_j]) = FLOAT4(a[data_offset_q+ smem_index_i*head_size+smem_index_j + from_block_part_index*head_size]);

                #pragma unroll
                for(int i=0;i<4;i++)
                {
                    wmma::load_matrix_sync(frag_q[i], &smem_q[(from_block_part_index/16)&1][(tidy/2)*8][i*16], 64+pad);
                }

                if(block_id_index != to_block_end - 1 || from_block_part_index != 48)
                {
                    int load_q_smem_addr_now = load_q_smem_addr[(from_block_part_index/16 + 1)&1];
                    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_q_smem_addr_now), "l"(&a[load_q_gmem_addr + ((from_block_part_index+16)%64)*head_size]));
                }
                wmma::fill_fragment(frag_s_out, __float2half(0.0));

                for(int i=0;i<4;i++){
                    wmma::mma_sync(frag_s_out, frag_q[i], frag_k[i], frag_s_out);
                }

                // if(tidy == from_block_part_index/16)
                wmma::store_matrix_sync(&smem_temp_score[(tidy/2)*8][(tidy%2)*32], frag_s_out, 64+pad, wmma::mem_row_major);

                __syncthreads();

                // 计算最大值 rowmax
                float value_h2[8];
                {
                    float score_value[4];
                    #pragma unroll
                    for(int i=0;i<4;i++)
                    {
                        value_h2[i*2] = __half2float(smem_temp_score[tidy+i*4][tidx*2]);
                        value_h2[i*2+1] = __half2float(smem_temp_score[tidy+i*4][tidx*2+1]);
                    }

                    #pragma unroll
                    for(int i=0;i<4;i++)
                    {
                        score_value[i] = value_h2[i*2] > value_h2[i*2+1] ? value_h2[i*2] : value_h2[i*2+1];
                    }

                    #pragma unroll
                    for(int i=0;i<4;i++)
                    {
                        temp_smem[tidy + i*4] = WarpReduce(temp_storage[tidy]).Reduce(score_value[i], cub::Max());
                    }
                }
                __syncthreads();
                if(tidy == 1 && tidx < 16)
                {
                    int idx = tidx+from_block_part_index;
                    pre_max_score[idx] = max_score[idx];
                    max_score[idx] = max(max_score[idx],temp_smem[tidx]);
                    max_score_diff[tidx] = exp(pre_max_score[idx]-max_score[idx]);
                }
                __syncthreads();

                half2 max_s[4];
                half2 value[4];
                for(int i=0;i<4;i++){
                    value[i] =  HALF2(smem_temp_score[tidy*4+i][0])[tidx];
                    max_s[i] = __half2half2(__float2half(max_score[tidy*4 + i + from_block_part_index]));
                }
                half2 diff_x = __half2half2(__float2half(max_score_diff[tidy*4 + tidx/8 ]));
                float4 out_t = FLOAT4(out_temp[from_block_part_index + tidy*4+(tidx*8)/64][(tidx*8)%64]);


                for(int i=0;i<4;i++){
                    half2 t = h2exp(__hsub2(value[i],max_s[i]));
                    HALF2(out_temp[from_block_part_index + tidy*4+(tidx*8)/64][(tidx*8)%64])[i] = __hmul2(HALF2(out_t)[i],diff_x);

                    HALF2(smem_temp_score[tidy*4+i][0])[tidx] = t;
                    float v_ = __half2float(__hadd(HALF(t)[0],HALF(t)[1]));
                    temp_smem[tidy*4 + i] = WarpReduce(temp_storage[tidy]).Sum(v_);
                }

                __syncthreads();

                wmma::load_matrix_sync(frag_out,&out_temp[(tidy/2)*8 + from_block_part_index][(tidy%2)*32],64+pad,wmma::mem_row_major);

                for(int i=0;i<4;i++)
                {
                    wmma::load_matrix_sync(frag_s[i], &smem_temp_score[(tidy/2)*8][i*16], 64+pad);
                }


                for(int i=0;i<4;i++){
                    wmma::mma_sync(frag_out, frag_s[i], frag_v[i], frag_out);
                }

                wmma::store_matrix_sync(&out_temp[(tidy/2)*8 + from_block_part_index][(tidy%2)*32],frag_out,64+pad,wmma::mem_row_major);

                if(tidy == 0 && tidx < 16)
                {
                    int idx = tidx+from_block_part_index;
                    global_sum_scores[idx] *= max_score_diff[tidx];
                    global_sum_scores[idx] += temp_smem[tidx];
                }

                asm ("cp.async.commit_group;\n" ::);
                asm ("cp.async.wait_group 0;\n" ::);
            }
        }

        // if(tidx == 0 && tidy == 0 && from_block_id == 0 && head_num == 0){
        //     for(int i=0;i<64;i++)
        //         printf("%f ",global_sum_scores[i]);
        //     printf("\n");
        // }


        __syncthreads();
        for(int i=0;i<16;i++){
            float sum_score_value = global_sum_scores[tidy*16+i];
            float2 out_temp_value = __half22float2(HALF2(out_temp[tidy*16+i][0])[tidx]);

            out_temp[tidy*16+i][tidx*2] = __float2half(out_temp_value.x/sum_score_value);
            out_temp[tidy*16+i][tidx*2 + 1] = __float2half(out_temp_value.y/sum_score_value);

        }

        __syncthreads();
        for(int i=0;i<4;i++)
            FLOAT4(out[data_offset_q+(i*16 + smem_index_i)*head_size+smem_index_j]) = FLOAT4(out_temp[i*16 + smem_index_i][smem_index_j]);
    }
}  


template <class DataType>
__global__ void sparse_attention_non_balanced(half *a,  half *b,  half *c, 
    half *out,const int *seq_len_info,const int *from_block_index, const int *from_block_index_position, const int *to_select_index,const int *to_select_index_position, const int batch_size,
    const int block_size,const int head_size){

    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;
    const int head_num = blockIdx.x;
    int from_block_id = blockIdx.y;
    const int g_dimy = gridDim.y;

    const int A_BM = 32;
    const int A_BK = 64;
    const int B_BK = 32;
    const int B_BN = 4;
    const int C_BK = 32;
    const int C_BN = 4;
    const int pad = 8;

    typedef cub::WarpReduce<float> WarpReduce;
    __shared__  typename WarpReduce::TempStorage temp_storage[4];
    __shared__  float global_sum_scores[64],pre_max_score[64],max_score[64];  
    __shared__  float temp_smem[32],max_score_diff[16],sum_temp[64];
    __shared__  half smem_q[2][16][64+pad],smem_k[64][64+pad],smem_v[64][64+pad],smem_temp_score[16][64+pad],out_temp[64][64+pad];


    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_q[4];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_k[4];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_s_out;

    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_s[4];
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_v[4];
    wmma::fragment<wmma::accumulator, 8, 32, 16, half> frag_out;

    const int smem_index_i =  tidy*4+tidx/8;
    const int smem_index_j = (tidx%8)*8;

    const unsigned long load_q_smem_addr[2] = {__cvta_generic_to_shared(smem_q[0]) + (smem_index_i*(64+pad)+smem_index_j)*2,__cvta_generic_to_shared(smem_q[1]) + (smem_index_i*(64+pad)+smem_index_j)*2};
    const int load_k_smem_addr = __cvta_generic_to_shared(smem_k[0]) + (smem_index_i*(64+pad)+smem_index_j)*2;
    const int load_v_smem_addr = __cvta_generic_to_shared(smem_v[0]) + (smem_index_i*(64+pad)+smem_index_j)*2;

    

    float4 zero4 = {0.0f,0.0f,0.0f,0.0f};

    // 计算Q的起始位置

        int seq_start_block_index = 0;
        int seq_block_len = 0;
        for(int i = 1;i<batch_size+1;i++){
            if(from_block_id >= seq_len_info[i])
                continue;
            else{
                seq_start_block_index = seq_len_info[i-1];
                seq_block_len = seq_len_info[i] - seq_len_info[i-1];
                break;
            }
        }

        from_block_id -= seq_start_block_index;

        const int seq_start_index = 12*seq_start_block_index*block_size*head_size + head_num*seq_block_len*block_size*head_size;
        const int data_offset_q = seq_start_index + from_block_id*block_size*head_size;

        const int to_block_start = to_select_index_position[from_block_id + seq_start_block_index];
        const int to_block_end = to_select_index_position[from_block_id + seq_start_block_index + 1];
        
        int load_q_smem_addr_now = load_q_smem_addr[0];

        int load_q_gmem_addr = data_offset_q+ smem_index_i*head_size+smem_index_j;
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_q_smem_addr_now), "l"(&a[load_q_gmem_addr]));

        for(int i=0;i<4;i++){
            FLOAT4(out_temp[i*16 + smem_index_i][smem_index_j]) = FLOAT4(zero4);
        }
        wmma::fill_fragment(frag_out, __float2half(0.0));

        int to_block_id = to_select_index[0];
        int data_k_start = seq_start_index + to_block_id * block_size * head_size;
        int load_k_gmem_addr = data_k_start+ smem_index_i*head_size+smem_index_j;
        for(int i=0;i<64;i+=16){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_k_smem_addr + i*(64+pad)*2), "l"(&b[load_k_gmem_addr+i*head_size]));
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_v_smem_addr + i*(64+pad)*2), "l"(&c[load_k_gmem_addr+i*head_size]));
        }

        if(tidy < 2){
            global_sum_scores[tidy*32+tidx] = 0.0;
            pre_max_score[tidy*32+tidx] = 0.0f;
            max_score[tidy*32+tidx] = 0.0f;
        }

        if(tidy == 0 && tidx < 16)
        {
            temp_smem[tidx] = 0.0;
            max_score_diff[tidx] = 1.0f;
        }

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        // 遍历K、V的每一个Block进行计算
        for(int block_id_index=to_block_start;block_id_index<to_block_end;block_id_index++){
            __syncthreads();

            #pragma unroll
            for(int i=0;i<4;i++){
                wmma::load_matrix_sync(frag_k[i], &smem_k[(tidy%2)*32][i*16], 64+pad);
                wmma::load_matrix_sync(frag_v[i], &smem_v[i*16][(tidy%2)*32], 64+pad);
            }

            if(block_id_index != to_block_end - 1){
                to_block_id = to_select_index[block_id_index];
                data_k_start = seq_start_index + to_block_id * block_size * head_size;
                
                int load_k_gmem_addr = data_k_start+ smem_index_i*head_size+smem_index_j;
                for(int i=0;i<64;i+=16){
                    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_k_smem_addr + i*(64+pad)*2), "l"(&b[load_k_gmem_addr+i*head_size]));
                    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_v_smem_addr + i*(64+pad)*2), "l"(&c[load_k_gmem_addr+i*head_size]));
                }
            }

            for(int from_block_part_index = 0;from_block_part_index<block_size;from_block_part_index+=16){

                #pragma unroll
                for(int i=0;i<4;i++)
                {
                    wmma::load_matrix_sync(frag_q[i], &smem_q[(from_block_part_index/16)&1][(tidy/2)*8][i*16], 64+pad);
                }

                if(block_id_index != to_block_end - 1 || from_block_part_index != 48)
                {
                    int load_q_smem_addr_now = load_q_smem_addr[(from_block_part_index/16 + 1)&1];
                    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_q_smem_addr_now), "l"(&a[load_q_gmem_addr + ((from_block_part_index+16)%64)*head_size]));
                }
                wmma::fill_fragment(frag_s_out, __float2half(0.0));

                for(int i=0;i<4;i++){
                    wmma::mma_sync(frag_s_out, frag_q[i], frag_k[i], frag_s_out);
                }
                //load next data

                wmma::store_matrix_sync(&smem_temp_score[(tidy/2)*8][(tidy%2)*32], frag_s_out, 64+pad, wmma::mem_row_major);
                
                __syncthreads();

                // 计算最大值 rowmax
                float value_h2[8];
                {
                    float score_value[4];
                    #pragma unroll
                    for(int i=0;i<4;i++)
                    {
                        value_h2[i*2] = __half2float(smem_temp_score[tidy+i*4][tidx*2]);
                        value_h2[i*2+1] = __half2float(smem_temp_score[tidy+i*4][tidx*2+1]);
                    }

                    #pragma unroll
                    for(int i=0;i<4;i++)
                    {
                        score_value[i] = value_h2[i*2] > value_h2[i*2+1] ? value_h2[i*2] : value_h2[i*2+1];
                    }

                    #pragma unroll
                    for(int i=0;i<4;i++)
                    {
                        temp_smem[tidy + i*4] = WarpReduce(temp_storage[tidy]).Reduce(score_value[i], cub::Max());
                    }
                }

                __syncthreads();
                if(tidy == 1 && tidx < 16)
                {
                    int idx = tidx+from_block_part_index;
                    pre_max_score[idx] = max_score[idx];
                    max_score[idx] = max(max_score[idx],temp_smem[tidx]);
                    max_score_diff[tidx] = exp(pre_max_score[idx]-max_score[idx]);
                }
                __syncthreads();

                half2 max_s[4];
                half2 value[4];
                for(int i=0;i<4;i++){
                    value[i] =  HALF2(smem_temp_score[tidy*4+i][0])[tidx];
                    max_s[i] = __half2half2(__float2half(max_score[tidy*4 + i + from_block_part_index]));
                }
                half2 diff_x = __half2half2(__float2half(max_score_diff[tidy*4 + tidx/8 ]));
                float4 out_t = FLOAT4(out_temp[from_block_part_index + tidy*4+(tidx*8)/64][(tidx*8)%64]);


                for(int i=0;i<4;i++){
                    half2 t = h2exp(__hsub2(value[i],max_s[i]));
                    HALF2(out_temp[from_block_part_index + tidy*4+(tidx*8)/64][(tidx*8)%64])[i] = __hmul2(HALF2(out_t)[i],diff_x);

                    HALF2(smem_temp_score[tidy*4+i][0])[tidx] = t;
                    float v_ = __half2float(__hadd(HALF(t)[0],HALF(t)[1]));
                    temp_smem[tidy*4 + i] = WarpReduce(temp_storage[tidy]).Sum(v_);
                }
                __syncthreads();

                wmma::load_matrix_sync(frag_out,&out_temp[(tidy/2)*8 + from_block_part_index][(tidy%2)*32],64+pad,wmma::mem_row_major);

                for(int i=0;i<4;i++)
                {
                    wmma::load_matrix_sync(frag_s[i], &smem_temp_score[(tidy/2)*8][i*16], 64+pad);
                }


                for(int i=0;i<4;i++){
                    wmma::mma_sync(frag_out, frag_s[i], frag_v[i], frag_out);
                }
                wmma::store_matrix_sync(&out_temp[(tidy/2)*8 + from_block_part_index][(tidy%2)*32],frag_out,64+pad,wmma::mem_row_major);

                if(tidy == 0 && tidx < 16)
                {
                    int idx = tidx+from_block_part_index;
                    global_sum_scores[idx] *= max_score_diff[tidx];
                    global_sum_scores[idx] += temp_smem[tidx];
                }

                asm ("cp.async.commit_group;\n" ::);
                asm ("cp.async.wait_group 0;\n" ::);
            }
        }

        __syncthreads();
        for(int i=0;i<16;i++){
            float sum_score_value = global_sum_scores[tidy*16+i];
            float2 out_temp_value = __half22float2(HALF2(out_temp[tidy*16+i][0])[tidx]);

            out_temp[tidy*16+i][tidx*2] = __float2half(out_temp_value.x/sum_score_value);
            out_temp[tidy*16+i][tidx*2 + 1] = __float2half(out_temp_value.y/sum_score_value);

        }
        __syncthreads();
        for(int i=0;i<4;i++)
            FLOAT4(out[data_offset_q+(i*16 + smem_index_i)*head_size+smem_index_j]) = FLOAT4(out_temp[i*16 + smem_index_i][smem_index_j]);

}


void test_gemm_1(half *a, half *b,half *c, half *out,int batch_size,int *seq_len_info,int *from_select_index,int *from_select_index_position,int *to_select_index,int *to_select_index_position,int block_limit, int block_num, int head_num,int block_size,int head_size,std::map<std::string,float>& info,bool balanced)
{


    // cudaEvent_t start,stop;
    // cudaEventCreate( &start );
    // cudaEventCreate( &stop ) ;
    // cudaEventRecord( start, 0 ) ;
    // 修改成最大线程块数量 80 * 2

    auto start_time = std::chrono::system_clock::now();

    if(balanced)
        sparse_attention_banlanced<half><<<dim3(block_limit),dim3(32,4)>>>(a,b,c,out,seq_len_info,from_select_index,from_select_index_position,to_select_index,to_select_index_position,batch_size,block_size,head_size);
    else{
        sparse_attention_non_balanced<half><<<dim3(head_num,block_num),dim3(32,4)>>>(a,b,c,out,seq_len_info,from_select_index,from_select_index_position,to_select_index,to_select_index_position,batch_size,block_size,head_size);
    }

    auto end_time = std::chrono::system_clock::now();
    if(info.find("attention") != info.end())
    {    auto dura = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count();
        info["attention"] += dura;
    }
    // cudaEventRecord(stop,0);
    // float elapsedTime;
    // cudaEventSynchronize(stop);
    // cudaDeviceSynchronize();
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // printf( "Time to generate:  %f ms\n", elapsedTime );
}

void test_gemm_(float *a, float *b,float *c, float *out,int *to_select_index,int *to_select_index_position, int block_num, int head_num,int block_size,int head_size)
{
    // std::cout<<*a<<std::endl;
    // sparse_attention<float><<<dim3(block_num,head_num),dim3(11,32)>>>(a,b,c,out,to_select_index,64,64);

    cudaEvent_t start,stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop ) ;
    // test_gemm<float><<<1,dim3(11,32)>>>(a,b,c,m,n,k,64);

    cudaEventRecord( start, 0 ) ;
    sparse_attention<float><<<dim3(head_num,block_num),dim3(32,8)>>>(a,b,c,out,to_select_index,to_select_index_position,64,64,11);

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


}
}
}