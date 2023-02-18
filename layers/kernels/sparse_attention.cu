#include "kernel.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
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
        wmma::fill_fragment(frag_s_out, 0.0);
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
                wmma::fill_fragment(frag_s_out, 0.0);
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
                    temp_score[i*8+tidy][tidx] = __half2float(smem_temp_half[i*8+tidy][tidx] + smem_temp_half[i*8+tidy+32][tidx]);
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
            out[data_offset_q+(a_bm*A_BM+index_x+i)*head_size+index_y] = out_temp[(index_x+i)][index_y] / global_sum_scores[index_x+i];
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
template <class DataType>
__global__ void sparse_attention_(DataType *a,  DataType *b,  DataType *c, 
    DataType *out,const int *seq_len_info,const int *from_block_index, const int *from_block_index_position, const int *to_select_index,const int *to_select_index_position, const int batch_size,
    const int block_size,const int head_size,const int select_block_num){


    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int b_dimx = 8;
    const int g_dimy = gridDim.y;

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
            wmma::fill_fragment(frag_s_out, 0.0);
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
                    wmma::fill_fragment(frag_s_out, 0.0);
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
                        temp_score[i*8+tidy][tidx] = __half2float(smem_temp_half[i*8+tidy][tidx] + smem_temp_half[i*8+tidy+32][tidx]);
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
                out[data_offset_q+(a_bm*A_BM+index_x+i)*head_size+index_y] = out_temp[(index_x+i)][index_y] / global_sum_scores[index_x+i];
            }
            __syncthreads();
        }
    }
}              
      
void test_gemm_1(half *a, half *b,half *c, half *out,int *seq_len_info,int *from_select_index,int *from_select_index_position,int *to_select_index,int *to_select_index_position, int block_num, int head_num,int block_size,int head_size)
{
    // std::cout<<*a<<std::endl;
    // sparse_attention<float><<<dim3(block_num,head_num),dim3(11,32)>>>(a,b,c,out,to_select_index,64,64);

    cudaEvent_t start,stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop ) ;
    // test_gemm<float><<<1,dim3(11,32)>>>(a,b,c,m,n,k,64);

    cudaEventRecord( start, 0 ) ;
    // 修改成最大线程块数量 80 * 2
    sparse_attention_<half><<<dim3(160),dim3(32,8)>>>(a,b,c,out,seq_len_info,from_select_index,from_select_index_position,to_select_index,to_select_index_position,2,64,64,11);

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

void test_gemm_(half *a, half *b,half *c, half *out,int *to_select_index,int *to_select_index_position, int block_num, int head_num,int block_size,int head_size)
{
    // std::cout<<*a<<std::endl;
    // sparse_attention<float><<<dim3(block_num,head_num),dim3(11,32)>>>(a,b,c,out,to_select_index,64,64);

    cudaEvent_t start,stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop ) ;
    // test_gemm<float><<<1,dim3(11,32)>>>(a,b,c,m,n,k,64);

    cudaEventRecord( start, 0 ) ;
    sparse_attention_with_tensor_core<half><<<dim3(head_num,block_num),dim3(32,8)>>>(a,b,c,out,to_select_index,to_select_index_position,64,64,11);

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