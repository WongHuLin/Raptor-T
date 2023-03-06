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
#include <cub/cub.cuh>
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FLOAT(pointer) (reinterpret_cast<float*>(&(pointer)))
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define HALF(pointer) (reinterpret_cast<half*>(&(pointer)))
#define HALF2(pointer) (reinterpret_cast<half2*>(&(pointer))[0])
using namespace nvcuda;
namespace sparse_transformers {
namespace layers {
namespace kernels {


template <class DataType>
__global__ void sparse_attention_(half *a,  half *b,  half *c, 
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
        
        int load_q_smem_addr_now = load_q_smem_addr[0];

        int load_q_gmem_addr = data_offset_q+ smem_index_i*head_size+smem_index_j;
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_q_smem_addr_now), "l"(&a[load_q_gmem_addr]));

        int to_block_id = to_select_index[0];
        int data_k_start = seq_start_index + to_block_id * block_size * head_size;
        int load_k_gmem_addr = data_k_start+ smem_index_i*head_size+smem_index_j;
        for(int i=0;i<64;i+=16){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_k_smem_addr + i*(64+pad)*2), "l"(&b[load_k_gmem_addr+i*head_size]));
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_v_smem_addr + i*(64+pad)*2), "l"(&c[load_k_gmem_addr+i*head_size]));
        }

        for(int i=0;i<4;i++){
            FLOAT4(out_temp[i*16 + smem_index_i][smem_index_j]) = FLOAT4(zero4);
            
        }
        wmma::fill_fragment(frag_out, 0.0);

        if(tidy < 2){
            global_sum_scores[tidy*32+tidx] = 0.0;
            pre_max_score[tidy*32+tidx] = 0.0f;
            max_score[tidy*32+tidx] = 0.0f;
        }

        if(tidy == 0 && tidx < 16)
        {
            temp_smem[tidx] = 0.0;
            max_score_diff[tidx] = 1;
        }

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        // 遍历K、V的每一个Block进行计算
        for(int block_id_index=to_block_start;block_id_index<to_block_end;block_id_index++){

            #pragma unroll
            for(int i=0;i<4;i++){
                wmma::load_matrix_sync(frag_k[i], &smem_k[(tidy%2)*32][i*16], 64+pad);
                wmma::load_matrix_sync(frag_v[i], &smem_v[i*16][(tidy%2)*32], 64+pad);
            }

            if(block_id_index != to_block_end - 1){
                to_block_id = to_select_index[block_id_index+1];
                data_k_start = seq_start_index + to_block_id * block_size * head_size;
                load_k_gmem_addr = data_k_start+ smem_index_i*head_size+smem_index_j;
                for(int i=0;i<4;i++){
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_k_smem_addr + i*(64+pad)*2), "l"(&b[load_k_gmem_addr+i*head_size]));
                asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_v_smem_addr + i*(64+pad)*2), "l"(&c[load_k_gmem_addr+i*head_size]));
                }
            }

            for(int from_block_part_index = 0;from_block_part_index<block_size;from_block_part_index+=16){

                //load next data
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

                wmma::store_matrix_sync(&smem_temp_score[(tidy/2)*8][(tidy%2)*32], frag_s_out, 64+pad, wmma::mem_row_major);
                wmma::load_matrix_sync(frag_out,&out_temp[(tidy/2)*8 + from_block_part_index][(tidy%2)*32],64+pad,wmma::mem_row_major);
                
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

                #pragma unroll
                for(int i=0;i<4;i++)
                {
                    wmma::load_matrix_sync(frag_s[i], &smem_temp_score[(tidy/2)*8][i*16], 64+pad);
                }

                __syncthreads();

                if(tidy == 1 && tidx < 16)
                {
                    int idx = tidx+from_block_part_index;
                    pre_max_score[idx] = max_score[idx];
                    max_score[idx] = max_score[idx] > temp_smem[tidx]?max_score[idx]:temp_smem[tidx];
                    max_score_diff[tidx] = exp(pre_max_score[idx]-max_score[idx]);
                }

                int max_value_index = tidx/4 + (tidy/2)*8;
                half2 max_value_ = __half2half2(__float2half(max(max_score[max_value_index + from_block_part_index],temp_smem[max_value_index])));
                float sum_after_sum = 0;
                half2 frag_out_value[4];
                for(int i=0;i<4;i++)
                {
                    frag_out_value[i] = HALF2(frag_out.x[i*2]);
                    HALF2(frag_s[i].x[0]) = h2exp(__hsub2(HALF2(frag_s[i].x[0]),max_value_));
                    HALF2(frag_s[i].x[2]) = h2exp(__hsub2(HALF2(frag_s[i].x[2]),max_value_));
                    if(tidy % 2 == 0){
                        sum_after_sum += __half2float(frag_s[i].x[0]);
                        sum_after_sum += __half2float(frag_s[i].x[1]);
                        sum_after_sum += __half2float(frag_s[i].x[2]);
                        sum_after_sum += __half2float(frag_s[i].x[3]);
                    }
                    if(i == 3 && (tidy % 2 == 0)){
                        sum_temp[(tidy/2)*32+tidx] = sum_after_sum;
                    }
                }

                

                #pragma unroll
                for(int i=0;i<4;i++){
                    wmma::mma_sync(frag_out, frag_s[i], frag_v[i], frag_out);
                }

                wmma::store_matrix_sync(&out_temp[(tidy/2)*8 + from_block_part_index][(tidy%2)*32],frag_out,64+pad,wmma::mem_row_major);

                // __syncthreads();

                 if(tidy == 1)
                {
                    int idx = tidx+from_block_part_index;
                    if(tidx > 15)
                        global_sum_scores[idx] *= max_score_diff[tidx - 16];
                    temp_smem[tidx] = sum_temp[tidx*2] + sum_temp[tidx*2 + 1];
                    if(tidx < 16)
                    {
                        sum_temp[tidx] = temp_smem[tidx*2] + temp_smem[tidx*2 + 1];
                        global_sum_scores[idx] += sum_temp[tidx];
                    }
                }
                // asm ("cp.async.commit_group;\n" ::);
                // asm ("cp.async.wait_group 0;\n" ::);
            }
        }

        __syncthreads();
        for(int i=0;i<16;i++){
            float sum_score_value = global_sum_scores[tidy*16+i];
            float2 out_temp_value = __half22float2(HALF2(out_temp[tidy*16+i][tidx*2]));

            out_temp[tidy*16+i][tidx*2] = __float2half(out_temp_value.x/sum_score_value);
            out_temp[tidy*16+i][tidx*2 + 1] = __float2half(out_temp_value.y/sum_score_value);
        }
        __syncthreads();
        for(int i=0;i<4;i++)
            FLOAT4(out[data_offset_q+(i*16 + smem_index_i)*head_size+smem_index_j]) = FLOAT4(out_temp[i*16 + smem_index_i][smem_index_j]);

    }
}




// void test_gemm_1(half *a, half *b,half *c, half *out,int *seq_len_info,int *from_select_index,int *from_select_index_position,int *to_select_index,int *to_select_index_position,int block_limit, int block_num, int head_num,int block_size,int head_size){
//     cudaEvent_t start,stop;
//     cudaEventCreate( &start );
//     cudaEventCreate( &stop ) ;
//     cudaEventRecord( start, 0 ) ;
//     sparse_attention_<half><<<dim3(block_limit),dim3(32,4)>>>(a,b,c,out,seq_len_info,from_select_index,from_select_index_position,to_select_index,to_select_index_position,2,64,64,11);
//     cudaEventRecord(stop,0);
//     float elapsedTime;
//     cudaEventSynchronize(stop);
//     cudaDeviceSynchronize();
//     cudaEventElapsedTime(&elapsedTime, start, stop);
//     printf( "Time to generate:  %f ms\n", elapsedTime );
// }
}
}
}

