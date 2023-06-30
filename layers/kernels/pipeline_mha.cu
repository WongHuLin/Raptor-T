#include "kernel.h"
#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <stdio.h>
#include <numeric>
#include <thrust/extrema.h>
#include <mma.h>
// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>
// Optionally include for memcpy_async() collective
#include <cooperative_groups/memcpy_async.h>
// Optionally include for reduce() collective
#include <cuda/barrier>


using namespace cooperative_groups;
// Alternatively use an alias to avoid polluting the namespace with collective algorithms
namespace cg = cooperative_groups;


#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FLOAT(pointer) (reinterpret_cast<float*>(&(pointer)))
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer)))
#define HALF(pointer) (reinterpret_cast<half*>(&(pointer)))
#define HALF2(pointer) (reinterpret_cast<half2*>(&(pointer)))

using namespace nvcuda;

namespace sparse_transformers {
namespace layers {
namespace kernels {

using barrier = cuda::barrier<cuda::thread_scope_block>;

// __device__ stage_1(barrier ready[],barrier filled[]){

// }

#define _CG_ABI_EXPERIMENTAL

__device__ void stage1(cooperative_groups::__v1::thread_block_tile<128U, cooperative_groups::__v1::thread_block>& tile_block,
barrier &ready, barrier &filled, half *a,  half *b, half smem_temp_score[][64+8], const int& compute_block_start, const int& compute_block_end,
const int *from_block_index, const int& batch_size, const int *seq_len_info, const int& block_size,
const int& head_size, const int *to_select_index_position, const int *to_select_index,float *temp_smem, float* pre_max_score, float* max_score, float* max_score_diff){
    const int tidy = tile_block.thread_rank()/32;
    const int tidx = threadIdx.x;
    const int pad = 8;
    barrier::arrival_token token;

    typedef cub::WarpReduce<float> WarpReduce;
    __shared__  typename WarpReduce::TempStorage temp_storage[4];

    __shared__  half smem_q[2][16][64+pad],smem_k[64][64+pad];


    const int smem_index_i =  tidy*4+tidx/8;
    const int smem_index_j = (tidx%8)*8;

    const int load_k_smem_addr = __cvta_generic_to_shared(smem_k[0]) + (smem_index_i*(64+pad)+smem_index_j)*2;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_q[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_k[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_s_out;

    const unsigned long load_q_smem_addr[2] = {__cvta_generic_to_shared(smem_q[0]) + (smem_index_i*(64+pad)+smem_index_j)*2,__cvta_generic_to_shared(smem_q[1]) + (smem_index_i*(64+pad)+smem_index_j)*2};
    
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

        const int head_num = (from_block_id - 12*seq_start_block_index)/seq_block_len;
        from_block_id = (from_block_id - 12*seq_start_block_index)%seq_block_len;

        const int seq_start_index = 12*seq_start_block_index*block_size*head_size + head_num*seq_block_len*block_size*head_size;
        const int data_offset_q = seq_start_index + from_block_id*block_size*head_size;

        const int to_block_start = to_select_index_position[from_block_id+seq_start_block_index];
        const int to_block_end = to_select_index_position[from_block_id+seq_start_block_index + 1];

        int load_q_smem_addr_now = load_q_smem_addr[0];
       
        int load_q_gmem_addr = data_offset_q+ smem_index_i*head_size+smem_index_j;
        asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_q_smem_addr_now), "l"(&a[load_q_gmem_addr]));

        int to_block_id = to_select_index[to_block_start];
        int data_k_start = seq_start_index + to_block_id * block_size * head_size;
        int load_k_gmem_addr = data_k_start+ smem_index_i*head_size+smem_index_j;
        for(int i=0;i<64;i+=16){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_k_smem_addr + i*(64+pad)*2), "l"(&b[load_k_gmem_addr+i*head_size]));
        }

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        // if(tile_block.thread_rank() == 0)
        // {
        //     printf("%d %d %d",to_block_id,data_k_start,load_k_gmem_addr);
        //     for(int i=0;i<64;i++){
        //         for(int j=0;j<6;j++)
        //             printf("%f ", __half2float(smem_k[i][j]));
        //         printf("\n");
        //     }
        //     // printf("%d\n",tidy*128*4+from_block_part_index*128+(tidx/8)*128+(tidx%8)*8+64*block_id_index);
        // }
        // tile_block.sync();

        for(int block_id_index=to_block_start;block_id_index<to_block_end;block_id_index++){
            #pragma unroll
            for(int i=0;i<4;i++){
                wmma::load_matrix_sync(frag_k[i], &smem_k[tidy*16][i*16], 64+pad);
            }

            tile_block.sync();

            if(block_id_index != to_block_end - 1){
                to_block_id = to_select_index[block_id_index+1];

                // if(tile_block.thread_rank() == 0)
                // {
                //     printf("%d %d %d %d %d\n",to_block_id,block_id_index,to_block_end - 1,to_select_index[0],to_select_index[1]);
                // }

                data_k_start = seq_start_index + to_block_id * block_size * head_size;
                
                int load_k_gmem_addr = data_k_start+ smem_index_i*head_size+smem_index_j;
                for(int i=0;i<64;i+=16){
                    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_k_smem_addr + i*(64+pad)*2), "l"(&b[load_k_gmem_addr+i*head_size]));
                }
            }

            for(int from_block_part_index = 0;from_block_part_index<block_size;from_block_part_index+=16){
                
                #pragma unroll
                for(int i=0;i<4;i++)
                {
                    wmma::load_matrix_sync(frag_q[i], &smem_q[(from_block_part_index/16)&1][0][i*16], 64+pad);
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

                // get single


                wmma::store_matrix_sync(&smem_temp_score[0][tidy*16], frag_s_out, 64+pad, wmma::mem_row_major);

                // 同步 
                // send single

                // FLOAT4(test_out[tidy*128*4+from_block_part_index*128+(tidx/8)*128+(tidx%8)*8+64*block_id_index]) = FLOAT4(smem_temp_score[tidy*4+tidx/8][(tidx%8)*8]);
                
                half2 value_h2[4];
                float max_temp[4];
                half2 diff_x[4];
                half2 out_t[4];
                float max_value[4];

                {
                    float score_value[4];
                    #pragma unroll
                    for(int i=0;i<4;i++)
                    {
                        max_value[i] = max_score[tidy*4+i+from_block_part_index];
                        value_h2[i] = HALF2(smem_temp_score[tidy*4+i][0])[tidx];
                    }
                    // tile_block.sync();

                    #pragma unroll
                    for(int i=0;i<4;i++)
                    {
                        score_value[i] = __half2float(__hmax(value_h2[i].x,value_h2[i].y));
                    }

                    #pragma unroll
                    for(int i=0;i<4;i++)
                    {
                        max_temp[i] = WarpReduce(temp_storage[tidy]).Reduce(score_value[i], cub::Max());
                    }
    
                    #pragma unroll
                    for(int i=0;i<4;i++){
                        max_temp[i] = __shfl_sync(0xffffffff, max_temp[i], 0); 
                        max_temp[i] = max(max_temp[i],max_value[i]);
                        pre_max_score[tidy*4+i+from_block_part_index] = max_value[i];
                        max_score[tidy*4+i+from_block_part_index] = max_temp[i];
                    }

                    tile_block.sync();
                    ready.arrive_and_wait();

                    for(int i=0;i<4;i++){                        
                        max_score_diff[tidy*4+i]  = exp(max_temp[i]-max_value[i]);

                        half2 t = h2exp(__hsub2(value_h2[i],__half2half2(__float2half(max_temp[i]))));
                        HALF2(smem_temp_score[tidy*4+i][0])[tidx] = t;
                        float v_ = __half2float(__hadd(t.x,t.y));
                        temp_smem[tidy*4 + i] = WarpReduce(temp_storage[tidy]).Sum(v_);
                    }

                    // token = filled.arrive();
                }

                asm ("cp.async.commit_group;\n" ::);
                asm ("cp.async.wait_group 0;\n" ::);  
            }
        }
    }
}
__device__ void stage_2(cooperative_groups::__v1::thread_block_tile<128U, cooperative_groups::__v1::thread_block>& tile_block,
barrier& ready, barrier& filled,barrier& ready_, barrier& filled_, const int& compute_block_start, const int& compute_block_end, const int *to_select_index_position,
const int *from_block_index, const int& batch_size, const int *seq_len_info, const int& block_size, half smem_temp_score[][64+8], 
float *temp_smem, float* pre_max_score, float* max_score, float* max_score_diff, float* global_sum_scores,half out_temp[][64+8]){

    const int tidy = tile_block.thread_rank()/32;
    const int tidx = threadIdx.x;
    barrier::arrival_token token;

    typedef cub::WarpReduce<float> WarpReduce;
    __shared__  typename WarpReduce::TempStorage temp_storage[4];

    filled.arrive();




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

        if(tidy < 2){
            pre_max_score[tidy*32+tidx] = 0.0f;
            max_score[tidy*32+tidx] = 0.0f;
        }

        float max_value[4] = {0.0f,0.0f,0.0f,0.0f};

        //还原原始的headnum和blockid
        const int head_num = (from_block_id - 12*seq_start_block_index)/seq_block_len;
        from_block_id = (from_block_id - 12*seq_start_block_index)%seq_block_len;

        const int to_block_start = to_select_index_position[from_block_id+seq_start_block_index];
        const int to_block_end = to_select_index_position[from_block_id+seq_start_block_index + 1];

        for(int block_id_index=to_block_start;block_id_index<to_block_end;block_id_index++){
            
            for(int from_block_part_index = 0;from_block_part_index<block_size;from_block_part_index+=16){
                
                // 计算最大值 rowmax

                ready.arrive_and_wait();

                // float value_h2[4];
                // float score_value[2];

                // for(int i=0;i<8;i++){
                //     value_h2[0] = __half2float(smem_temp_score[i*2][tidx]);
                //     value_h2[1] = __half2float(smem_temp_score[i*2][tidx+32]);
                //     value_h2[2] = __half2float(smem_temp_score[i*2+1][tidx]);
                //     value_h2[3] = __half2float(smem_temp_score[i*2+1][tidx+32]);
                //     score_value[0] = max(value_h2[0],value_h2[1]);
                //     score_value[1] = max(value_h2[3],value_h2[4]);
                //     temp_smem[i*2] = WarpReduce(temp_storage[0]).Reduce(score_value[0], cub::Max());
                //     temp_smem[i*2+1] = WarpReduce(temp_storage[1]).Reduce(score_value[1], cub::Max());
                // }

                // if(tile_block.thread_rank() < 16){
                //     int idx = tidx+from_block_part_index;
                //     pre_max_score[idx] = max_score[idx];
                //     max_score[idx] = max(max_score[idx],temp_smem[tidx]);
                //     max_score_diff[tidx] = exp(pre_max_score[idx]-max_score[idx]);
                // }
                half2 value_h2[4];
                float max_temp[4];
                half2 diff_x[4];
                half2 out_t[4];
                {
                    float score_value[4];
                    #pragma unroll
                    for(int i=0;i<4;i++)
                    {
                        max_value[i] = max_score[tidy*4+i+from_block_part_index];
                        value_h2[i] = HALF2(smem_temp_score[tidy*4+i][0])[tidx];
                    }
                    // tile_block.sync();
                    filled.arrive();

                    #pragma unroll
                    for(int i=0;i<4;i++)
                    {
                        score_value[i] = __half2float(__hmax(value_h2[i].x,value_h2[i].y));
                    }

                    #pragma unroll
                    for(int i=0;i<4;i++)
                    {
                        max_temp[i] = WarpReduce(temp_storage[tidy]).Reduce(score_value[i], cub::Max());
                    }
    
                    #pragma unroll
                    for(int i=0;i<4;i++){
                        max_temp[i] = __shfl_sync(0xffffffff, max_temp[i], 0); 
                        out_t[i] = HALF2(out_temp[tidy*4+i+from_block_part_index][0])[tidx];
                        max_temp[i] = max(max_temp[i],max_value[i]);
                        diff_x[i] = __half2half2(__float2half(exp(max_temp[i]-max_value[i])));
                        max_score[tidy*4+i+from_block_part_index] = max_temp[i];
                        pre_max_score[tidy*4+i+from_block_part_index] = max_value[i];
                    }

                    ready_.arrive_and_wait();

                    for(int i=0;i<4;i++){
                        half2 t = h2exp(__hsub2(value_h2[i],__half2half2(__float2half(max_temp[i]))));
                        HALF2(out_temp[tidy*4+i+from_block_part_index][0])[tidx] = __hmul2(out_t[i],diff_x[i]);
                        HALF2(smem_temp_score[tidy*4+i][0])[tidx] = t;
                        float v_ = __half2float(__hadd(t.x,t.y));
                        temp_smem[tidy*4 + i] = WarpReduce(temp_storage[tidy]).Sum(v_);
                    }
                }

                // tile_block.sync();

                // if(tidy == 1 && tidx < 16)
                // {
                //     int idx = tidx+from_block_part_index;
                //     pre_max_score[idx] = max_score[idx];
                //     max_score[idx] = max(max_score[idx],temp_smem[tidx]);
                //     max_score_diff[tidx] = exp(pre_max_score[idx]-max_score[idx]);
                // }
                // // tile_block.sync();

                // half2 max_s[4];
                // for(int i=0;i<4;i++){
                //     max_s[i] = __half2half2(__float2half(max_score[tidy*4 + i + from_block_part_index]));
                // }
                // half2 diff_x = __half2half2(__float2half(max_score_diff[tidy*4 + tidx/8 ]));
                // float4 out_t = FLOAT4(out_temp[from_block_part_index + tidy*4+(tidx*8)/64][(tidx*8)%64]);


                // for(int i=0;i<4;i++){
                //     half2 t = h2exp(__hsub2(value_h2[i],max_s[i]));
                //     HALF2(out_temp[from_block_part_index + tidy*4+(tidx*8)/64][(tidx*8)%64])[i] = __hmul2(HALF2(out_t)[i],diff_x);

                //     HALF2(smem_temp_score[tidy*4+i][0])[tidx] = t;
                //     float v_ = __half2float(__hadd(HALF(t)[0],HALF(t)[1]));
                //     temp_smem[tidy*4 + i] = WarpReduce(temp_storage[tidy]).Sum(v_);
                // }

                token = filled_.arrive();

                if(tidy == 0 && tidx < 16)
                {
                    int idx = tidx+from_block_part_index;
                    global_sum_scores[idx] *= max_score_diff[tidx];
                    global_sum_scores[idx] += temp_smem[tidx];
                }

                tile_block.sync();

                // tile_block.sync();


            }
            
        }
    }
}

__device__ void stage_3(cooperative_groups::__v1::thread_block_tile<128U, cooperative_groups::__v1::thread_block>& tile_block,
barrier& ready, barrier& filled, const int& compute_block_start, const int& compute_block_end, half* c, half* out,const int *to_select_index_position,
const int *from_block_index, const int& batch_size, const int& head_size, const int *seq_len_info, const int& block_size, half smem_temp_score[][64+8], 
float *temp_smem, float* pre_max_score, float* max_score, float* max_score_diff, const int *to_select_index, float* global_sum_scores, half out_temp[][72]){

    const int tidy = tile_block.thread_rank()/32;
    const int tidx = threadIdx.x;
    const int pad = 8;
    barrier::arrival_token token;

    __shared__  half smem_v[64][64+pad];



    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_s[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_v[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_out;

    const int smem_index_i =  tidy*4+tidx/8;
    const int smem_index_j = (tidx%8)*8;

    filled.arrive();

    const int load_v_smem_addr = __cvta_generic_to_shared(smem_v[0]) + (smem_index_i*(64+pad)+smem_index_j)*2;


    float4 zero4 = {0.0f,0.0f,0.0f,0.0f};

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

        tile_block.sync();

        for(int i=0;i<4;i++){
            FLOAT4(out_temp[i*16 + smem_index_i][smem_index_j]) = FLOAT4(zero4);
        }

        if(tidy < 2){
            global_sum_scores[tidy*32+tidx] = 0.0;
        }

        int to_block_id = to_select_index[to_block_start];
        int data_k_start = seq_start_index + to_block_id * block_size * head_size;
        int load_k_gmem_addr = data_k_start+ smem_index_i*head_size+smem_index_j;
        for(int i=0;i<64;i+=16){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_v_smem_addr + i*(64+pad)*2), "l"(&c[load_k_gmem_addr+i*head_size]));
        }

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        for(int block_id_index=to_block_start;block_id_index<to_block_end;block_id_index++){

            #pragma unroll
            for(int i=0;i<4;i++){
                wmma::load_matrix_sync(frag_v[i], &smem_v[i*16][tidy*16], 64+pad);
            }

            if(block_id_index != to_block_end - 1){
                to_block_id = to_select_index[block_id_index+1];
                data_k_start = seq_start_index + to_block_id * block_size * head_size;
                
                int load_k_gmem_addr = data_k_start+ smem_index_i*head_size+smem_index_j;
                for(int i=0;i<64;i+=16){
                    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_v_smem_addr + i*(64+pad)*2), "l"(&c[load_k_gmem_addr+i*head_size]));
                }
            }

            for(int from_block_part_index = 0;from_block_part_index<block_size;from_block_part_index+=16){
                
                // ready.arrive_and_wait();

                for(int i=0;i<4;i++)
                {
                    wmma::load_matrix_sync(frag_s[i], &smem_temp_score[0][i*16], 64+pad);
                }


                half2 diff_x[4];
                half2 out_t[4];

                for(int i=0;i<4;i++){
                    out_t[i] = HALF2(out_temp[tidy*4+i+from_block_part_index][0])[tidx];
                    diff_x[i] = __half2half2(__float2half(max_score_diff[tidy*4+i])); 
                }

                if(tidy == 0 && tidx < 16)
                {
                    int idx = tidx+from_block_part_index;
                    global_sum_scores[idx] *= max_score_diff[tidx];
                    global_sum_scores[idx] += temp_smem[tidx];
                }

                // token = filled.arrive();


                for(int i=0;i<4;i++){
                    HALF2(out_temp[tidy*4+i+from_block_part_index][0])[tidx] = __hmul2(out_t[i],diff_x[i]);
                }

                tile_block.sync();

            //     half2 max_s[4];
            //     half2 value[4];
            //     for(int i=0;i<4;i++){
            //         value[i] =  HALF2(smem_temp_score[tidy*4+i][0])[tidx];
            //         max_s[i] = __half2half2(__float2half(max_score[tidy*4 + i + from_block_part_index]));
            //     }
            //     half2 diff_x = __half2half2(__float2half(max_score_diff[tidy*4 + tidx/8 ]));
            //     float4 out_t = FLOAT4(out_temp[from_block_part_index + tidy*4+(tidx*8)/64][(tidx*8)%64]);
                
            //     for(int i=0;i<4;i++){
            //         half2 t = h2exp(__hsub2(value[i],max_s[i]));
            //         HALF2(out_temp[from_block_part_index + tidy*4+(tidx*8)/64][(tidx*8)%64])[i] = __hmul2(HALF2(out_t)[i],diff_x);

            //         HALF2(smem_temp_score[tidy*4+i][0])[tidx] = t;
            //         float v_ = __half2float(__hadd(HALF(t)[0],HALF(t)[1]));
            //         temp_smem[tidy*4 + i] = WarpReduce(temp_storage[tidy]).Sum(v_);
            //     }

            //     // if(tile_block.thread_rank() == 0 && from_block_part_index == 0){
            //     //     for(int i=0;i<64;i++){
            //     //         printf("%f ",__half2float(smem_temp_score[0][i]));
            //     //     }

            //     //     printf("\n");

            //     //     printf("%f %f %f %f\n",max_score[0],max_score_diff[0],global_sum_scores[0],__half2float(out_temp[0][0]));
                    
            //     // }

            //     // tile_block.sync();
                
            //     // for(int i=0;i<8;i++){
            //     //     half value = smem_temp_score[(tile_block.thread_rank()*8)/64][(tile_block.thread_rank()*8)%64+i];
            //     //     smem_temp_score[(tile_block.thread_rank()*8)/64][(tile_block.thread_rank()*8)%64+i] =  hexp(__hsub(value, max_score[(tile_block.thread_rank()*8)/64 + from_block_part_index]));
            //     //     value = out_temp[(tile_block.thread_rank()*8)/64 + from_block_part_index][(tile_block.thread_rank()*8)%64+i];
            //     //     out_temp[(tile_block.thread_rank()*8)/64 + from_block_part_index][(tile_block.thread_rank()*8)%64+i] = __hmul(value,max_score_diff[(tile_block.thread_rank()*8)/64]);
            //     // }

            //     // tile_block.sync();

            //     // if(tile_block.thread_rank()<16){
            //     //     float sum_temp = 0.0f;
            //     //     for(int i=0;i<64;i++){
            //     //         sum_temp += __half2float(smem_temp_score[tidx][i]);
            //     //     }
            //     //     temp_smem[tidx] = sum_temp;
            //     //     // printf("%f\n",sum_temp);
            //     // }

            //    // 同步
            //     tile_block.sync();

                // if(tidy == 0 && tidx < 16)
                // {
                //     int idx = tidx+from_block_part_index;
                //     global_sum_scores[idx] *= max_score_diff[tidx];
                //     global_sum_scores[idx] += temp_smem[tidx];
                // }

                // if(tile_block.thread_rank() == 0 && from_block_part_index == 0){
                //     for(int i=0;i<64;i++){
                //         printf("%f ",__half2float(smem_temp_score[0][i]));
                //     }

                //     printf("\n");

                //     printf("%f %f %f\n",temp_smem[0],global_sum_scores[0],__half2float(out_temp[0][0]));
                    
                // }
                // tile_block.sync();

                wmma::load_matrix_sync(frag_out,&out_temp[from_block_part_index][tidy*16],64+pad,wmma::mem_row_major);

                for(int i=0;i<4;i++){
                    wmma::mma_sync(frag_out, frag_s[i], frag_v[i], frag_out);
                }

                wmma::store_matrix_sync(&out_temp[from_block_part_index][tidy*16],frag_out,64+pad,wmma::mem_row_major);

            }

            asm ("cp.async.commit_group;\n" ::);
            asm ("cp.async.wait_group 0;\n" ::);
        }

        tile_block.sync();

        for(int i=0;i<16;i++){
            float sum_score_value = global_sum_scores[tidy*16+i];
            float2 out_temp_value = __half22float2(HALF2(out_temp[tidy*16+i][0])[tidx]);

            out_temp[tidy*16+i][tidx*2] = __float2half(out_temp_value.x/sum_score_value);
            out_temp[tidy*16+i][tidx*2 + 1] = __float2half(out_temp_value.y/sum_score_value);

        }

        tile_block.sync();

        for(int i=0;i<4;i++)
            FLOAT4(out[data_offset_q+(i*16 + smem_index_i)*head_size+smem_index_j]) = FLOAT4(out_temp[i*16 + smem_index_i][smem_index_j]);


    }

}

__global__ void sparse_attention_test(half *a,  half *b,  half *c, 
    half *out, const int *seq_len_info,const int *from_block_index, 
    const int *from_block_index_position, const int *to_select_index,
    const int *to_select_index_position, const int batch_size,
    const int block_size, const int head_size){

    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    __shared__ barrier bar[4];

    thread_block thb = this_thread_block();
    auto tile_block = cg::tiled_partition<128>(thb);


    if(thb.thread_rank() == 0)
    {
        init(bar+0, 256);
        init(bar+1, 256);
        init(bar+2, 256);
        init(bar+3, 256);
    }


    __shared__ __align__(32) float global_sum_scores[64],pre_max_score[64],max_score[64];  
    __shared__ __align__(32) half smem_temp_score[16][64+8],out_temp[64][64+8];
    __shared__  float temp_smem[32],max_score_diff[16];



    const int compute_block_start = from_block_index_position[bidx];
    const int compute_block_end = from_block_index_position[bidx + 1];
    const int compute_block_num = compute_block_end - compute_block_start;

    thb.sync();

    if(tile_block.meta_group_rank() == 0)
        stage1(tile_block,bar[1],bar[0],a,b,smem_temp_score,compute_block_start,
        compute_block_end,from_block_index,batch_size,seq_len_info,block_size,
        head_size,to_select_index_position,to_select_index,temp_smem,
        pre_max_score,max_score,max_score_diff);
    
    // if(tile_block.meta_group_rank() == 2)
    //     stage_2(tile_block,bar[0],bar[1],bar[2],bar[3],compute_block_start,compute_block_end,to_select_index_position,
    //     from_block_index,batch_size,seq_len_info,block_size,smem_temp_score,temp_smem,
    //     pre_max_score,max_score,max_score_diff,global_sum_scores,out_temp);


    if(tile_block.meta_group_rank() == 1)
        stage_3(tile_block,bar[1],bar[0],compute_block_start,compute_block_end,c,out,
        to_select_index_position,from_block_index,batch_size,head_size,seq_len_info,block_size,
        smem_temp_score,temp_smem,pre_max_score,max_score,max_score_diff,to_select_index,
        global_sum_scores,out_temp);

    // thb.sync();
    // if(thb.thread_rank() == 0){
    //     for(int i=0;i<64;i++)
    //     {
    //         for(int j=0;j<64;j++){
    //             printf("%f ",__half2float(out[i*64+j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // thb.sync();

}

void sparse_attention_test_larunch(half *a, half *b,half *c, half *out,int batch_size,int *seq_len_info,int *from_select_index,int *from_select_index_position,int *to_select_index,int *to_select_index_position, int block_size,int head_size)
{
    sparse_attention_test<<<dim3(160),dim3(32,8)>>>(a,b,c,out,seq_len_info,from_select_index,from_select_index_position,to_select_index,to_select_index_position,batch_size,block_size,head_size);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    printf("ending\n");
}

// void test_gemm_1(half *a, half *b,half *c, half *out,int batch_size,int *seq_len_info,int *from_select_index,int *from_select_index_position,int *to_select_index,int *to_select_index_position,int block_limit, int block_num, int head_num,int block_size,int head_size,std::map<std::string,float>& info,bool balanced)
// {
//     auto start_time = std::chrono::system_clock::now();
    
    
//     if(balanced)
//         sparse_attention_test<<<dim3(block_limit),dim3(32,12)>>>(a,b,c,out,seq_len_info,from_select_index,from_select_index_position,to_select_index,to_select_index_position,batch_size,block_size,head_size);
//     // else{
//     //     sparse_attention_non_balanced<half><<<dim3(head_num,block_num),dim3(32,4)>>>(a,b,c,out,seq_len_info,from_select_index,from_select_index_position,to_select_index,to_select_index_position,batch_size,block_size,head_size);
//     // }

//     auto end_time = std::chrono::system_clock::now();
//     if(info.find("attention") != info.end())
//     {    auto dura = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count();
//         info["attention"] += dura;
//     }
// }






}
}
}