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
// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>
// Optionally include for memcpy_async() collective
#include <cooperative_groups/memcpy_async.h>
// Optionally include for reduce() collective
#include <cooperative_groups/reduce.h>
// Optionally include for inclusive_scan() and exclusive_scan() collectives
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


__device__ void stage_1(cooperative_groups::__v1::thread_block_tile<128U, cooperative_groups::__v1::thread_block> tile_block,
barrier &ready, barrier &filled){

    if(tile_block.thread_rank() == 0)
        printf("stage 1 start %d\n",tile_block.meta_group_rank());

    for(int i=0;i<4;i++)
    {
        for(int j=0;j<5;j++){
            
            // printf("waiting\n");

            ready.arrive_and_wait();

            if(tile_block.thread_rank() == 0)
                printf("stage 1 finish\n");

            tile_block.sync();
            barrier::arrival_token token = filled.arrive();
            
        }
    }
}

__device__ void stage_2(cooperative_groups::__v1::thread_block_tile<128U, cooperative_groups::__v1::thread_block> tile_block,
barrier& ready, barrier& filled){

    if(tile_block.thread_rank() == 0)
        printf("stage 2 start %d\n",tile_block.meta_group_rank());

    

    for(int i=0;i<4;i++)
    {
        for(int j=0;j<5;j++){

            ready.arrive_and_wait();

            if(tile_block.thread_rank() == 0)
                printf("stage 2 finish\n");
 
            tile_block.sync();
            
            barrier::arrival_token token = filled.arrive();
        }
    }

}

__device__ void stage_3(cooperative_groups::__v1::thread_block_tile<128U, cooperative_groups::__v1::thread_block> tile_block,
barrier& ready, barrier& filled){
    if(tile_block.thread_rank() == 0)
        printf("stage 3 start %d\n",tile_block.meta_group_rank());

    barrier::arrival_token token = filled.arrive();
    
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<5;j++){

            ready.arrive_and_wait();

            if(tile_block.thread_rank() == 0)
                printf("stage 3 finish\n");

            tile_block.sync();
            barrier::arrival_token token = filled.arrive();
            
        }
    }

}

__global__ void test_bar(){
    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;

    __shared__  barrier bar[3];

    // printf("%d %d \n",tidx,tidy);


    //   __shared__ experimental::block_tile_memory<4, 256> shared;
    thread_block thb = this_thread_block();

    // printf("%d %d \n",tidx,tidy);


    auto tile = tiled_partition<128>(thb);

    // __shared__ experimental::block_tile_memory<4, 256> shared;
    auto block = cooperative_groups::this_thread_block();

    if(block.thread_rank() == 0)
    {
        init(bar+0, 256);
        init(bar+1, 256);
        init(bar+2, 256);

    }

    block.sync();

    auto tile_block = cg::tiled_partition<128>(block);

    if(tile_block.meta_group_rank() == 0)
        stage_1(tile_block,bar[1],bar[0]);

    if(tile_block.meta_group_rank() == 1)
        stage_2(tile_block,bar[0],bar[2]);

    if(tile_block.meta_group_rank() == 2)
        stage_3(tile_block,bar[2],bar[1]);

    // printf("%d %d \n",tidx,tidy);
}

void test_bar_(){
    test_bar<<<1,dim3(32, 12)>>>();
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    printf("exit\n");
}



__global__ void test_grp(){

    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;

    thread_block block = this_thread_block();

    thread_block_tile<32> tile32 = tiled_partition<32>(block);

    if(tile32.thread_rank() == 5){
        printf("%d %d rank = %d %d %d\n",tidx,tidy,tile32.thread_rank(),tile32.size(),tile32.meta_group_rank());
    }

}

void test_thread_group(){

    test_grp<<<1,dim3(32, 1)>>>();

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    printf("exit\n");

}

__global__ void test(half *a,  half *b,half *c){

    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_q;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_k;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_s_out;
    __shared__  half a_smem[16][16], b_smem[16][16],out_smem[16][16];

    const int index = tidx/2*16+(tidx%2)*8;

    printf("tidx:%d tidy:%d %d %d %d\n",tidx,tidy,tidx/2,(tidx%2)*8,index);
    __syncthreads();

    FLOAT4(a_smem[tidx/2][(tidx%2)*8]) = FLOAT4(a[index]);
    FLOAT4(b_smem[tidx/2][(tidx%2)*8]) = FLOAT4(b[index]);

    // if(tidx == 0 && tidy == 0){
    //     printf("matrix A\n");
    //     for(int i=0;i<16;i++)
    //     {
    //         for(int j=0;j<16;j++)
    //             printf("%f ",__half2float(a_smem[i][j]));
    //         printf("\n");
    //     }
    //     printf("matrix B\n");
    //     for(int i=0;i<16;i++)
    //     {
    //         for(int j=0;j<16;j++)
    //             printf("%f ",__half2float(b_smem[i][j]));
    //         printf("\n");
    //     }
    // }
    // __syncthreads();

    wmma::load_matrix_sync(frag_q, &a_smem[0][0], 16);
    wmma::load_matrix_sync(frag_k, &b_smem[0][0], 16);

    

    // for(int t=0; t<frag_q.num_elements; t++)
    //     printf("%d %d %d %f\n",tidx,tidy,t,__half2float(frag_q.x[t]));

    // __syncthreads();

    // for(int t=0; t<frag_k.num_elements; t++)
    //     printf("%d %d %d %f\n",tidx,tidy,t,__half2float(frag_k.x[t]));
    
    wmma::fill_fragment(frag_s_out, __float2half(0.0));

    wmma::mma_sync(frag_s_out, frag_q, frag_k, frag_s_out);

    wmma::store_matrix_sync(&out_smem[0][0], frag_s_out, 16, wmma::mem_row_major);

    // FLOAT4(c[index]) = FLOAT4(out_smem[tidx/2][(tidx%2)*8]);
    // printf("%d %d %d \n",frag_q.num_elements,frag_k.num_elements,frag_s_out.num_elements);

    
    // for(int t=0; t<frag_s_out.num_elements; t++)
    //     printf("frag_out %d %d %d %f\n",tidx,tidy,t,__half2float(frag_s_out.x[t]));

    FLOAT4(frag_q.x[0]) = FLOAT4(frag_s_out.x[0]);
    FLOAT4(frag_q.x[8]) = FLOAT4(frag_s_out.x[0]);
    for(int i=0;i<frag_q.num_elements/2;i+=4){
        HALF2(frag_q.x[i])[0] = HALF2(frag_s_out.x[i])[0];
        HALF2(frag_q.x[i+2])[0] = HALF2(frag_s_out.x[i+2])[0];
        HALF2(frag_q.x[i+8])[0] = HALF2(frag_s_out.x[i])[0];
        HALF2(frag_q.x[i+10])[0] = HALF2(frag_s_out.x[i+2])[0];

    }
    

    for(int t=0; t<frag_q.num_elements; t++)
        printf("1 %d %d %d %f\n",tidx,tidy,t,__half2float(frag_q.x[t]));


    wmma::load_matrix_sync(frag_q, &out_smem[0][0], 16);

    __syncthreads();

    for(int t=0; t<frag_q.num_elements; t++)
        printf("2 %d %d %d %f\n",tidx,tidy,t,__half2float(frag_q.x[t]));
    
}

__global__ void test_shfl(){
    float value = threadIdx.x;
    
    unsigned maxk_temp = 0x0000000f;

    float x = 0.0f;

    typedef cub::WarpReduce<float> WarpReduce;
    __shared__  typename WarpReduce::TempStorage temp_storage;

    float t = WarpReduce(temp_storage).Reduce(value, cub::Max());
    x = __shfl_sync(0, t, 0,4);
    printf("max %d %d %f\n",threadIdx.x,0xffffffff&(maxk_temp<<(0)),x);

    t = WarpReduce(temp_storage).Reduce(value, cub::Min());
    x = __shfl_sync(0xffffffff&(maxk_temp<<(4)), t, 0);
    printf("min %d %f\n",threadIdx.x,x);

    t = WarpReduce(temp_storage).Sum(value);
    x = __shfl_sync(0xffffffff&(maxk_temp<<(8)), t, 0);
    printf("sum %d %f\n",threadIdx.x,x);

}

void test_wmma(half *a, half *b,half *c)
{
    test_shfl<<<1,dim3(32, 1)>>>();
    printf("exit\n");
}



using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ void producer(barrier ready[], barrier filled[], cooperative_groups::__v1::thread_block block, int N, int buffer_len)
{
    for (int i = 0; i < (N/buffer_len); ++i) {
        ready[i%2].arrive_and_wait(); /* wait for buffer_(i%2) to be ready to be filled */
        /* produce, i.e., fill in, buffer_(i%2)  */
        if(block.thread_rank() == 0)
            printf("produce data\n");
        barrier::arrival_token token = filled[i%2].arrive(); /* buffer_(i%2) is filled */
    }
}

__device__ void consumer(barrier ready[], barrier filled[], cooperative_groups::__v1::thread_block block, int N, int buffer_len)
{
    barrier::arrival_token token1 = ready[0].arrive(); /* buffer_0 is ready for initial fill */
    barrier::arrival_token token2 = ready[1].arrive(); /* buffer_1 is ready for initial fill */
    for (int i = 0; i < (N/buffer_len); ++i) {
        filled[i%2].arrive_and_wait(); /* wait for buffer_(i%2) to be filled */
        /* consume buffer_(i%2) */
        if(block.thread_rank() == 32)
            printf("consumer data\n");
        barrier::arrival_token token = ready[i%2].arrive(); /* buffer_(i%2) is ready to be re-filled */
    }
}

//N is the total number of float elements in arrays in and out
__global__ void producer_consumer_pattern(int N, int buffer_len) {

    // Shared memory buffer declared below is of size 2 * buffer_len
    // so that we can alternatively work between two buffers.
    // buffer_0 = buffer and buffer_1 = buffer + buffer_len
    // __shared__ extern float buffer[];

    // bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
    // while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively
    __shared__ barrier bar[4];


    auto block = cooperative_groups::this_thread_block();

    printf("%d\n",block.size());

    if (block.thread_rank() < 4)
        init(bar + block.thread_rank(), block.size());
    block.sync();

    if (block.thread_rank() < warpSize)
        producer(bar, bar+2, block, N, buffer_len);
    else
        consumer(bar, bar+2, block, N, buffer_len);
}

void test_p_c(){
    producer_consumer_pattern<<<1,dim3(256, 1)>>>(10,2);
    printf("exit\n");
}

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

}
}
}