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

using namespace nvcuda;

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FLOAT(pointer) (reinterpret_cast<float*>(&(pointer)))
#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer)))
#define HALF(pointer) (reinterpret_cast<half*>(&(pointer)))
#define HALF2(pointer) (reinterpret_cast<half2*>(&(pointer)))

namespace sparse_transformers {
namespace layers {
namespace kernels {


__inline__ __device__ void warpReduceSum(half values[][72],const int tidx, const int tidy, float *store_v){
    float4 value[2] = {FLOAT4(values[tidx][tidy*16]),FLOAT4(values[tidx][tidy*16 + 8])};

    float v[2];
    float result[16];
    float temp[2];
    for(int i=0;i<8;i++){
        v[0] = __half2float(HALF(value)[i*2]);
        v[1] = __half2float(HALF(value)[i*2 + 1]);
        for(int j=16; j>=1; j/=2){
            temp[0] += __shfl_xor_sync(0xffffffff, v[0], j, 32);
            temp[1] += __shfl_xor_sync(0xffffffff, v[1], j, 32);
            // v[0] += temp[0];
            // v[1] += temp[1];

        }
        result[i*2] = __shfl_sync(0xffffffff, v[0], 0);
        result[i*2 + 1] = __shfl_sync(0xffffffff, v[1], 0);
    }
    store_v[0] = result[tidx/4];
    store_v[1] = result[tidx/4 + 8];

}

__inline__ __device__ void warpReduceMax(half values[][72],const int tidx, const int tidy, float *store_v){
    float4 value[2] = {FLOAT4(values[tidx][tidy*16]),FLOAT4(values[tidx][tidy*16 + 8])};
    float v[2];
    float result[16];
    for(int i=0;i<8;i++){
        v[0] = __half2float(HALF(value)[i*2]);
        v[1] = __half2float(HALF(value)[i*2 + 1]);
        for(int j=16; j>=1; j/=2){
            v[0] = max(v[0], __shfl_xor_sync(0xffffffff, v[0], j, 32));
            v[1] = max(v[1], __shfl_xor_sync(0xffffffff, v[1], j, 32));
        }
        result[i*2] = __shfl_sync(0xffffffff, v[0], 0);
        result[i*2 + 1] = __shfl_sync(0xffffffff, v[1], 0);
    }

    store_v[0] = result[tidx/4];
    store_v[1] = result[tidx/4 + 8];
}

__global__ void sparse_attention_lastest(half *a,  half *b,  half *c, 
    half *out,const int *seq_len_info,const int *from_block_index, 
    const int *from_block_index_position, const int *to_select_index,
    const int *to_select_index_position, const int batch_size,
    const int block_size,const int head_size){

    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int pad = 8;
    
    // typedef cub::WarpReduce<float> WarpReduce;
    // __shared__ __align__(32)  typename WarpReduce::TempStorage temp_storage[4];
    __shared__ __align__(32)  float global_sum_scores[64],pre_max_score[64],max_score[64];  
    __shared__ __align__(32)  float temp_smem[64],max_score_diff[16],sum_temp[64];
    __shared__ __align__(32)  half smem_q[64][64+pad],smem_k[2][32][64+pad],smem_v[2][32][64+pad],smem_temp_score[16][64+pad],out_temp[64][64+pad];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_q[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_k[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_s_out[2];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_s[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_v[4][2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_out[4];

    const int smem_index_i =  tidy*4+tidx/8;
    const int smem_index_j = (tidx%8)*8;

    const int load_k_smem_addr[2] = {__cvta_generic_to_shared(smem_k[0]) + (smem_index_i*(64+pad)+smem_index_j)*2,__cvta_generic_to_shared(smem_k[1]) + (smem_index_i*(64+pad)+smem_index_j)*2};
    const int load_v_smem_addr[2] = {__cvta_generic_to_shared(smem_v[0]) + (smem_index_i*(64+pad)+smem_index_j)*2,__cvta_generic_to_shared(smem_v[1]) + (smem_index_i*(64+pad)+smem_index_j)*2};
    const int load_q_smem_addr = __cvta_generic_to_shared(smem_q[0]) + (smem_index_i*(64+pad)+smem_index_j)*2;
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
        const int to_block_num = to_block_end - to_block_start;

        int to_block_id = to_select_index[0];
        int data_k_start = seq_start_index + to_block_id * block_size * head_size;
        int load_k_gmem_addr = data_k_start+ smem_index_i*head_size+smem_index_j;
        int load_q_gmem_addr = data_offset_q+ smem_index_i*head_size+smem_index_j;

        for(int i=0;i<64;i+=16){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_q_smem_addr + i*(64+pad)*2), "l"(&a[load_q_gmem_addr+i*head_size]));
        }

        for(int i=0;i<32;i+=16){
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_k_smem_addr[0] + i*(64+pad)*2), "l"(&b[load_k_gmem_addr+i*head_size]));
            asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_v_smem_addr[0] + i*(64+pad)*2), "l"(&c[load_k_gmem_addr+i*head_size]));
        }
        
        for(int i=0;i<4;i++)
            wmma::fill_fragment(frag_out[i], __float2half(0.0));

        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);

        for(int i=0;i<4;i++){
            wmma::load_matrix_sync(frag_q[i], &smem_q[tidy*16][i*16], 64+pad);
        }

        float pre_max_score[2] = {FLT_MIN,FLT_MIN};
        float max_score[2] = {FLT_MIN,FLT_MIN};
        half2 max_diff[2];
        float global_sum[2] = {0.0f, 0.0f};


        for(int block_id_index=0;block_id_index<to_block_num*2;block_id_index++){
            #pragma unroll
            for(int i=0;i<2;i++)
            {
                #pragma
                for(int j=0;j<4;j++){
                    wmma::load_matrix_sync(frag_k[i][j], &smem_k[block_id_index & 1][i*16][j*16], 64+pad);
                    wmma::load_matrix_sync(frag_v[j][i], &smem_v[block_id_index & 1][i*16][j*16], 64+pad);
                }

                 wmma::fill_fragment(frag_s_out[i], __float2half(0.0));
            }

            if(block_id_index != to_block_num*2 - 1){
                to_block_id = to_select_index[(block_id_index+1)/2];
                data_k_start = seq_start_index + to_block_id * block_size * head_size ;
                
                int load_k_gmem_addr = data_k_start+ smem_index_i*head_size+smem_index_j;
                if((block_id_index+1) & 1){
                    load_k_gmem_addr += 32*head_size;
                }

                for(int i=0;i<32;i+=16){
                    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_k_smem_addr[(block_id_index+1) & 1] + i*(64+pad)*2), "l"(&b[load_k_gmem_addr+i*head_size]));
                    asm ("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(load_v_smem_addr[(block_id_index+1) & 1] + i*(64+pad)*2), "l"(&c[load_k_gmem_addr+i*head_size]));
                }
            }

            #pragma unroll
            for(int i=0;i<2;i++)
            {
                #pragma
                for(int j=0;j<4;j++){
                    wmma::mma_sync(frag_s_out[i], frag_q[j], frag_k[i][j], frag_s_out[i]);
                }
                 wmma::store_matrix_sync(&smem_q[i*16][tidy*16], frag_s_out[i], 64+pad, wmma::mem_col_major);
            }

            float value[4];

            float max_t[2];
            unsigned maxk_temp = 0x0000000f;

            // for(int i=0;i<4;i++){
            //     value[0] = __half2float(smem_q[tidy*16+i*4+0][tidx]);
            //     value[1] = __half2float(smem_q[tidy*16+i*4+1][tidx]);
            //     value[2] = __half2float(smem_q[tidy*16+i*4+2][tidx]);
            //     value[3] = __half2float(smem_q[tidy*16+i*4+3][tidx]);
            //     temp_smem[tidy*16+i*4+0] = WarpReduce(temp_storage[tidy]).Reduce(value[0], cub::Max());
            //     temp_smem[tidy*16+i*4+1] = WarpReduce(temp_storage[tidy]).Reduce(value[1], cub::Max());
            //     temp_smem[tidy*16+i*4+2] = WarpReduce(temp_storage[tidy]).Reduce(value[2], cub::Max());
            //     temp_smem[tidy*16+i*4+3] = WarpReduce(temp_storage[tidy]).Reduce(value[3], cub::Max());
            // }
            warpReduceMax(smem_q,tidx,tidy,max_t);

            // max_t[0] = temp_smem[tidy*16+tidx/4];
            // max_t[1] = temp_smem[tidy*16+tidx/4 + 8];

            FLOAT2(pre_max_score[0])[0] = FLOAT2(max_score[0])[0];
            max_score[0] = max(max_score[0],max_t[0]);
            max_score[1] = max(max_score[1],max_t[1]);



            max_diff[0] = __half2half2(__float2half(exp(pre_max_score[0]-max_score[0])));
            max_diff[1] = __half2half2(__float2half(exp(pre_max_score[1]-max_score[1])));

            for(int i=0;i<frag_s_out[0].num_elements;i+=4){
                HALF2(frag_s_out[0].x[i])[0]  = h2exp(__hsub2(HALF2(frag_s_out[0].x[i])[0],__half2half2(__float2half(max_score[0]))));
                HALF2(frag_s_out[0].x[i+2])[0]  = h2exp(__hsub2(HALF2(frag_s_out[0].x[i+2])[0],__half2half2(__float2half(max_score[1]))));
                HALF2(frag_s_out[1].x[i])[0]  = h2exp(__hsub2(HALF2(frag_s_out[1].x[i])[0],__half2half2(__float2half(max_score[0]))));
                HALF2(frag_s_out[1].x[i+2])[0]  = h2exp(__hsub2(HALF2(frag_s_out[1].x[i+2])[0],__half2half2(__float2half(max_score[1]))));   
            }

            for(int i=0;i<2;i++){
                 wmma::store_matrix_sync(&smem_q[i*16][tidy*16], frag_s_out[i], 64+pad, wmma::mem_col_major);
            }


            // float tt;

            // for(int i=0;i<4;i++){
            //     value[0] = __half2float(smem_q[tidy*16+i*4+0][tidx]);
            //     value[1] = __half2float(smem_q[tidy*16+i*4+1][tidx]);
            //     value[2] = __half2float(smem_q[tidy*16+i*4+2][tidx]);
            //     value[3] = __half2float(smem_q[tidy*16+i*4+3][tidx]);
            //     temp_smem[tidy*16+i*4+0] = WarpReduce(temp_storage[tidy]).Reduce(value[0], cub::Sum());
            //     temp_smem[tidy*16+i*4+1] = WarpReduce(temp_storage[tidy]).Reduce(value[1], cub::Sum());
            //     temp_smem[tidy*16+i*4+2] = WarpReduce(temp_storage[tidy]).Reduce(value[2], cub::Sum());
            //     temp_smem[tidy*16+i*4+3] = WarpReduce(temp_storage[tidy]).Reduce(value[3], cub::Sum());
            // }
            float global_sum_t[2];

            warpReduceSum(smem_q,tidx,tidy,global_sum_t);

            // 没问题
            global_sum[0] *= __half2float(max_diff[0].x);
            global_sum[1] *= __half2float(max_diff[1].x);
            global_sum[0] += global_sum_t[0];
            global_sum[1] += global_sum_t[1];


            // 没问题
            for(int i=0;i<frag_s[0].num_elements/2;i+=4){
                HALF2(frag_s[0].x[i])[0] = HALF2(frag_s_out[0].x[i])[0];
                HALF2(frag_s[0].x[i+2])[0] = HALF2(frag_s_out[0].x[i+2])[0];
                HALF2(frag_s[0].x[i+8])[0] = HALF2(frag_s_out[0].x[i])[0];
                HALF2(frag_s[0].x[i+10])[0] = HALF2(frag_s_out[0].x[i+2])[0];
                HALF2(frag_s[1].x[i])[0] = HALF2(frag_s_out[1].x[i])[0];
                HALF2(frag_s[1].x[i+2])[0] = HALF2(frag_s_out[1].x[i+2])[0];
                HALF2(frag_s[1].x[i+8])[0] = HALF2(frag_s_out[1].x[i])[0];
                HALF2(frag_s[1].x[i+10])[0] = HALF2(frag_s_out[1].x[i+2])[0];
            }

            for(int i=0;i<4;i++){
                for(int j=0;j<frag_out[0].num_elements;j+=4){
                    HALF2(frag_out[i].x[j])[0] = __hmul2(HALF2(frag_out[i].x[j])[0],max_diff[0]);
                    HALF2(frag_out[i].x[j+2])[0] = __hmul2(HALF2(frag_out[i].x[j+2])[0],max_diff[1]);
                }
            }

            for(int i=0;i<4;i++){
                for(int j=0;j<2;j++)
                    wmma::mma_sync(frag_out[i], frag_s[j], frag_v[i][j], frag_out[i]);
            }

            asm ("cp.async.commit_group;\n" ::);
            asm ("cp.async.wait_group 0;\n" ::);

        }

        for(int i=0;i<4;i++){
            for(int j=0;j<frag_out[0].num_elements;j+=4){
                HALF2(frag_out[i].x[j])[0] = __h2div(HALF2(frag_out[i].x[j])[0],__half2half2(__float2half(global_sum[0])));
                HALF2(frag_out[i].x[j+2])[0] = __h2div(HALF2(frag_out[i].x[j+2])[0],__half2half2(__float2half(global_sum[1])));
            }
        }

        for(int i=0;i<4;i++)
            wmma::store_matrix_sync(&smem_q[tidy*16][i*16],frag_out[i],64+pad,wmma::mem_row_major);

        for(int i=0;i<4;i++)
            FLOAT4(out[data_offset_q+(i*16 + smem_index_i)*head_size+smem_index_j]) = FLOAT4(smem_q[i*16 + smem_index_i][smem_index_j]);

    }
}

void sparse_attention_lastest_larunch(half *a, half *b,half *c, half *out,int batch_size,int *seq_len_info,int *from_select_index,int *from_select_index_position,int *to_select_index,int *to_select_index_position,int block_limit, int block_num, int head_num,int block_size,int head_size)
{
    // sparse_attention_lastest<<<dim3(160),dim3(32,4)>>>(a,b,c,out,seq_len_info,from_select_index,from_select_index_position,to_select_index,to_select_index_position,batch_size,block_size,head_size);
    sparse_transformers::layers::kernels::sparse_attention_lastest<<<dim3(block_limit),dim3(32,4)>>>(a,b,c,out,seq_len_info,from_select_index,from_select_index_position,to_select_index,to_select_index_position,batch_size,block_size,head_size);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    printf("ending\n");
}


}
}
}