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
    //(12,64)(32*8)
//input_data: seq_len * all_head_size * 3   bias: all_head_size  q,k,v: seq * all_head_size
template <class DataType>
__global__ void add_bias_and_transpose(DataType *input_data, DataType *bias, 
    DataType *q, DataType *k, DataType *v, int q_offset, int k_offset, 
    int v_offset,int *seq_len_info,int batch_size, int head_num, int block_size, 
    int head_size, int block_num){
    // For K and V: batch_size * seq_len * all_head_size -> batch_size * head_num * block_num * block_size * head_size
    // For Q: batch_size * seq_len * all_head_size -> batch_size * head_num * block_num * head_size* block_size
    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int all_head_size = head_num * head_size;
    const int start_read_data_index = bidy * block_size * head_size * 3 * head_num + bidx * head_size;
    const int start_write_data_index = bidx * block_num * block_size * head_size + bidy * block_size * head_size;
    __shared__ float q_bias[64],k_bias[64],v_bias[64];

    int read_seq_data_start = 0;
    int write_seq_data_start = 0;

    for(int i=1;i<batch_size+1;i++)
        if(bidy >= seq_len_info[i])
            continue;
        else{
            int seq_start_index = seq_len_info[i-1];
            int seq_len_index = bidy - seq_start_index;  //seq内的index
            int len = (seq_len_info[i] - seq_len_info[i-1]); //seq 的长度
            // 数据开始读取为：seq的开始位置 + 当前token block的index开始位置 + 
            read_seq_data_start =  seq_start_index*block_size*head_num*head_size*3 + seq_len_index*block_size*head_num*head_size*3 + bidx*head_size;
            write_seq_data_start = seq_start_index*block_size*head_num*head_size + bidx*len*block_size*head_size + seq_len_index*block_size*head_size;
            // if(tidx == 0 && tidy == 0){
            //     printf("%d %d %d %d %d %d\n",bidx,bidy,seq_start_index*block_size*head_num*head_size,bidx*len*block_size*head_size, seq_len_index*block_size*head_size,write_seq_data_start);
            // }
            break;
        }

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
            int read_index = read_seq_data_start + head_num*head_size*3*(block_row+smem_row_index) + block_col + smem_col_index;
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

            int write_index = write_seq_data_start + (block_row+smem_row_index)*head_size + block_col + smem_col_index;
            // if(tidx == 0 && tidy == 0 && block_row == 0 && block_col == 0){
            //     printf("%d %d %f\n",bidx,write_index,smem_k[0]);
            // }
            // __syncthreads();
            FLOAT4(k[write_index]) = FLOAT4(smem_k[smem_index]);
            FLOAT4(v[write_index]) = FLOAT4(smem_v[smem_index]);

            __syncthreads();
            // if(bidx == 0 && bidy == 0  && block_col == 0){
            //     printf("%d %d %d %d %d %d %f\n",tidx,tidy,block_row,block_col,write_seq_data_start,write_seq_data_start + (tidy*4+1+block_col)*head_size + tidx + block_row,smem_q[tidx][(tidy*4+1)]);
            // }
            for(int i=0;i<4;i++){
                q[write_seq_data_start + (tidy*4+i+block_col)*head_size + tidx + block_row] = smem_q[tidx][(tidy*4+i)];
                // if(write_seq_data_start + (tidy*4+i+block_col)*head_size + tidx + block_row == 3079)
                //     printf("%d %d %d %d %d 3079 %f \n",bidx,bidy,tidx,tidy,write_seq_data_start,smem_q[tidx][(tidy*4+i)]);
            }
            __syncthreads();
        }
    }
}

void test_add_bias_and_transpose(float *bias,float *input_data,float *q, float *k,float *v, int q_offset, int k_offset, int v_offset,int *seq_len_info,int batch_size, int head_num, int block_size,int block_num, int head_size){
    cudaEvent_t start,stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop ) ;
    cudaEventRecord( start, 0 ) ;
    add_bias_and_transpose<float><<<dim3(head_num,block_num),dim3(32,8)>>>(input_data,bias,q,k,v,q_offset,k_offset,v_offset,seq_len_info,batch_size,head_num,block_size,head_size,block_num);
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