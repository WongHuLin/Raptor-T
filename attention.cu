#include "attention.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

#include <thrust/extrema.h>

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


// lock-based
__device__ volatile int g_mutex;
// GPU lock-based synchronization function
__device__ void __gpu_sync(int goalVal )
{
    // thread ID in a block
    int tid_in_block = threadIdx.x * blockDim.y + threadIdx.y;
    // only thread 0 is used for synchronization
    if (tid_in_block == 0)
    {
    	atomicAdd((int*) &g_mutex, 1);
    	// only when all blocks add 1 go g_mutex
    	// will g_mutex equal to goalVal
    	while (g_mutex != goalVal)
    	{
    		// Do nothing here
    	}
    }
    __syncthreads();
}
  

template <class DataType>
__global__ void sparse_attention(DataType *a,  DataType *b,  DataType *c, DataType *out,const int *select_index,const int block_size,const int head_size,const int select_block_num){


    const int tidy = threadIdx.y;
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int b_dimx = 8;
    const int g_dimy = gridDim.y;

    // 计算Q的起始位置
    const int data_offset_a = (bidx*g_dimy + bidy) * block_size * head_size;

    const int A_BM = 32;
    const int A_BK = 64;
    const int B_BK = 32;
    const int B_BN = 4;
    const int C_BK = 32;
    const int C_BN = 4;


    __shared__ float smem_a[A_BM*A_BK],smem_b[8][B_BK*B_BN],temp_score[8][A_BM*B_BN],smem_c[8][C_BK*C_BN];

    __shared__ float out_temp[A_BM*64],sum_scores[8][32],global_sum_scores[32],max_values[8][32],pre_max_score[32],max_score[32];

    float zero4[4] = {0.0f,0.0f,0.0f,0.0f};



    const int block_dim_x = blockDim.x;

    for(int a_bm = 0; a_bm< block_size/A_BM; a_bm++){
        const int smem_index =  32*8*tidy + tidx*8; // warp_size * 8
        const int global_a_index_i = (smem_index / 32);
        const int global_a_index_j = (smem_index % 32 + a_bm*A_BM);

        // 加载Q的部分数据
        #pragma unroll
        for(int i=0;i<8;i+=4){
            FLOAT4(smem_a[smem_index+i]) = FLOAT4(a[data_offset_a + global_a_index_i*block_size+global_a_index_j+i]); 
            FLOAT4(out_temp[smem_index+i]) = FLOAT4(zero4[0]);
        }

        // 初始化sharedmem
        if(tidy == 0){
            max_score[tidx] = 0.0f;
            // sum_score_max[tidx] = 0.0f;
            pre_max_score[tidx] = 0.0f;
            global_sum_scores[tidx] = 0.0f;
        }

        //遍历K、V的每一个Block进行计算
        for(int block_id=0;block_id<select_block_num;block_id++)
        {
            // 计算KV块的起始位置
            const int data_offset_b = (select_index[(bidx*g_dimy+bidy)*11+block_id]*g_dimy +bidy) * block_size * head_size;

            // KV按照 32*32 的大小进行加载计算
            for(int b_bn=0;b_bn<block_size/32;b_bn++){
                
                #pragma unroll
                for(int b_bk=0;b_bk<head_size/B_BK;b_bk++){
                    const int smem_index = tidx*B_BN; // warp_size * 8
                    const int global_b_index_i = (smem_index / 32 + tidy*4 + b_bn*32);
                    const int global_b_index_j = (smem_index % 32 + b_bk*B_BK);
                    // 加载K
                    #pragma unroll
                    for(int i=0;i<B_BN;i+=4){
                        FLOAT4(smem_b[tidy][smem_index+i]) = FLOAT4(b[data_offset_b+global_b_index_i*head_size+global_b_index_j+i]);
                    }

                    if(b_bk == 0){
                        for(int i=0;i<B_BN;i+=4){
                            FLOAT4(temp_score[tidy][smem_index+i]) = FLOAT4(zero4[0]);
                        }
                    }
                    __syncthreads();
                    
                    // 计算Q*K
                    for(int i=0;i<B_BK;i++){
                        for(int j=0;j<B_BN;j++){
                            temp_score[tidy][j*32+tidx] += smem_a[(i+b_bk*B_BK)*32+tidx]*smem_b[tidy][j*B_BK+i];
                        }
                    }
                    __syncthreads();

                }

                //计算最大值 rowmax
                {
                    float max_value = 0.0;
                    #pragma unroll
                    for(int i=0;i<B_BN;i++){
                        if(max_value<temp_score[tidy][i*32+tidx]){
                            max_value = temp_score[tidy][i*32+tidx];
                        }
                    }
                    max_values[tidy][tidx] = max_value;
                    sum_scores[tidy][tidx] = 0;
                    if(tidy == 0){
                        pre_max_score[tidx] = max_score[tidx];
                        #pragma unroll
                        for(int i=0;i<8;i++){
                            if(max_score[tidx] < max_values[i][tidx]){
                                max_score[tidx] = max_values[i][tidx];
                            }
                        }
                    }
                }

                //计算差值
                {
                    __syncthreads();
                    #pragma unroll
                    for(int i=0;i<B_BN;i++){
                        float temp =  exp(temp_score[tidy][i*32+tidx] - max_score[tidx]);
                        temp_score[tidy][i*32+tidx] = temp;
                        sum_scores[tidy][tidx] += temp;
                    }
                    float diff = pre_max_score[tidx] - max_score[tidx];
                    
                    const int smem_index =  tidx*64 + tidy*8;
                    const float diff_exp = exp(diff);
                    if(diff != 0)
                    {
                        #pragma unroll
                        for(int i=0;i<8;i++){
                            out_temp[smem_index+i] *= diff_exp;
                        }
                    }

                    __syncthreads();
                    if(tidy == 0){
                        float before = global_sum_scores[tidx];
                        global_sum_scores[tidx] *= diff_exp;
                        float then = global_sum_scores[tidx];
                        for(int i=0;i<8;i++)
                            global_sum_scores[tidx] += sum_scores[i][tidx];

                        // if(tidx == 0 && tidy == 0 && bidx == 0 && bidy == 0  && a_bm == 0)
                        //     printf("%.f %.f %.f %.f %.f %.f %.f\n",pre_max_score[tidx],max_score[tidx],before,diff,exp(diff),then,global_sum_scores[tidx]);
                    }
                }

                // 计算S*V
                #pragma unroll
                for(int c_bk=0;c_bk<head_size/C_BK;c_bk++){

                    const int smem_index = tidx*4; // warp_size * 8
                    const int global_c_index_i = (smem_index / 32 + tidy*4 + b_bn*32);
                    const int global_c_index_j = (smem_index % 32 + c_bk*C_BK);
                    // 加载V
                    #pragma unroll
                    for(int i=0;i<C_BN;i+=4){
                        FLOAT4(smem_c[tidy][smem_index+i]) = FLOAT4(c[data_offset_b+global_c_index_i*head_size+global_c_index_j+i]);
                    }

                    __syncthreads();
                    // 计算S*V
                    #pragma unroll
                    for(int i = 0;i<32;i += b_dimx){
                        for(int j=0;j<32;j++){
                            const int out_global_index_i = tidy + i;
                            const int out_global_index_j = tidx + c_bk*C_BK;
                            const int index = out_global_index_i*64+out_global_index_j;
                            const float score = temp_score[j/4][(j%4)*32+out_global_index_i];
                            out_temp[index]  += score*smem_c[j/4][(j%4)*32+tidx];
                        }
                    }
                    __syncthreads();
                }
            }
        }

        const int index_x = (tidy%4)*8;
        const int index_y = tidx + (tidy/4)*32;
        // 结果写入global mem
        #pragma unroll
        for(int i=0;i<8;i+=1){
            out[data_offset_a+(a_bm*A_BM+index_x+i)*head_size+index_y] = out_temp[(index_x+i)*64+index_y] / global_sum_scores[index_x+i];
        }

        __syncthreads();
        // printf("111\n");
    }
}


void test_cpu(){
    float a = 1008521344.0;
    float b = 3995.0;
    float c = 19228.0;
    float d = 0;
    d = a + b * c;
    printf("%.6f %.6f %.6f %.6f\n",a,b,c,d);
}

void test_gemm_(float *a, float *b,float *c, float *out,int *select_index, int block_num, int head_num,int block_size,int head_size)
{
    // std::cout<<*a<<std::endl;
    // sparse_attention<float><<<dim3(block_num,head_num),dim3(11,32)>>>(a,b,c,out,select_index,64,64);

    cudaEvent_t start,stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop ) ;
    // test_gemm<float><<<1,dim3(11,32)>>>(a,b,c,m,n,k,64);

    cudaEventRecord( start, 0 ) ;
    sparse_attention<float><<<dim3(block_num,head_num),dim3(32,8)>>>(a,b,c,out,select_index,64,64,11);

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
