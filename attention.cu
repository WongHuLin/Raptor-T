#include "attention.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

#include <thrust/extrema.h>

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <class DataType>
__global__ void sparse_attention(DataType *a,  DataType *b,  DataType *c, DataType *out, const int M, const int N,const int K,const int block_size){
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const int data_offset_a = tidx*K;
    const int data_offset_b = tidy*block_size;
    const int data_offset_out = tidx*N;

    const int A_BM = 32;
    const int A_BK = 64;
    const int B_BK = 32;
    const int B_BN = 4;
    const int C_BK = 32;
    const int C_BN = 4;


    __shared__ float smem_a[A_BM*A_BK],smem_b[11][B_BK*B_BN],temp_score[11][A_BM*B_BN],smem_c[11][C_BK*C_BN];

    __shared__ float out_temp[A_BM*64],sum_scores[11][32],global_sum_scores[32],max_values[11][32],pre_max_score[32],max_score[32];

    float zero4[4] = {0.0f,0.0f,0.0f,0.0f};



    const int block_dim_x = blockDim.x;

    for(int a_bm = 0; a_bm< M/A_BM; a_bm++){
        if(tidx < 8){
            const int smem_index =  32*8*tidx + tidy*8; // warp_size * 8
            const int global_a_index_i = (smem_index / 32);
            const int global_a_index_j = (smem_index % 32 + a_bm*A_BM);

            #pragma unroll
            for(int i=0;i<8;i+=4){
                FLOAT4(smem_a[smem_index+i]) = FLOAT4(a[global_a_index_i*M+global_a_index_j+i]); 
                FLOAT4(out_temp[smem_index+i]) = FLOAT4(zero4[0]);
            }
        }

        if(tidx == 8){
            max_score[tidy] = 0.0f;
            // sum_score_max[tidy] = 0.0f;
            pre_max_score[tidy] = 0.0f;
            global_sum_scores[tidy] = 0.0f;
        }

        for(int b_bn=0;b_bn<block_size/B_BN;b_bn++){
            for(int b_bk=0;b_bk<K/B_BK;b_bk++){
                const int smem_index = tidy*B_BN;
                const int global_b_index_i = tidy + b_bk*B_BK;
                const int global_b_index_j = tidx*block_size + b_bn*B_BN;
                for(int i=0;i<B_BN;i+=4){
                    FLOAT4(smem_b[tidx][smem_index+i]) = FLOAT4(b[global_b_index_i*N+global_b_index_j+i]);
                }

                if(b_bk == 0){
                    for(int i=0;i<B_BN;i+=4){
                        FLOAT4(temp_score[tidx][smem_index+i]) = FLOAT4(zero4[0]);
                    }
                }
                __syncthreads();

                for(int i=0;i<B_BK;i++){
                    for(int j=0;j<B_BN;j++){
                        temp_score[tidx][j*32+tidy] += smem_a[(i+b_bk*B_BK)*32+tidy]*smem_b[tidx][i*B_BN+j];
                    }
                }
                __syncthreads();

            }

            //计算最大值 rowmax
            {
                float max_value = 0.0;
                #pragma unroll
                for(int i=0;i<B_BN;i++){
                    if(max_value<temp_score[tidx][i*32+tidy]){
                        max_value = temp_score[tidx][i*32+tidy];
                    }
                }
                max_values[tidx][tidy] = max_value;
                sum_scores[tidx][tidy] = 0;
                if(tidx == 0){
                    pre_max_score[tidy] = max_score[tidy];
                    float sum = 0.0;
                    #pragma unroll
                    for(int i=0;i<11;i++){
                        if(max_score[tidy] < max_values[i][tidy]){
                            max_score[tidy] = max_values[i][tidy];
                        }
                    }
                }
            }

            //计算差值
            {
                __syncthreads();
                for(int i=0;i<B_BN;i++){
                    float temp =  exp(temp_score[tidx][i*32+tidy] - max_score[tidy]);
                    temp_score[tidx][i*32+tidy] = temp;
                    sum_scores[tidx][tidy] += temp;
                }
                if(tidx < 8){
                    const int smem_index =  32*8*tidx + tidy*8;
                    const float diff = exp(pre_max_score[tidy] - max_score[tidy]);
                    #pragma unroll
                    for(int i=0;i<8;i+=4){
                        out_temp[smem_index+i] *= diff;
                    }
                }

                __syncthreads();

                if(tidx == 9){
                    global_sum_scores[tidy] *= exp(pre_max_score[tidy]-max_score[tidy]);
                    for(int i=0;i<11;i++)
                        global_sum_scores[tidy] += sum_scores[i][tidy];
                }
            }

            for(int c_bk=0;c_bk<K/C_BK;c_bk++){
                const int smem_index = tidy*4; // warp_size * 8
                const int global_c_index_i = (smem_index / 32 + b_bn*B_BN + tidx*64);
                const int global_c_index_j = (smem_index % 32 + c_bk*C_BK);
                for(int i=0;i<C_BN;i+=4){
                    FLOAT4(smem_c[tidx][smem_index+i]) = FLOAT4(b[global_c_index_i*K+global_c_index_j+i]);
                }

                __syncthreads();
                for(int i=0;i<C_BK;i++){
                    int temp = i + tidx*2;
                    temp = temp < 32 ? temp:temp-32;
                    for(int j=0;j<B_BN;j++){
                        const int out_global_index_i = tidy;
                        const int out_global_index_j = temp + c_bk*C_BK;
                        const int index = out_global_index_i*64+out_global_index_j;
                        const float score = temp_score[tidx][j*32+tidy];

                        out_temp[index]  += score*smem_c[tidx][j*32+temp];

                    }
                    if(i&1){
                        __syncthreads();
                    }
                }

            }
        }
        if(tidx < 8){
            // const int smem_index =  32*8*tidx + tidy*8; // warp_size * 8

            // const int smem_index =  tidy*64 + tidx/4; // warp_size * 8

            const int index_x = (tidx%4)*8;
            const int index_y = tidy + (tidx/4)*32;

            #pragma unroll
            for(int i=0;i<8;i+=1){
                out[(a_bm*A_BM+index_x+i)*64+index_y] = out_temp[(index_x+i)*64+index_y] / global_sum_scores[index_x+i];
                // out[(a_bm*A_BM)*64+smem_index+i] = out_temp[smem_index+i] / global_sum_scores[tidy];

                // out_temp[smem_index+i] = out_temp[smem_index+i] / global_sum_scores[tidy];
            }

        }
        __syncthreads();
    }
}


__global__ void test_gpu(){
    float a = 1008521344.0;
    float b = 3995.0;
    float c = 19228.0;
    float d = 0;
    d = a + b * c;
    printf("%.6f %.6f %.6f %.6f\n",a,b,c,d);
}

void test_cpu(){
    float a = 1008521344.0;
    float b = 3995.0;
    float c = 19228.0;
    float d = 0;
    d = a + b * c;
    printf("%.6f %.6f %.6f %.6f\n",a,b,c,d);
}

void test_gemm_(float *a, float *b,float *c, float *out, int m, int n, int k)
{
    // std::cout<<*a<<std::endl;
    cudaEvent_t start,stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop ) ;
    // test_gemm<float><<<1,dim3(11,32)>>>(a,b,c,m,n,k,64);
    cudaEventRecord( start, 0 ) ;
    sparse_attention<float><<<1,dim3(11,32)>>>(a,b,c,out,m,n,k,64);
    // test_gpu<<<1,1>>>();
    // test_cpu();
    cudaEventRecord(stop,0);
    float elapsedTime;
    cudaEventSynchronize(stop);
    // cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time to generate:  %f ms\n", elapsedTime );
    // printf("%f\n",*a);

}
