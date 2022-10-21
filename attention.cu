#include "attention.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <class DataType>
__global__ void test_gemm(DataType *a,  DataType *b, DataType *out, const int M, const int N,const int K,const int block_size){
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const int data_offset_a = tidx*K;
    const int data_offset_b = tidy*block_size;
    const int data_offset_out = tidx*N;

    const int BK = 16;
    const int BM = 32;
    const int BN = 32;

    const int block_dim_x = blockDim.x;

    const int sub_block_num = (M*N)/(BK*BM);

    __shared__ float smem_a[BK*BM],smem_b[11][BK*BN],smem_temp[11][32];

    // 32 is warp size
    const int sub_blcok_a_offset = (tidy/(32/BK))*M + (tidy%(32/BK))*BK;
    const int sub_blcok_b_offset = (tidy/(32/BK))*N + (tidy%(32/BK))*BK;

    const int smem_index = tidy * BK;



    for(int bm = 0; bm< M/BM; bm++){
        for(int bk = 0; bk < K/BK; bk++){
            // load partition of a
            if(tidx == 0){
                const int global_a_elem_offset = bk*M*BK + 32*bm;

                const int global_a_index_i = tidy/(32/BK) + bk*BK;
                
                const int global_a_index_j = (tidy%(32/BK))*BK + bm*BM;

                #pragma unroll
                for(int i=0;i<BK;i+=4){
                    // FLOAT4(smem_a[smem_index+i]) = FLOAT4(a[global_a_elem_offset+sub_blcok_a_offset+i]);
                    FLOAT4(smem_a[smem_index+i]) = FLOAT4(a[global_a_index_i*M+global_a_index_j+i]);
                }

            }


            for(int bn=0;bn<block_size/BN;bn++){
                // load partition of b
                const int global_b_elem_offset = tidx*block_size + bk*BK*N ;

                const int global_b_index_i = tidy/(32/BK) + bk*BK;

                const int global_b_index_j = tidx * block_size + tidy % (32/BK)*BK + bn*BN;

                #pragma unroll
                for(int i=0;i<BK;i+=4){
                    // FLOAT4(smem_b[tidx][smem_index+i]) = FLOAT4(b[global_b_elem_offset+sub_blcok_b_offset+i]);
                    FLOAT4(smem_b[tidx][smem_index+i]) = FLOAT4(b[global_b_index_i*N+global_b_index_j+i]);

                    //  __syncthreads();

                    // if(tidx == 0 && tidy == 1){
                    //     printf("global_b_elem_offset: %d sub_blcok_b_offset: %d total: %d value:%3.f smem_b:%3.f\n",global_b_elem_offset,sub_blcok_b_offset,global_b_elem_offset+sub_blcok_b_offset+i,b[global_b_elem_offset+sub_blcok_b_offset+i],smem_b[tidx][smem_index+i]);
                    //     printf("i: %d  %3.f\n",i,smem_b[tidx][64+i]);
                    // }
                }

                __syncthreads();


                // if(tidx == 0 && tidy == 0){
                //     for(int i=0;i<BK;i++){
                //         for(int j=0;j<BN;j++)
                //             printf("%3.1f ",smem_b[tidx][i*BN+j]);
                //         printf("\n");
                //     }
                //     printf("\n\n\n");
                // }
                
                // smem_temp[tidx][tidy] = ;
                
                float max = -1;
                float total_max = 0;
                float score = 1;

                for(int i=0;i<32;i++){
                    int real_index_i = i + bm * BM;
                    int real_index_j = tidx * block_size + tidy + bn * BN;
                    int temp = out[real_index_i*N + real_index_j];
                    for(int j=0;j<BK;j++){
                        
                        // float t1 = out[real_index_i*N + real_index_j];
                        // float t2 = smem_a[j*32+i]*smem_b[tidx][j*32+tidy];
                        // float t3 = t1 + t2;
                        temp += smem_a[j*32+i]*smem_b[tidx][j*32+tidy];
                        // __syncthreads();


                        // if(tidx == 0 && tidy == 0){
                            
                        // }

                        // if((real_index_i*N + real_index_j) == 32){
                        //     // printf("j: %d\n",j);
                        //     // printf("%d %3.1f\n",j*32+tidy,smem_b[tidx][j*32+tidy]);
                        //     // printf("compute  %3.6f %.6f %.6f\n",smem_a[j*32+i],smem_b[tidx][j*32+tidy],smem_a[j*32+i]*smem_b[tidx][j*32+tidy]);
                        //     // printf("%.6f %.6f %.6f %.6f\n",t1,t2,t3,out[real_index_i*N + real_index_j]);
                        //     // printf("%3.1f %3.6f %3.6f\n",out[real_index_i*N + real_index_j],smem_a[j*32+i],smem_b[tidx][j*32+tidy],smem_a[j*32+i]*smem_b[tidx][j*32+tidy]);

                        // }

                    }
                    out[real_index_i*N + real_index_j] = temp ;
                    __syncthreads();

                    // get max num

                }

                // total_max =  total_max + max

                // 分子：e^(value-max)*e
                // 分母：s = s*e*()

                
            }

        }
    }
    __syncthreads();

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

void test_gemm_(float *a, float *b, float *c, int m, int n, int k)
{
    // std::cout<<*a<<std::endl;
    cudaEvent_t start,stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop ) ;
    // test_gemm<float><<<1,dim3(11,32)>>>(a,b,c,m,n,k,64);
    cudaEventRecord( start, 0 ) ;
    test_gemm<float><<<1,dim3(11,32)>>>(a,b,c,m,n,k,64);
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

// void test_gemm_1(int *a, int *b, int *c, int m, int n, int k)
// {
//     // std::cout<<*a<<std::endl;
//     cudaEvent_t start,stop;
//     cudaEventCreate( &start );
//     cudaEventCreate( &stop ) ;
//     // test_gemm<float><<<1,dim3(11,32)>>>(a,b,c,m,n,k,64);
//     cudaEventRecord( start, 0 ) ;
//     // test_gemm<int><<<1,dim3(11,32)>>>(a,b,c,m,n,k,64);
//     cudaEventRecord(stop,0);
//     float elapsedTime;
//     cudaEventSynchronize(stop);
//     // cudaDeviceSynchronize();
//     cudaEventElapsedTime(&elapsedTime, start, stop);
//     printf( "Time to generate:  %f ms\n", elapsedTime );
//     // printf("%f\n",*a);

// }

