#include "attention.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cooperative_groups/memcpy_async.h>
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


    __shared__ float smem_q[64][32],smem_k[32][64],temp_score[32][32],smem_v[32][64];

    __shared__ float out_temp[32][64],global_sum_scores[32],temp_smem[16][32],pre_max_score[32],max_score[32];

    float zero4[4] = {0.0f,0.0f,0.0f,0.0f};

    auto block_k = cooperative_groups::this_thread_block();
    auto block_v = cooperative_groups::this_thread_block();


    const int block_dim_x = blockDim.x;

    for(int a_bm = 0; a_bm< block_size/A_BM; a_bm++){
        
        int data_b_start = (select_index[(bidx*g_dimy+bidy)*11+0]*g_dimy +bidy) * block_size * head_size;
        
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

        // for(int i=0;i<64;i++)
        //     cooperative_groups::memcpy_async(block_k, smem_k[i], a+data_offset_a+i*64+a_bm*32, sizeof(float)*32);

        // for(int i=0;i<4;i++)
        // {
        //     out_temp[tidy*4+i][tidx] = 0.0f;
        //     out_temp[tidy*4+i][tidx+32] = 0.0f;
        // }
        
        // 初始化sharedmem
        if(tidy == 0){
            max_score[tidx] = 0.0f;
            // sum_score_max[tidx] = 0.0f;
            pre_max_score[tidx] = 0.0f;
            global_sum_scores[tidx] = 0.0f;
        }

        

        __syncthreads();

        // float temp_q[64];
        // for(int i=0;i<64;i++)
        // {
        //     temp_q[i] = smem_q[i][tidx];
        // }
        // __syncthreads();

        // 遍历K、V的每一个Block进行计算
        for(int block_id=0;block_id<select_block_num;block_id++)
        {
            // 计算KV块的起始位置
            const int data_offset_b = (select_index[(bidx*g_dimy+bidy)*11+block_id]*g_dimy +bidy) * block_size * head_size;
            
            // KV按照 32*64 的大小进行加载计算
            for(int b_bn=0;b_bn<block_size/32;b_bn++){
                
                cooperative_groups::wait(block_k);

                // 计算Q*K
                // for(int i=0;i<4;i++){
                //     float4 t = {0.0f,0.0f,0.0f,0.0f};
                //     float4 k_re[2];
                //     k_re[0] = FLOAT4(smem_k[tidy*4+i][0]);
                //     for(int j=0;j<16;j++){
                //         if(j != 15)
                //             k_re[1&(j+1)] = FLOAT4(smem_k[tidy*4+i][(j+1)*4]);
                //         t.x += k_re[j&1].x*temp_q[j*4];
                //         t.y += k_re[j&1].y*temp_q[j*4+1];
                //         t.z += k_re[j&1].z*temp_q[j*4+2];
                //         t.w += k_re[j&1].w*temp_q[j*4+3];
                //     }
                //     temp_score[tidy*4+i][tidx] = (t.x + t.y + t.z + t.w);
                // }


                for(int i=0;i<4;i++){
                    float temp = 0.0f;
                    for(int j=0;j<64;j++){
                        temp += smem_q[j][tidx] * smem_k[tidy*4+i][j];
                    }
                    temp_score[tidy*4+i][tidx] = temp;
                }


                const int next_block_id = b_bn == 1 ? block_id+1:block_id;
                const int next_bn = (b_bn + 1) & 1;
                const int data_b_start = (select_index[(bidx*g_dimy+bidy)*11+next_block_id]*g_dimy +bidy) * block_size * head_size;
                __syncthreads();
                if(block_id != select_block_num - 1 || b_bn != 1)
                {
                    cooperative_groups::memcpy_async(block_k, smem_k[0], b+data_b_start+next_bn*32*head_size, sizeof(float)*32*64);
                }
            
                // if(a_bm == 0 && block_id == 0 && b_bn == 1 && tidx == 0 && tidy == 0 && bidx == 0 && bidy == 0)
                // {
                //     for(int i=0;i<32;i++)
                //     {
                //         for(int j=0;j<32;j++)
                //             printf("%.f ",temp_score[i][j]);
                //         printf("\n");
                //     }
                // }
                // __syncthreads();
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

                // __syncthreads();
                // if( a_bm == 0 && block_id == 0 && b_bn == 1 && tidx == 0 && tidy == 0 && bidx == 0 && bidy == 0)
                // {
                //     for(int i=0;i<32;i++)
                //         printf("%f ",max_score[i]);
                //     printf("\n");
                // }
                // __syncthreads();
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
                            int value1 = temp_score[tidy][tidx];
                            int value2 = temp_score[tidy+16][tidx];
                            int value3 = temp_score[tidy+8][tidx];
                            int value4 = temp_score[tidy+24][tidx];
                            temp_smem[tidy][tidx] = value1 + value2;
                            temp_smem[tidy+8][tidx] = value3 + value4;
                        }
                        else if(tidy < num){
                            int value1 = temp_smem[tidy][tidx];
                            int value2 = temp_smem[tidy+num][tidx];
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
                if(block_id != select_block_num - 1 || b_bn != 1)
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
