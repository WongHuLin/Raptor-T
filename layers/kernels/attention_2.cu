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


    __shared__ float smem_q[32][64],smem_k[64][32],temp_score[32][32],smem_v[32][64];

    __shared__ float out_temp[32][64],global_sum_scores[32],temp_smem[16][32],pre_max_score[32],max_score[32];

    float zero4[4] = {0.0f,0.0f,0.0f,0.0f};

    auto block = cooperative_groups::this_thread_block();
    auto block_v = cooperative_groups::this_thread_block();


    const int block_dim_x = blockDim.x;

    for(int a_bm = 0; a_bm< block_size/A_BM; a_bm++){
        // const int smem_index =  32*8*tidy + tidx*8; // warp_size * 8
        // const int global_a_index_i = (smem_index / 64 + a_bm*A_BM);
        // const int global_a_index_j = (smem_index % 64 );

        // // 加载Q的部分数据
        // #pragma unroll
        // for(int i=0;i<8;i+=4){
        //     FLOAT4(smem_q[smem_index/64][smem_index % 64+i]) = FLOAT4(a[data_offset_a + global_a_index_i*head_size+global_a_index_j+i]); 
        //     FLOAT4(out_temp[smem_index/64][smem_index % 64+i]) = FLOAT4(zero4[0]);
        // }


        cooperative_groups::memcpy_async(block, smem_q[0], a+data_offset_a+a_bm*A_BM*head_size, sizeof(float)*32*64);
        
        

        // __syncthreads();
        // if(a_bm == 1 && tidx == 0  && tidy == 0 && bidx == 1 && bidy == 2)
        // {
        //     for(int i=0;i<32;i++)
        //     {
        //         for(int j=0;j<64;j++)
        //             printf("%.f ",smem_q[i][j]);
        //         printf("\n");
        //     }
        // }
        // __syncthreads();

        // 初始化sharedmem
        if(tidy == 0){
            max_score[tidx] = 0.0f;
            // sum_score_max[tidx] = 0.0f;
            pre_max_score[tidx] = 0.0f;
            global_sum_scores[tidx] = 0.0f;
        }

        int data_b_start = (select_index[(bidx*g_dimy+bidy)*11+0]*g_dimy +bidy) * block_size * head_size;
        // 遍历K、V的每一个Block进行计算
        cooperative_groups::memcpy_async(block, smem_v[0], c+data_b_start, sizeof(float)*32*64);

        const int smem_index = 32*4*tidy + tidx*4; // warp_size * 8
        const int global_b_index_j = (smem_index % 32 );
        const int global_b_index_i = (smem_index / 32 );
        // 加载K
        #pragma unroll
        for(int i=0;i<2;i++){
            FLOAT4(smem_k[smem_index/32 + i*32][smem_index%32]) = FLOAT4(b[data_b_start+(global_b_index_i+i*32)*block_size+global_b_index_j]);
        }

        __syncthreads();

        for(int block_id=0;block_id<select_block_num;block_id++)
        {
            // 计算KV块的起始位置
            const int data_offset_b = (select_index[(bidx*g_dimy+bidy)*11+block_id]*g_dimy +bidy) * block_size * head_size;
            
            // KV按照 32*32 的大小进行加载计算
            for(int b_bn=0;b_bn<block_size/32;b_bn++){
                
                const int smem_index = 32*4*tidy + tidx*4; // warp_size * 8
                for(int i=0;i<B_BN;i+=4){
                    // printf("fasfasfasf");
                    FLOAT4(temp_score[smem_index/32][smem_index%32+i]) = FLOAT4(zero4[0]);
                }
                
                cooperative_groups::wait(block);

                float temp[64];
                for(int i=0;i<64;i++)
                {
                    temp[i] = smem_k[i][tidx];
                }
                // 计算Q*K
                for(int i=0;i<4;i++){
                    float4 t = {0.0f,0.0f,0.0f,0.0f};
                    float4 q_re[2];
                    q_re[0] = FLOAT4(smem_q[tidy*4+i][0]);
                    for(int j=0;j<16;j++){
                        if(j != 15)
                            q_re[1&(j+1)] = FLOAT4(smem_q[tidy*4+i][(j+1)*4]);
                        t.x += q_re[j&1].x*temp[j*4];
                        t.y += q_re[j&1].y*temp[j*4+1];
                        t.z += q_re[j&1].z*temp[j*4+2];
                        t.w += q_re[j&1].w*temp[j*4+3];
                    }
                    temp_score[tidx][tidy*4+i] += (t.x + t.y + t.z + t.w);
                }
                const int next_block_id = b_bn == 1 ? block_id+1:block_id;
                const int next_bn = (b_bn + 1) & 1;
                const int data_b_start = (select_index[(bidx*g_dimy+bidy)*11+next_block_id]*g_dimy +bidy) * block_size * head_size;
                __syncthreads();
                if(next_block_id < select_block_num)
                {

                    const int smem_index = 32*4*tidy + tidx*4; // warp_size * 8
                    const int global_b_index_j = (smem_index % 32 + next_bn * 32);
                    const int global_b_index_i = (smem_index / 32 );
                    // 加载K
                    
                    #pragma unroll
                    for(int i=0;i<2;i+=1){
                        FLOAT4(smem_k[smem_index/32+i*32][smem_index%32]) = FLOAT4(b[data_b_start+(global_b_index_i+i*32)*block_size+global_b_index_j]);
                    }

                    // 遍历K、V的每一个Block进行计算
                    // const int smem_index = 32*8*tidy + tidx*8; // warp_size * 8
                    // const int global_b_index_j = (smem_index % 32 + next_bn * 32);
                    // const int global_b_index_i = (smem_index / 32 );
                    // // 加载K
                    // #pragma unroll
                    // for(int i=0;i<8;i+=4){
                    //     FLOAT4(smem_k[smem_index/32][smem_index%32+i]) = FLOAT4(b[data_b_start+global_b_index_i*block_size+global_b_index_j+i]);
                    // }
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
                // //计算差值

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

                // // 计算S*V
                // #pragma unroll
                // for(int c_bk=0;c_bk<head_size/C_BK;c_bk++){

                //     const int smem_index = 32*4*tidy + tidx*4; // warp_size * 8
                //     const int global_b_index_j = (smem_index % 32 + c_bk*C_BK);
                //     const int global_b_index_i = (smem_index / 32 + b_bn*32);
                //     // 加载K
                //     #pragma unroll
                //     for(int i=0;i<B_BN;i+=4){
                //         FLOAT4(smem_v[smem_index/32][smem_index%32+i]) = FLOAT4(c[data_offset_b+global_b_index_i*head_size+global_b_index_j+i]);
                //     }

                //     __syncthreads();
                //     // 计算S*V
                //     #pragma unroll
                //     for(int i = 0;i<4; i++){
                //         for(int j=0;j<32;j++){
                //             out_temp[tidy*4+i][tidx+c_bk*C_BK] += temp_score[j][tidy*4+i]*smem_v[j][tidx];
                //             // if(tidx+c_bk*C_BK == 20 && tidy*4+i == 0 && a_bm == 0 && tidx == 20  && tidy == 0 && bidx == 0 && bidy == 7)
                //                 // printf("%.f %.f\n",temp_score[j][tidy*4+i],smem_v[j][tidx]);

                //         }
                //     }
                //     __syncthreads();
                // }
                #pragma unroll
                for(int i = 0;i<4; i++){
                    for(int j=0;j<32;j++){
                        out_temp[tidy*4+i][tidx] += temp_score[j][tidy*4+i]*smem_v[j][tidx];
                        out_temp[tidy*4+i][tidx+32] += temp_score[j][tidy*4+i]*smem_v[j][tidx+32];
                    }
                }
                __syncthreads();
                cooperative_groups::wait(block_v);
                if(block_id != select_block_num - 1 || b_bn != 1)
                {
                    cooperative_groups::memcpy_async(block_v, smem_v[0], c+data_b_start+next_bn*32*head_size, sizeof(float)*32*64);
                }
            }
        }

        // if(tidx == 0 && tidy == 0 && bidx == 0 && bidy == 0)
        //     printf("asfasfasf\n");
        const int index_x = (tidy%4)*8;
        const int index_y = tidx + (tidy/4)*32;
        // 结果写入global mem
        #pragma unroll
        for(int i=0;i<8;i+=1){
            // if(a_bm == 1 && tidx == 0  && tidy == 0 && bidx == 1 && bidy == 2)
            //     printf("%.f %.f %.f\n",out_temp[9][0],global_sum_scores[9],max_score[9]);
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
