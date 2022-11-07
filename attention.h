#include "cuda_runtime.h"

void test_gemm_(float *a, float *b,float *c, float *out, int m, int n, int k);
void test_gemm_1(int *a, int *b, int *c, int m, int n, int k);
void test_gemm_(float *a, float *b,float *c, float *out,int *select_index, int block_num, int head_num,int block_size,int head_size);