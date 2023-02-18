#include "cuda_runtime.h"
#include <cuda_fp16.h>
namespace sparse_transformers {
namespace layers {
namespace kernels {
enum class ActivationType { Gelu = 0, Tanh = 1, Relu = 2 };
void test_gemm_(float *a, float *b,float *c, float *out, int m, int n, int k);
void test_gemm_1(float *a, float *b,float *c, float *out,int *seq_len_info,int *from_select_index,int *from_select_index_position,int *to_select_index,int *to_select_index_position, int block_num, int head_num,int block_size,int head_size);
void test_gemm_(float *a, float *b,float *c, float *out,int *to_select_index,int *to_select_index_position, int block_num, int head_num,int block_size,int head_size);
void test_gemm_(half *a, half *b,half *c, half *out,int *to_select_index,int *to_select_index_position, int block_num, int head_num,int block_size,int head_size);
void test_add_bias_and_transpose(float *bias,float *input_data,float *q, float *k,float *v, int q_offset, int k_offset, int v_offset,int *seq_len_info,int batch_size, int head_num, int block_size,int block_num, int head_size);
void test_add_bias_and_layernorm(float *out,float *input_data, float *bias,int seq_len, int handle_row, int normalized_len, float eps,float* layernorm_weight,float* layernorm_bias);
void test_add_bias_act(float *bias, float* out, int total_seq_len, int dim_size);
void test_add_bias_and_transpose__(float *bias,float *input_data,float *q, float *k,float *v, int q_offset, int k_offset, int v_offset,int batch_size, int seq_len, int head_num, int block_size,int block_num, int head_size);
void test_gemm_(half *a, half *b,half *c, half *out,int *to_select_index,int *to_select_index_position, int block_num, int head_num,int block_size,int head_size);
void test_gemm_1(half *a, half *b,half *c, half *out,int *seq_len_info,int *from_select_index,int *from_select_index_position,int *to_select_index,int *to_select_index_position, int block_num, int head_num,int block_size,int head_size);
}
}
}