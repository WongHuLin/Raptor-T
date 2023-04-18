#pragma once
#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <torch/torch.h>
#include <chrono>  
#include <map>
#include <string>
using namespace std::chrono;
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
void test_add_bias_and_transpose(float *bias,float *input_data,half *q, half *k,half *v, int q_offset, int k_offset, int v_offset,int *seq_len_info,int batch_size, int head_num, int block_size,int block_num, int head_size);
void test_gemm_(half *a, half *b,half *c, half *out,int *to_select_index,int *to_select_index_position, int block_num, int head_num,int block_size,int head_size);
void test_gemm_1(half *a, half *b,half *c, half *out,int *seq_len_info,int *from_select_index,int *from_select_index_position,int *to_select_index,int *to_select_index_position, int block_num, int head_num,int block_size,int head_size);
void test_gemm_1(half *a, half *b,half *c, float *out,int *seq_len_info,int *from_select_index,int *from_select_index_position,int *to_select_index,int *to_select_index_position, int block_num, int head_num,int block_size,int head_size);
void test_gemm_1(half *a, half *b,half *c, float *out,int *seq_len_info,int *from_select_index,int *from_select_index_position,int *to_select_index,int *to_select_index_position,int block_limit, int block_num, int head_num,int block_size,int head_size);
void test_add_bias_and_transpose(half *bias,half *input_data,half *q, half *k,half *v, int q_offset, int k_offset, int v_offset,int *seq_len_info,int batch_size, int head_num, int block_size,int block_num, int head_size);
void test_add_bias_and_layernorm(half *out,half *input_data, half *bias,int seq_len, int handle_row, int normalized_len, float eps,half* layernorm_weight,half* layernorm_bias);
void test_add_bias_act(half *bias, half* out, int total_seq_len, int dim_size);
void test_gemm_1(half *a, half *b,half *c, half *out,int *seq_len_info,int *from_select_index,int *from_select_index_position,int *to_select_index,int *to_select_index_position,int block_limit, int block_num, int head_num,int block_size,int head_size);
void test_gemm_1(half *a, half *b,half *c, half *out,int batch_size,int *seq_len_info,int *from_select_index,int *from_select_index_position,int *to_select_index,int *to_select_index_position,int block_limit, int block_num, int head_num,int block_size,int head_size);
void add_bias_and_transpose_kernel(const torch::Tensor&bias, torch::Tensor &input_data, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v,int q_offset, int k_offset, int v_offset,int *seq_len_info,int batch_size, int head_num, int block_size,int block_num, int head_size,std::map<std::string,float>& info);
void add_bias_and_layernorm_kernel(torch::Tensor &out,const torch::Tensor &input_data,const torch::Tensor &bias,int seq_len, int handle_row, int normalized_len, float eps,const torch::Tensor &layernorm_weight,const torch::Tensor &layernorm_bias,torch::nn::LayerNorm& layernorm,std::map<std::string,float> &info);
void add_bias_act_kernel(const torch::Tensor &bias, torch::Tensor& out, int total_seq_len, int dim_size,torch::nn::GELU gelu,std::map<std::string,float> &info);
void add_bias_act_kernel(const torch::Tensor &bias, torch::Tensor& out, int total_seq_len, int dim_size,torch::nn::GELU gelu,std::map<std::string,float>& info, bool kernel_fusion);
void add_bias_and_transpose_kernel(const torch::Tensor&bias, torch::Tensor &input_data, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v,int q_offset, int k_offset, int v_offset,int *seq_len_info,int batch_size, int head_num, int block_size,int block_num, int head_size,std::map<std::string,float>& info, bool kernel_fusion);
void add_bias_and_layernorm_kernel(torch::Tensor &out,const torch::Tensor &input_data,const torch::Tensor &bias,int seq_len, int handle_row, int normalized_len, float eps,const torch::Tensor &layernorm_weight,const torch::Tensor &layernorm_bias,torch::nn::LayerNorm& layernorm,std::map<std::string,float> &info, bool kernel_fusion);
void test_gemm_1(half *a, half *b,half *c, half *out,int batch_size,int *seq_len_info,int *from_select_index,int *from_select_index_position,int *to_select_index,int *to_select_index_position,int block_limit, int block_num, int head_num,int block_size,int head_size,std::map<std::string,float>& info,bool balanced = true);
}
}
}