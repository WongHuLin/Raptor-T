// Copyright (C) 2020 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.

#pragma once


#include <torch/torch.h>
#include <torch/script.h>
#include <cublas_v2.h>

#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <future>

#include "../layers/tensor_set.h"
#include "../layers/semaphore.h"

namespace sparse_transformers {
namespace layers {

class MultiHeadedAttention {
public:

  MultiHeadedAttention(){}

  MultiHeadedAttention(torch::Tensor k_weight, torch::Tensor k_bias,
                       torch::Tensor v_weight, torch::Tensor v_bias,
                       torch::Tensor q_weight, torch::Tensor q_bias,
                       torch::Tensor dense_weight, torch::Tensor dense_bias,
                       torch::Tensor qkv_weight, torch::Tensor qkv_bias,
                       int64_t num_attention_heads)
      : k_weight_(std::move(k_weight)),  //(768, 768)
        k_bias_(std::move(k_bias)),
        v_weight_(std::move(v_weight)),  //(768, 768)
        v_bias_(std::move(v_bias)),
        q_weight_(std::move(q_weight)),  //(768, 768)
        q_bias_(std::move(q_bias)),
        dense_weight_(std::move(dense_weight)),
        dense_bias_(std::move(dense_bias)),
        qkv_weight_(std::move(qkv_weight)),
        qkv_bias_(std::move(qkv_bias)),
        layernorm_gamma_(torch::empty(0)),
        layernorm_beta_(torch::empty(0)),
        num_attention_heads_(num_attention_heads) {
            sparse_index = false;
            tensor_set = TensorSet::get_instance();
            semaphore = Semaphore::get_instance();
            cublasCreate(&handle_);
            to_select_index_tensor = torch::empty({622},torch::kInt).to(torch::kCUDA).contiguous();
            to_select_index_position_tensor = torch::empty({65},torch::kInt).to(torch::kCUDA).contiguous();
            thread_ =  std::async(std::launch::async,&MultiHeadedAttention::GenerateSparseBlockIndex,this);
            std::cout<<"init"<<std::endl;
  }
    
    MultiHeadedAttention(torch::Tensor k_weight, torch::Tensor k_bias,
                         torch::Tensor v_weight, torch::Tensor v_bias,
                         torch::Tensor q_weight, torch::Tensor q_bias,
                         torch::Tensor dense_weight, torch::Tensor dense_bias,
                         torch::Tensor qkv_weight, torch::Tensor qkv_bias,
                         torch::Tensor layernorm_gamma,
                         torch::Tensor layernorm_beta,
                         int64_t num_attention_heads, int layer_idx)
        : k_weight_(std::move(k_weight)),  //(768, 768)
          k_bias_(std::move(k_bias)),
          v_weight_(std::move(v_weight)),  //(768, 768)
          v_bias_(std::move(v_bias)),
          q_weight_(std::move(q_weight)),  //(768, 768)
          q_bias_(std::move(q_bias)),
          dense_weight_(std::move(dense_weight)),
          dense_bias_(std::move(dense_bias)),
          qkv_weight_(std::move(qkv_weight)),
          qkv_bias_(std::move(qkv_bias)),
          layernorm_gamma_(std::move(layernorm_gamma)),
          layernorm_beta_(std::move(layernorm_beta)),
          num_attention_heads_(num_attention_heads),
          layer_idx_(layer_idx){
            sparse_index = false;
            tensor_set = TensorSet::get_instance();
            semaphore = Semaphore::get_instance();
            semaphore -> register_layer(layer_idx_);
            cublasCreate(&handle_);
            to_select_index_tensor = torch::empty({622},torch::kInt).to(torch::kCUDA).contiguous();
            to_select_index_position_tensor = torch::empty({65},torch::kInt).to(torch::kCUDA).contiguous();
            thread_ =  std::async(std::launch::async,&MultiHeadedAttention::GenerateSparseBlockIndex,this);

    }

    ~MultiHeadedAttention(){
      cublasDestroy(handle_);
    }

    // void GenerateSparseBlockIndex(torch::Tensor& select_index_tensor, 
    // torch::Tensor& select_index_position_tensor, std::vector<int> seq_len_info,
    // int total_seq_len, int block_size, int num_rand_blocks) const;
    void GenerateSparseBlockIndex() const;


    void SetContextFlag(
      const std::unordered_map<std::string, torch::Tensor*>& layer_cache) const;

    void FuseGemm012AddBIasTranspose(
    const torch::Tensor& input_tensor, torch::Tensor& q_out, 
    torch::Tensor& k_out, torch::Tensor& v_out) const;

    void FuseGemm012AddBIasTranspose(
    const torch::Tensor& input_tensor, torch::Tensor& q_out, 
    torch::Tensor& k_out, torch::Tensor& v_out, torch::Tensor &seq_len_info_tensor, int total_seq_len, int d_num) const;
    
    void operator()(
    const torch::Tensor& input_tensor, const torch::Tensor attention_mask,
    const std::string attn_type, torch::Tensor &output, torch::Tensor att_score,
    const std::vector<int> seq_len_info,torch::Tensor &seq_len_info_tensor, const int block_limit, const int head_size, const int block_size, const int d_num, const torch::Tensor &from_select_index_position_tensor,
    const torch::Tensor &from_select_index_tensor ) const ;

private:
    torch::Tensor k_weight_;
    torch::Tensor k_bias_;
    torch::Tensor v_weight_; 
    torch::Tensor v_bias_;
    torch::Tensor q_weight_;
    torch::Tensor q_bias_;
    torch::Tensor dense_weight_;
    torch::Tensor dense_bias_;

    torch::Tensor qkv_weight_;  // store fused qkv weight/bias for "self" type
                             // attention computation.
    torch::Tensor qkv_bias_;

    torch::Tensor layernorm_gamma_;
    torch::Tensor layernorm_beta_;

    mutable torch::Tensor to_select_index_tensor;
    mutable torch::Tensor to_select_index_position_tensor;

    int64_t num_attention_heads_;

    mutable int total_seq_len_;
    mutable int head_num_;
    mutable int d_num_;
    mutable int block_size_; 
    mutable int head_size_;
    mutable int block_num_; 
    mutable int batch_size_;
    mutable bool sparse_index;


    mutable int64_t size_per_head_;
    // mutable torch::Device devtype_;
    
    cublasHandle_t handle_;
    TensorSet::Ptr tensor_set;
    Semaphore::Ptr semaphore;
    std::future<void> thread_;
    int layer_idx_;
};

}  // namespace layers
}  // namespace turbo_transformers
