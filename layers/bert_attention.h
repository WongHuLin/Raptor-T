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
#include "./multi_headed_attention.h"

namespace sparse_transformers {
namespace layers {

class BertAttention : public MultiHeadedAttention {
 public:
  BertAttention(torch::Tensor qkv_weight, torch::Tensor qkv_bias,
                torch::Tensor dense_weight, torch::Tensor dense_bias,
                torch::Tensor layer_norm_weight, torch::Tensor layer_norm_bias,
                int64_t num_attention_heads)
      : MultiHeadedAttention(
            std::move(torch::empty(0)), std::move(torch::empty(0)),
            std::move(torch::empty(0)), std::move(torch::empty(0)),
            std::move(torch::empty(0)), std::move(torch::empty(0)),
            std::move(dense_weight), std::move(dense_bias),
            std::move(qkv_weight), std::move(qkv_bias),
            std::move(layer_norm_weight),  //(768)
            std::move(layer_norm_bias), num_attention_heads) {
    head_num_ = 12;
    d_num_ = 768;
    block_size_ = 64;
    head_size_ = d_num_/head_num_;
  }

    void operator()(const torch::Tensor &input_tensor,
                  const torch::Tensor &attention_mask, torch::Tensor &output,
                  const int total_seq_len) const;

    mutable int64_t head_num_;
    mutable int64_t d_num_;
    mutable int64_t block_size_; 
    mutable int64_t head_size_;
};

}  // namespace layers
}  // namespace turbo_transformers
