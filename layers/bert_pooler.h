#pragma once
#include <memory>
#include <utility>
#include <torch/torch.h>

namespace sparse_transformers {
namespace layers {

class BertPooler {
 public:
  BertPooler(torch::Tensor dense_weight, torch::Tensor dense_bias)
      : dense_weight_(std::move(dense_weight)),
        dense_bias_(std::move(dense_bias)) {
  }

  void operator()(const torch::Tensor& input_tensor, torch::Tensor& output) const;

 private:
  torch::Tensor dense_weight_;
  torch::Tensor dense_bias_;
};

}  // namespace layers
}  // namespace turbo_transformers
