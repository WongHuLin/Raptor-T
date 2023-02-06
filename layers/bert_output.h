#pragma once
#include <memory>
#include <utility>
#include <torch/torch.h>
namespace sparse_transformers {
namespace layers {

class BertOutput {
 public:
  BertOutput(torch::Tensor dense_weight, torch::Tensor dense_bias,
             torch::Tensor layer_norm_weight, torch::Tensor layer_norm_bias)
      : dense_weight_(std::move(dense_weight)),
        dense_bias_(std::move(dense_bias)),
        layer_norm_weight_(std::move(layer_norm_weight)),
        layer_norm_bias_(std::move(layer_norm_bias)) {
  }

  void operator()(const torch::Tensor &hidden_states,
                  const torch::Tensor &input_tensor, torch::Tensor &output) const;

 private:
  torch::Tensor dense_weight_;
  torch::Tensor dense_bias_;
  torch::Tensor layer_norm_weight_;
  torch::Tensor layer_norm_bias_;
};

}  // namespace layers
}  // namespace turbo_transformers
