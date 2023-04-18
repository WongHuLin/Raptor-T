#pragma once
#include <memory>
#include <utility>
#include <torch/torch.h>
#include <cublas_v2.h>

namespace sparse_transformers {
namespace layers {

class BertOutput {
 public:
  BertOutput(torch::Tensor dense_weight, torch::Tensor dense_bias,
             torch::Tensor layer_norm_weight, torch::Tensor layer_norm_bias)
      : dense_weight_(std::move(dense_weight)),
        dense_bias_(std::move(dense_bias)),
        layer_norm_weight_(std::move(layer_norm_weight)),
        layer_norm_bias_(std::move(layer_norm_bias)){
            cublasCreate(&handle_);
            layer_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({768}).elementwise_affine(true).eps(1e-5));
           layer_norm ->to(torch::kCUDA);
           for (auto& p : layer_norm->parameters()) {
                p.set_data(p.data().to(torch::kHalf));
          }
  }

  ~BertOutput(){
    cublasDestroy(handle_);
  }

  std::map<std::string,float> operator()(const torch::Tensor &hidden_states,
                  const torch::Tensor &input_tensor, torch::Tensor &output,std::map<std::string,float> &info, bool kernel_fusion = true) const;

 private:
  torch::Tensor dense_weight_;
  torch::Tensor dense_bias_;
  torch::Tensor layer_norm_weight_;
  torch::Tensor layer_norm_bias_;

  mutable torch::nn::LayerNorm layer_norm = nullptr;
  cublasHandle_t handle_;

};

}  // namespace layers
}  // namespace turbo_transformers
