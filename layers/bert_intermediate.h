#pragma once
#include <memory>
#include <utility>
#include <torch/torch.h>
#include <cublas_v2.h>

namespace sparse_transformers {
namespace layers {

class BertIntermediate {
 public:
  BertIntermediate(torch::Tensor dense_weight, torch::Tensor dense_bias)
      : dense_weight_(std::move(dense_weight)),
        dense_bias_(std::move(dense_bias)) {
            cublasCreate(&handle_);
            gelu = torch::nn::GELU();
  }

  ~BertIntermediate(){
    cublasDestroy(handle_);
  }

  std::map<std::string,float> operator()(const torch::Tensor& input_tensor, torch::Tensor& output,std::map<std::string,float> &info, bool kernel_fusion = true) const;

 private:
  torch::Tensor dense_weight_;
  torch::Tensor dense_bias_;

  cublasHandle_t handle_;
  torch::nn::GELU gelu = nullptr;

};

}  // namespace layers
}  // namespace turbo_transformers
