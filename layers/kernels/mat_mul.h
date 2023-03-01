#pragma once
#include <torch/torch.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "nvToolsExt.h"
#include <cuda_runtime.h>
namespace sparse_transformers {
namespace layers {
namespace kernels {
extern void MatMul(const torch::Tensor& A, bool a_trans, const torch::Tensor& B,
        bool b_trans, float alpha, torch::Tensor& out, float beta,cublasHandle_t headle_, const std::string name = "MatMul");
extern void BatchMatMul(const torch::Tensor& A, bool a_trans,
                        const torch::Tensor& B, bool b_trans, float alpha,
                        torch::Tensor* C, float beta,
                        const std::string name = "BatchMatMul");

}  // namespace kernels
}  // namespace layers
}  // namespace sparse_transformers