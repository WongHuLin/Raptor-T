#include "./bert_intermediate.h"
#include "./kernels/mat_mul.h"
namespace sparse_transformers {
namespace layers {

void BertIntermediate::operator()(const torch::Tensor& input_tensor,
                                  torch::Tensor& output_tensor) const {

    output_tensor = torch::zeros({input_tensor.sizes()[0],input_tensor.sizes()[1],dense_weight_.sizes()[1]});


    kernels::MatMul(input_tensor, false, dense_weight_, false, 1.0, output_tensor,0.0);
    // kernels::AddBiasAct<float, kernels::ActivationType::Gelu>(
    //     dense_bias_, output_tensor, "BertIntermediate/AddBiasAct");

}

void BertIntermediate::EnforceShapeAndType() const {
  TT_ENFORCE_EQ(dense_weight_.n_dim(), 2, "dense weight must be matrix");
  TT_ENFORCE_EQ(dense_bias_.n_dim(), 1, "dense bias must be vector");
  TT_ENFORCE_EQ(dense_weight_.shape(1), dense_bias_.shape(0),
                "weight and bias shape mismatch %d, %d", dense_weight_.shape(1),
                dense_bias_.shape(0));

  if (loguru::current_verbosity_cutoff() >= 3) {
    std::ostringstream os;
    os << ">>>>>>>>>>>> query_weight <<<<<<<<<<<<" << std::endl;
    dense_weight_.Print<float>(os);
    os << ">>>>>>>>>>>> query_bias <<<<<<<<<<<<" << std::endl;
    dense_bias_.Print<float>(os);
    LOG_S(3) << os.str();
  }
}

}  // namespace layers
}  // namespace turbo_transformers
