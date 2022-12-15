#include "./bert_pooler.h"
#include "./kernels/mat_mul.h"
namespace sparse_transformers {
namespace layers {
void BertPooler::operator()(const torch::Tensor& input_tensor,
                            torch::Tensor& output_tensor) const {
    
    output_tensor = torch::zeros({input_tensor.sizes()[0], dense_weight_.sizes()[0]});


    kernels::MatMul(input_tensor, false, dense_weight_, false, 1.0, output_tensor,
                    0.0);

    // add_bias and act
    // kernels::AddBiasAct<float, kernels::ActivationType::Tanh>(dense_bias_,
                                                            // output_tensor);
}

}
}