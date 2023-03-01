#include "./bert_pooler.h"
#include "./kernels/mat_mul.h"
#include "./kernels/kernel.h"
namespace sparse_transformers {
namespace layers {
void BertPooler::operator()(const torch::Tensor& input_tensor,
                            torch::Tensor& output_tensor) const {
    
    // output_tensor = torch::zeros({input_tensor.sizes()[0], dense_weight_.sizes()[0]});

    kernels::MatMul(input_tensor, false, dense_weight_, false, 1.0, output_tensor,
                    0.0,handle_);

    // add_bias and act
    kernels::test_add_bias_act(reinterpret_cast<float*>(dense_bias_.data_ptr()),reinterpret_cast<float*>(output_tensor.data_ptr()),input_tensor.sizes()[0],dense_bias_.sizes()[0]);
}

}
}