#include "./bert_output.h"
#include "./kernels/mat_mul.h"
#include "./kernels/kernel.h"
namespace sparse_transformers {
namespace layers {

void BertOutput::operator()(const torch::Tensor &hidden_states,
                            const torch::Tensor &input_tensor,
                            torch::Tensor &output_tensor) const {
    // output_tensor = torch::zeros({hidden_states.sizes()[0], hidden_states.sizes()[1], dense_weight_.sizes()[1]});

  // NOTE the out of this bert layer should be the input of the next layer
  //      "BertOutput/Reshape");
    // std::cout<<"Output"<<std::endl;
    // std::cout<<hidden_states.sizes()<<std::endl;
    // std::cout<<output_tensor.sizes()<<std::endl;


    kernels::MatMul(hidden_states, false, dense_weight_, false, 1.0,
                  output_tensor, 0.0,handle_,"BertOutput");
    int seq_len = input_tensor.sizes()[0];

    kernels::test_add_bias_and_layernorm(reinterpret_cast<float*>(output_tensor.data_ptr()),
    reinterpret_cast<float*>(input_tensor.data_ptr()),reinterpret_cast<float*>(dense_bias_.data_ptr()),
    seq_len,2,768,float(1e-5),reinterpret_cast<float*>(layer_norm_weight_.data_ptr()),
    reinterpret_cast<float*>(layer_norm_bias_.data_ptr()));

}

}  // namespace layers
}  // namespace turbo_transformers
