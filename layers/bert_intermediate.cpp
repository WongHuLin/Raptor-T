#include "./bert_intermediate.h"
#include "./kernels/mat_mul.h"
#include "./kernels/kernel.h"
namespace sparse_transformers {
namespace layers {

std::map<std::string,float> BertIntermediate::operator()(const torch::Tensor& input_tensor,
                                  torch::Tensor& output_tensor,std::map<std::string,float> &info, bool kernel_fusion) const {

    // output_tensor = torch::zeros({input_tensor.sizes()[0],input_tensor.sizes()[1],dense_weight_.sizes()[1]});
    // std::cout<<"BertIntermediate"<<std::endl;
    // std::cout<<input_tensor.sizes()<<std::endl;
    // std::cout<<output_tensor.sizes()<<std::endl;

    kernels::MatMul(input_tensor, false, dense_weight_, false, 1.0, output_tensor,0.0,handle_,"BertIntermediate");

    kernels::add_bias_act_kernel(dense_bias_,output_tensor,input_tensor.sizes()[0],dense_bias_.sizes()[0],gelu,info,kernel_fusion);


    // if(dense_bias_.dtype() == torch::kFloat)
    //     kernels::test_add_bias_act(reinterpret_cast<float*>(dense_bias_.data_ptr()),reinterpret_cast<float*>(output_tensor.data_ptr()),input_tensor.sizes()[0],dense_bias_.sizes()[0]);
    // else
    //     kernels::test_add_bias_act(reinterpret_cast<half*>(dense_bias_.data_ptr()),reinterpret_cast<half*>(output_tensor.data_ptr()),input_tensor.sizes()[0],dense_bias_.sizes()[0]);
    // kernels::AddBiasAct<float, kernels::ActivationType::Gelu>(
    //     dense_bias_, output_tensor, "BertIntermediate/AddBiasAct");

    return info;

}


}  // namespace layers
}  // namespace turbo_transformers
