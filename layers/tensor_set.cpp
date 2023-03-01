#include "tensor_set.h"

namespace sparse_transformers{
namespace layers{

    
    TensorSet::Ptr TensorSet::m_instance_ptr = nullptr;
    std::mutex TensorSet::m_mutex;

    void TensorSet::update_tensor_set(int total_seq_len, int to_select_index_len, int to_select_index_position_len,std::vector<int> seq_len_info){
        total_seq_len_ = total_seq_len;
        to_select_index_len_ = to_select_index_len;
        to_select_index_position_len_ = to_select_index_position_len;
        seq_len_info_ = seq_len_info;
        for(auto &it:tensor_map){
            auto tensor_name = it.first;
            auto tensor = it.second;
            if(tensor_name == "tmp_qkv_out1"){
                tensor.resize_({total_seq_len,768*3}).contiguous();
            }
            else{
                tensor.resize_({total_seq_len,768}).contiguous();
            }
        }
    }
    torch::Tensor TensorSet::get_tensor(std::string tensor_name){
        if(tensor_map.find(tensor_name) != tensor_map.end()){
            return tensor_map[tensor_name];
        }
        else{
            std::cout<<"no exists tensor: "<<tensor_name<<std::endl;
            return torch::empty(0);
        }
    }
}
}