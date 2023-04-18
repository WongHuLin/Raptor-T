#include "./bert_attention.h"
namespace sparse_transformers {
namespace layers {
std::map<std::string,float> BertAttention::operator()(const torch::Tensor &input_tensor,
        const torch::Tensor &attention_mask, torch::Tensor &output, 
        const std::vector<int> seq_position_info,torch::Tensor &seq_len_info_tensor, const torch::Tensor &partition_part_index_tensor,
        const torch::Tensor &partition_part_tensor, const int block_limit ,std::map<std::string,float> &info, bool kernel_fusion,bool balanced) const {
        MultiHeadedAttention::operator()(
        input_tensor, attention_mask, "self", output,
        torch::empty(0), seq_position_info,seq_len_info_tensor,block_limit,head_size_,block_size_,d_num_,partition_part_index_tensor,partition_part_tensor,info,kernel_fusion,balanced);
        return info;
    }
}
}