#include "./bert_attention.h"

namespace sparse_transformers {
namespace layers {
void BertAttention::operator()(const torch::Tensor &input_tensor,
        const torch::Tensor &attention_mask, torch::Tensor &output, 
        const int total_seq_len) const {
        MultiHeadedAttention::operator()(
        input_tensor, attention_mask, "self", output,
        torch::empty(0), torch::empty(0), total_seq_len,head_num_,head_size_,total_seq_len/block_size_,block_size_,d_num_);
    }
}
}