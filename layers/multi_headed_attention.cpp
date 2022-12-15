#include "multi_headed_attention.h"
#include "./kernels/kernel.h"
#include "./kernels/mat_mul.h"
#include <random>
#include <pthread.h>
#include <thread>

#define ENFORCE_EQ(a, b, ...) TT_ENFORCE((a) == (b), __VA_ARGS__)

namespace sparse_transformers {
namespace layers {

void MultiHeadedAttention::GenerateSparseBlockIndex(torch::Tensor& select_index_tensor, 
torch::Tensor& select_index_position_tensor, const torch::Tensor& seq_len_info,int total_seq_len, 
int block_size, int num_rand_blocks,int last_idx) const{

    int block_num = total_seq_len/block_size;
    int last_block = block_num - 1;
    if(last_idx != -1)
        last_block = last_idx / block_size - 1;
    std::cout<<last_block<<std::endl;
    std::vector<int> ivec(last_block);
    std::iota(ivec.begin(), ivec.end(), 1);
    std::vector<int> select_index;
    std::vector<int> select_block_position_index;
    select_block_position_index.push_back(0);
    for(int i=0;i<block_num;i++){
        std::vector<int> temp(ivec);
        if(i == 0 || i == block_num - 1){
            std::vector<int> t(block_num);
            std::iota(t.begin(), t.end(), 0);
            select_index.insert(select_index.end(),t.begin(),t.end());
        }
        else{
            //slide
            select_index.push_back(i-1);
            select_index.push_back(i);
            select_index.push_back(i+1);
        }
        if (i == 1)
        {
            //global
            select_index.push_back(block_num-1);

            //ramdon
            std::shuffle(temp.begin()+2,temp.end(),std::mt19937{ std::random_device{}()});
            select_index.insert(select_index.end(),temp.begin()+2,temp.begin()+2+num_rand_blocks);
        }
        else if (i == 2)
        {
            //global
            select_index.push_back(block_num-1);
            select_index.push_back(0);

            //ramdon
            std::shuffle(temp.begin()+3,temp.end(),std::mt19937{ std::random_device{}()});
            select_index.insert(select_index.end(),temp.begin()+3,temp.begin()+3+num_rand_blocks);
        }
        else if (i == block_num - 3)
        {
            //global
            select_index.push_back(0);
            select_index.push_back(block_num-1);

            //ramdon
            std::shuffle(temp.begin(),temp.end(),std::mt19937{ std::random_device{}()});
            select_index.insert(select_index.end(),temp.begin(),temp.begin()+num_rand_blocks);
        }
        else if (i == block_num - 2)
        {
            //global
            select_index.push_back(0);

            //ramdon
            std::shuffle(temp.begin(),temp.end(),std::mt19937{ std::random_device{}()});
            select_index.insert(select_index.end(),temp.begin(),temp.begin()+num_rand_blocks);
        }
        else if (i != 0 && i != block_num - 1){
            //global
            select_index.push_back(0);
            select_index.push_back(block_num-1);

            int start = i - 2;

            if(start > last_block){
                std::shuffle(temp.begin(),temp.end(),std::mt19937{ std::random_device{}()});
                select_index.insert(select_index.end(),temp.begin(),temp.begin()+num_rand_blocks);
            }
            else if(i >= last_block){
                std::shuffle(temp.begin(),temp.begin()+start,std::mt19937{ std::random_device{}()});
                select_index.insert(select_index.end(),temp.begin(),temp.begin()+num_rand_blocks);
            }
            else{
                temp.erase(temp.begin()+i-2,temp.begin()+i);
                std::shuffle(temp.begin(),temp.end(),std::mt19937{ std::random_device{}()});
                select_index.insert(select_index.end(),temp.begin(),temp.begin()+num_rand_blocks);
            }

        }
        select_block_position_index.push_back(select_index.size());
    }
    select_index_tensor = torch::from_blob(select_index.data(),{int(select_index.size())},
    at::TensorOptions().dtype(torch::kInt)).clone().to(at::kCUDA);
    select_index_position_tensor = torch::from_blob(select_block_position_index.data(),
    {int(select_block_position_index.size())},at::TensorOptions().dtype(torch::kInt)).clone().to(at::kCUDA);
    sparse_index = true;
}


void MultiHeadedAttention::FuseGemm012AddBIasTranspose(
    const torch::Tensor& input_tensor, torch::Tensor& q_out, 
    torch::Tensor& k_out, torch::Tensor& v_out, int total_seq_len, int d_num) const{

    torch::Tensor tmp_qkv_out1 = torch::zeros({total_seq_len,3*d_num},torch::kFloat).to(torch::kCUDA);

    kernels::MatMul(input_tensor,false,qkv_weight_,false,1,tmp_qkv_out1,0);

    kernels::test_add_bias_and_transpose(reinterpret_cast<float*>(qkv_bias_.data_ptr()),
    reinterpret_cast<float*>(tmp_qkv_out1.data_ptr()),reinterpret_cast<float*>(q_out.data_ptr()),
    reinterpret_cast<float*>(k_out.data_ptr()),reinterpret_cast<float*>(v_out.data_ptr()),
    0,d_num,d_num*2,1,total_seq_len,head_num_,block_size_,block_num_,head_size_);

}

void MultiHeadedAttention::operator()(
    const torch::Tensor& input_tensor, const torch::Tensor attention_mask,
    const std::string attn_type, torch::Tensor &output, torch::Tensor att_score,
    const torch::Tensor seq_len_info, const int total_seq_len, 
    const int head_num, const int head_size, const int block_num, const int block_size, 
    const int d_num) const {
    
    torch::Tensor select_index_tensor,select_index_position_tensor;

    torch::Tensor a;
    std::thread t1(&layers::MultiHeadedAttention::GenerateSparseBlockIndex,this,
    std::ref(select_index_tensor),std::ref(select_index_position_tensor),std::ref(a),
    total_seq_len,block_size,3,1024);
    // td::thread t1(&MultiHeadedAttention::GenerateSparseBlockIndex,this,111);
    t1.detach();


    total_seq_len_ = total_seq_len;
    head_num_ = head_num;
    d_num_ = d_num;
    block_size_ = block_size;
    head_size_ = head_size;
    block_num_ = block_num;

    torch::Tensor q_out =torch::zeros({total_seq_len,d_num},torch::kFloat).to(torch::kCUDA);
    torch::Tensor k_out =torch::zeros({total_seq_len,d_num},torch::kFloat).to(torch::kCUDA);
    torch::Tensor v_out =torch::zeros({total_seq_len,d_num},torch::kFloat).to(torch::kCUDA);
    FuseGemm012AddBIasTranspose(input_tensor,q_out,k_out,v_out,4096,768);
    while(!sparse_index){
    }
    
    sparse_index = false;

    torch::Tensor attention_out = torch::zeros({total_seq_len,d_num},torch::kFloat).to(torch::kCUDA);

    kernels::test_gemm_(reinterpret_cast<float*>(q_out.data_ptr()),reinterpret_cast<float*>(k_out.data_ptr()),
    reinterpret_cast<float*>(v_out.data_ptr()),reinterpret_cast<float*>(attention_out.data_ptr()),
    reinterpret_cast<int*>(select_index_tensor.data_ptr()),reinterpret_cast<int*>(select_index_position_tensor.data_ptr()),
    block_num,head_num,block_size,head_size);

    kernels::MatMul(attention_out,false,dense_weight_,false,1,output,0);

    //layernorm
    kernels::test_add_bias_and_layernorm(reinterpret_cast<float*>(output.data_ptr()),
    reinterpret_cast<float*>(output.data_ptr()),reinterpret_cast<float*>(dense_bias_.data_ptr()),
    total_seq_len_,2,d_num_,float(1e-5),reinterpret_cast<float*>(layernorm_gamma_.data_ptr()),
    reinterpret_cast<float*>(layernorm_beta_.data_ptr()));

    }       
}
}