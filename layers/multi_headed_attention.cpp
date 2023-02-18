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
torch::Tensor& select_index_position_tensor, std::vector<int> seq_len_info,int total_seq_len, 
int block_size, int num_rand_blocks) const{

    
    std::vector<int> select_index;
    std::vector<int> select_block_position_index;
    select_block_position_index.push_back(0);
    for(int seq_len_index = 0; seq_len_index < seq_len_info.size()-1; seq_len_index++){
        int seq_start = seq_len_info[seq_len_index];
        int seq_end = seq_len_info[seq_len_index+1];
        int block_num = seq_end - seq_start;
        int last_block = block_num - 1;
        std::vector<int> ivec(block_num);
        std::iota(ivec.begin(), ivec.end(), 0);

    for(int i=0;i<block_num;i++){

        std::vector<int> temp(ivec);
        if(i == 0 || i == block_num - 1){
            select_index.insert(select_index.end(),ivec.begin(),ivec.end());
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
            std::shuffle(temp.begin()+3,temp.end()-1,std::mt19937{ std::random_device{}()});
            select_index.insert(select_index.end(),temp.begin()+3,temp.begin()+3+num_rand_blocks);
        }
        else if (i == 2)
        {
            //global
            select_index.push_back(block_num-1);
            select_index.push_back(0);

            //ramdon
            std::shuffle(temp.begin()+4,temp.end()-1,std::mt19937{ std::random_device{}()});
            select_index.insert(select_index.end(),temp.begin()+4,temp.begin()+4+num_rand_blocks);
        }
        else if (i == block_num - 3)
        {
            //global
            select_index.push_back(0);
            select_index.push_back(block_num-1);

            //ramdon
            std::shuffle(temp.begin()+1,temp.end()-4,std::mt19937{ std::random_device{}()});
            select_index.insert(select_index.end(),temp.begin()+1,temp.begin()+1+num_rand_blocks);
        }
        else if (i == block_num - 2)
        {
            //global
            select_index.push_back(0);

            //ramdon
            std::shuffle(temp.begin()+1,temp.end()-3,std::mt19937{ std::random_device{}()});
            select_index.insert(select_index.end(),temp.begin()+1,temp.begin()+1+num_rand_blocks);
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
                // if(seq_len_index == 1){
                //     std::cout<<i<<std::endl;
                //     std::cout<<temp<<std::endl;
                // }
                temp.erase(temp.begin()+i-1,temp.begin()+i+2);
                // if(seq_len_index == 1){
                //     std::cout<<i<<std::endl;
                //     std::cout<<temp<<std::endl;
                // }
                std::shuffle(temp.begin()+1,temp.end()-1,std::mt19937{ std::random_device{}()});
                select_index.insert(select_index.end(),temp.begin()+1,temp.begin()+num_rand_blocks+1);
            }

        }
        select_block_position_index.push_back(select_index.size());
    }
    }
    select_index_tensor = torch::from_blob(select_index.data(),{int(select_index.size())},
    at::TensorOptions().dtype(torch::kInt)).clone().to(at::kCUDA);
    select_index_position_tensor = torch::from_blob(select_block_position_index.data(),
    {int(select_block_position_index.size())},at::TensorOptions().dtype(torch::kInt)).clone().to(at::kCUDA);
    sparse_index = true;
}


void MultiHeadedAttention::FuseGemm012AddBIasTranspose(
    const torch::Tensor& input_tensor, torch::Tensor& q_out, 
    torch::Tensor& k_out, torch::Tensor& v_out, torch::Tensor &seq_len_info_tensor, int total_seq_len, int d_num) const{
    
    torch::Tensor tmp_qkv_out1 = torch::zeros({total_seq_len,3*d_num},torch::kFloat).to(torch::kCUDA);

    kernels::MatMul(input_tensor,false,qkv_weight_,false,1,tmp_qkv_out1,0);

    kernels::test_add_bias_and_transpose(reinterpret_cast<float*>(qkv_bias_.data_ptr()),
    reinterpret_cast<float*>(tmp_qkv_out1.data_ptr()),reinterpret_cast<float*>(q_out.data_ptr()),
    reinterpret_cast<float*>(k_out.data_ptr()),reinterpret_cast<float*>(v_out.data_ptr()),
    0,d_num,d_num*2,reinterpret_cast<int*>(seq_len_info_tensor.data_ptr()),batch_size_,head_num_,block_size_,block_num_,head_size_);

}

void MultiHeadedAttention::operator()(
    const torch::Tensor& input_tensor, const torch::Tensor attention_mask,
    const std::string attn_type, torch::Tensor &output, torch::Tensor att_score,
    const std::vector<int> seq_len_info,torch::Tensor &seq_len_info_tensor, const int head_size, const int block_size, const int d_num, const torch::Tensor &from_select_index_position_tensor,
    const torch::Tensor &from_select_index_tensor ) const {

    std::cout<<seq_len_info<<std::endl;
    total_seq_len_ = seq_len_info.back()*block_size;
    batch_size_ = seq_len_info.size()-1;
    head_num_ = d_num/head_size;
    d_num_ = d_num;
    block_size_ = block_size;
    head_size_ = head_size;
    block_num_ =  seq_len_info.back();

    
    torch::Tensor to_select_index_tensor,to_select_index_position_tensor;
    
    torch::Tensor a;
    std::thread t1(&layers::MultiHeadedAttention::GenerateSparseBlockIndex,this,
    std::ref(to_select_index_tensor),std::ref(to_select_index_position_tensor),seq_len_info,
    total_seq_len_,block_size,3);
    // td::thread t1(&MultiHeadedAttention::GenerateSparseBlockIndex,this,111);
    t1.detach();


    torch::Tensor q_out =torch::zeros({total_seq_len_,d_num},torch::kFloat).to(torch::kCUDA);
    torch::Tensor k_out =torch::zeros({total_seq_len_,d_num},torch::kFloat).to(torch::kCUDA);
    torch::Tensor v_out =torch::zeros({total_seq_len_,d_num},torch::kFloat).to(torch::kCUDA);
    FuseGemm012AddBIasTranspose(input_tensor,q_out,k_out,v_out,seq_len_info_tensor,total_seq_len_,d_num_);
    // std::cout<<1<<std::endl;
    while(!sparse_index){

    }
    
    sparse_index = false;

    // std::cout<<partition_part_index_tensor<<std::endl;
    // std::cout<<partition_part_tensor<<std::endl;
    // std::cout<<select_index_tensor<<std::endl;
    // std::cout<<select_index_position_tensor<<std::endl;

    torch::Tensor attention_out = torch::zeros({total_seq_len_,d_num},torch::kFloat).to(torch::kCUDA);
    // kernels::test_gemm_1(reinterpret_cast<float*>(q_out.data_ptr()),reinterpret_cast<float*>(k_out.data_ptr()),
    // reinterpret_cast<float*>(v_out.data_ptr()),reinterpret_cast<float*>(attention_out.data_ptr()),reinterpret_cast<int*>(seq_len_info_tensor.data_ptr()),reinterpret_cast<int*>(from_select_index_tensor.data_ptr()),reinterpret_cast<int*>(from_select_index_position_tensor.data_ptr()),
    // reinterpret_cast<int*>(to_select_index_tensor.data_ptr()),reinterpret_cast<int*>(to_select_index_position_tensor.data_ptr()),
    // block_num_,head_num_,block_size,head_size);
    // std::cout<<2<<std::endl;

    // std::cout<<attention_out.sizes()<<std::endl;
    // std::cout<<dense_weight_.sizes()<<std::endl;
    // std::cout<<output.sizes()<<std::endl;


    kernels::MatMul(attention_out,false,dense_weight_,false,1,output,0);
    // std::cout<<3<<std::endl;


    //layernorm
    kernels::test_add_bias_and_layernorm(reinterpret_cast<float*>(output.data_ptr()),
    reinterpret_cast<float*>(output.data_ptr()),reinterpret_cast<float*>(dense_bias_.data_ptr()),
    total_seq_len_,2,d_num_,float(1e-5),reinterpret_cast<float*>(layernorm_gamma_.data_ptr()),
    reinterpret_cast<float*>(layernorm_beta_.data_ptr()));
    // std::cout<<4<<std::endl;


    }       
}
}