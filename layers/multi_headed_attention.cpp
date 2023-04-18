#include "multi_headed_attention.h"
#include "./kernels/kernel.h"
#include "./kernels/mat_mul.h"
#include <random>
#include <pthread.h>
#include <thread>

#include "nvToolsExt.h"
#define ENFORCE_EQ(a, b, ...) TT_ENFORCE((a) == (b), __VA_ARGS__)

namespace sparse_transformers {
namespace layers {
std::condition_variable cr;
std::mutex mtx;
void MultiHeadedAttention::GenerateSparseBlockIndex() const{
    while(true){
        if(async_)
        {
            semaphore->Wait(layer_idx_);
            if(semaphore->get_terminate_single()){
                break;
            }
        }

        std::vector<int> seq_len_info = tensor_set->seq_len_info_;
        int total_seq_len = tensor_set->total_seq_len_;
        int num_rand_blocks = 3;
        int to_select_index_len = tensor_set->to_select_index_len_;
        int to_select_index_position_len = tensor_set->to_select_index_position_len_;
        to_select_index_tensor = to_select_index_tensor.resize_({to_select_index_len}).contiguous();
        to_select_index_position_tensor = to_select_index_position_tensor.resize_({to_select_index_position_len}).contiguous();

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
        // cudaMemcpy(reinterpret_cast<int*>(to_select_index_tensor.data_ptr()),select_index.data(),sizeof(int)*select_index.size(),cudaMemcpyHostToDevice);
        // cudaMemcpy(reinterpret_cast<int*>(to_select_index_position_tensor.data_ptr()),select_block_position_index.data(),sizeof(int)*select_block_position_index.size(),cudaMemcpyHostToDevice);


        to_select_index_tensor = torch::from_blob(select_index.data(),{int(select_index.size())},at::TensorOptions().dtype(torch::kInt)).to(at::kCUDA);
        to_select_index_position_tensor = torch::from_blob(select_block_position_index.data(),{int(select_block_position_index.size())},at::TensorOptions().dtype(torch::kInt)).to(at::kCUDA);
        this->sparse_index = true;
        if(async_)
            cr.notify_all();
        else
            break;
    }
    
}


void MultiHeadedAttention::FuseGemm012AddBIasTranspose(
    const torch::Tensor& input_tensor, torch::Tensor& q_out, 
    torch::Tensor& k_out, torch::Tensor& v_out, torch::Tensor &seq_len_info_tensor, int total_seq_len, int d_num,std::map<std::string,float> &info, bool kernel_fusion) const{
    torch::Tensor tmp_qkv_out1 = tensor_set->get_tensor("tmp_qkv_out1");
    kernels::MatMul(input_tensor,false,qkv_weight_,false,1,tmp_qkv_out1,0,handle_,"FuseGemm012AddBIasTranspose_MatMul");

    kernels::add_bias_and_transpose_kernel(qkv_bias_,tmp_qkv_out1,q_out,k_out,v_out,0,d_num,d_num*2,reinterpret_cast<int*>(seq_len_info_tensor.data_ptr()),batch_size_,head_num_,block_size_,block_num_,head_size_,info,kernel_fusion);


}

void MultiHeadedAttention::operator()(
    const torch::Tensor& input_tensor, const torch::Tensor attention_mask,
    const std::string attn_type, torch::Tensor &output, torch::Tensor att_score,
    const std::vector<int> seq_len_info,torch::Tensor &seq_len_info_tensor, const int block_limit, const int head_size, const int block_size, const int d_num, const torch::Tensor &from_select_index_position_tensor,
    const torch::Tensor &from_select_index_tensor,std::map<std::string,float> &info, bool kernel_fusion, bool balanced) const {

    total_seq_len_ = seq_len_info.back()*block_size;
    batch_size_ = seq_len_info.size()-1;
    head_num_ = d_num/head_size;
    d_num_ = d_num;
    block_size_ = block_size;
    head_size_ = head_size;
    block_num_ =  seq_len_info.back();



    torch::Tensor q_out = tensor_set->get_tensor("q_out");
    torch::Tensor k_out = tensor_set->get_tensor("k_out");
    torch::Tensor v_out = tensor_set->get_tensor("v_out");


    FuseGemm012AddBIasTranspose(input_tensor,q_out,k_out,v_out,seq_len_info_tensor,total_seq_len_,d_num_,info,kernel_fusion);

    auto start_time = std::chrono::system_clock::now();

    if(async_){
        while(!sparse_index){
            std::unique_lock<std::mutex> lck(mtx);
            cr.wait(lck);
        }
        sparse_index = false;
    }
    else{
        GenerateSparseBlockIndex();
    }
    auto end_time = std::chrono::system_clock::now();
    if(info.find("pre_process") != info.end())
    {    auto dura = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count();
        info["pre_process"] += dura;
    }

    torch::Tensor attention_out = tensor_set->get_tensor("attention_out");
    

    if(attention_out.dtype() == torch::kFloat)
        kernels::test_gemm_1(reinterpret_cast<half*>(q_out.data_ptr()),reinterpret_cast<half*>(k_out.data_ptr()),reinterpret_cast<half*>(v_out.data_ptr()),reinterpret_cast<float*>(attention_out.data_ptr()),reinterpret_cast<int*>(seq_len_info_tensor.data_ptr()),reinterpret_cast<int*>(from_select_index_tensor.data_ptr()),reinterpret_cast<int*>(from_select_index_position_tensor.data_ptr()),reinterpret_cast<int*>(to_select_index_tensor.data_ptr()),reinterpret_cast<int*>(to_select_index_position_tensor.data_ptr()),block_limit,block_num_,head_num_,block_size,head_size);
    else{
        kernels::test_gemm_1(reinterpret_cast<half*>(q_out.data_ptr()),reinterpret_cast<half*>(k_out.data_ptr()),reinterpret_cast<half*>(v_out.data_ptr()),reinterpret_cast<half*>(attention_out.data_ptr()),batch_size_,reinterpret_cast<int*>(seq_len_info_tensor.data_ptr()),reinterpret_cast<int*>(from_select_index_tensor.data_ptr()),reinterpret_cast<int*>(from_select_index_position_tensor.data_ptr()),reinterpret_cast<int*>(to_select_index_tensor.data_ptr()),reinterpret_cast<int*>(to_select_index_position_tensor.data_ptr()),block_limit,block_num_,head_num_,block_size,head_size,info,balanced);
    }

    // std::cout<<attention_out.min()<<attention_out.max()<<std::endl;

    kernels::MatMul(attention_out,false,dense_weight_,false,1,output,0,handle_,"Attention Output");



    kernels::add_bias_and_layernorm_kernel(output,input_tensor,dense_bias_,total_seq_len_,2,d_num_,float(1e-5),layernorm_gamma_,layernorm_beta_,layer_norm,info,kernel_fusion);

    // std::cout<<output.min()<<output.max()<<std::endl;
    // std::cout<<torch::mean(output)<<std::endl;


    }       
}
}