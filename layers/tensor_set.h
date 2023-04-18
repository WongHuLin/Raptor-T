#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory> // shared_ptr
#include <mutex>  // mutex
#include <vector>

// #include<boost/enable_shared_from_this.hpp>

namespace sparse_transformers{
namespace layers{

class TensorSet
{
public:
    typedef std::shared_ptr<TensorSet> Ptr;
    static Ptr m_instance_ptr;
    std::vector<int> seq_len_info_;
    int total_seq_len_;
    int to_select_index_len_;
    int to_select_index_position_len_;

private:

    static std::mutex m_mutex;

    std::map<std::string, torch::Tensor> tensor_map;
    // std::map<std::string, std::vector<int>> tensor_dim;

public:

    TensorSet(){
    }

    TensorSet(int total_seq_len, int to_select_index_len, int to_select_index_position_len){
        total_seq_len_ = total_seq_len;
        to_select_index_len_ = to_select_index_len;
        to_select_index_position_len_ = to_select_index_position_len;
        tensor_map["q_out"] = torch::empty({total_seq_len_,768},torch::kHalf).to(torch::kCUDA).contiguous();
        tensor_map["k_out"] = torch::empty({total_seq_len_,768},torch::kHalf).to(torch::kCUDA).contiguous();
        tensor_map["v_out"] = torch::empty({total_seq_len_,768},torch::kHalf).to(torch::kCUDA).contiguous();
        tensor_map["attention_out"] = torch::empty({total_seq_len_,768},torch::kHalf).to(torch::kCUDA).contiguous();
        tensor_map["tmp_qkv_out1"] = torch::empty({total_seq_len_,768*3},torch::kHalf).to(torch::kCUDA).contiguous();


    }
    ~TensorSet(){
        std::cout<<"~TensorSet"<<std::endl;
    }

    torch::Tensor get_tensor(std::string tensor_name);

    void update_tensor_set(int total_seq_len, int to_select_index_len, int to_select_index_position_len,std::vector<int> seq_len_info);

    static Ptr get_instance(){
        // "double checked lock"
        if(m_instance_ptr==nullptr){
            std::lock_guard<std::mutex> lk(m_mutex);
            if(m_instance_ptr == nullptr){
              m_instance_ptr = std::shared_ptr<TensorSet>(new TensorSet(4096,622,65));
            }
        }
        return m_instance_ptr;
    }
};


}
}