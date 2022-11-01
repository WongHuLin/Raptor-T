#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include "attention.h"
#include <iomanip>
void generate_array(float* data, int len){
    for(int i=0;i<len;i++){
        data[i] = i%100;
    }
}

//检查结果是否正确
template<class type>
bool check_value(type* A,type *B, int len_a, int len_b){
    if(len_a != len_b){
        return false;
    }
    std::cout<<setiosflags(std::ios::fixed);
    for(int i=0;i<len_a;i++){
        if(abs(A[i]-B[i])>1e-6){
            std::cout<<std::setprecision(2)<<i<<" "<<A[i]<<" "<<B[i]<<" "<<abs(A[i]-B[i])<<std::endl;
            return false;
        }
        // return false;
    }
    return true;
}

void print_tensor(float* data,int len1, int len2){
    for(int i=0;i<len1;i++){
        for(int j=0;j<len2;j++){
            std::cout<<data[i*len1+j]<<" ";
        }
        std::cout<<std::endl;
    }
}

int main()
{
    int m = 64, k = 64, n = 11*64;
    torch::Tensor query = torch::zeros({m,k},torch::kFloat);
    torch::Tensor key =torch::zeros({k,n},torch::kFloat);
    torch::Tensor value =torch::zeros({n,k},torch::kFloat);
    torch::Tensor out = torch::zeros({m,k},torch::kFloat).to(at::kCUDA);

    generate_array(reinterpret_cast<float*>(query.data_ptr()),m*k);
    generate_array(reinterpret_cast<float*>(key.data_ptr()),n*k);
    generate_array(reinterpret_cast<float*>(value.data_ptr()),n*k);


    query = query.transpose(0,1).contiguous().to(at::kCUDA);
    key = key.to(at::kCUDA);
    value = value.to(at::kCUDA);


    test_gemm_(reinterpret_cast<float*>(query.data_ptr()),reinterpret_cast<float*>(key.data_ptr()),reinterpret_cast<float*>(value.data_ptr()),reinterpret_cast<float*>(out.data_ptr()),m,n,k);
    

    // auto q = ;
    query = query.transpose(0,1).contiguous().to(at::kCUDA);

    torch::Tensor out1 = torch::mm(query,key);
    auto max_value = std::get<0>(torch::max(out1,-1)).unsqueeze(1);
    // std::cout<<out1.sizes()<<std::endl;
    // std::cout<<max_value<<std::endl;

    // auto attn_weights = torch::exp(out1 - max_value);
    auto attn_weights = torch::exp(out1 - max_value);

    auto sum_weight = attn_weights.sum(-1);
    // std::cout<<sum_weight.unsqueeze(1).sizes()<<std::endl;
    torch::Tensor out2 = torch::mm(attn_weights,value);
    std::cout<<out2[0][8]<<std::endl;
    std::cout<<sum_weight[0]<<std::endl;
    out2 = out2/sum_weight.unsqueeze(1);
    // std::cout<<out1.index({torch::indexing::Slice(0, 1),"..."})<<std::endl;
    // std::cout<<key.index({"...",torch::indexing::Slice(128,129)})<<std::endl;
    // std::cout<<out2.index({torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)})<<std::endl;
    // print_tensor(reinterpret_cast<float*>(out2.to(at::kCPU).data_ptr()),64,64);
    // std::cout<<out1.index({torch::indexing::Slice(0, 1),torch::indexing::Slice(0, 1)})<<std::endl;

    // std::cout<<out[0][0]<<" "<<out<<std::endl;

    
    std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(out.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(out2.to(at::kCPU).data_ptr()),out.numel(),out2.numel())<<std::endl;

    // std::cout<<out[0][0]<<std::endl;

    // torch::DeviceType device_type;
    // if (torch::cuda::is_available()) {
    //     std::cout << "CUDA available! Predicting on GPU." << std::endl;
    //     device_type = torch::kCUDA;
    // }
    // else {
    //     std::cout << "Predicting on CPU." << std::endl;
    //     device_type = torch::kCPU;
    // }
    // torch::Device device(device_type);
    // torch::Tensor tensor = torch::eye(3);
    // tensor = tensor.to(at::kCUDA);
    // std:cout<<tensor<<std::endl;
    std::cout<<"hello torch"<<std::endl;
    // std::cout<<tensor<<std::endl;
    return 0;
}