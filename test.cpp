#include <torch/torch.h>
#include <iostream>
#include "attention.h"
#include<iomanip>
void generate_array(float* data, int len){
    for(int i=0;i<len;i++){
        data[i] = i%100;
    }
}

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

int main()
{
    int m = 64, k = 64, n = 11*64;
    torch::Tensor query = torch::zeros({m,k},torch::kFloat);
    torch::Tensor key =torch::zeros({k,n},torch::kFloat);
    torch::Tensor out = torch::zeros({m,n},torch::kFloat).to(at::kCUDA);

    generate_array(reinterpret_cast<float*>(query.data_ptr()),m*k);
    generate_array(reinterpret_cast<float*>(key.data_ptr()),n*k);

    query = query.transpose(0,1).contiguous().to(at::kCUDA);
    key = key.to(at::kCUDA);

    float temp = 0;
    for(int i=0;i<k;i++){
        temp += i*i*11*64;
        // std::cout<<query[i][0]<<" "<<key[i][0]<<std::endl;
    }
    std::cout<<setiosflags(std::ios::fixed);

    std::cout<<std::setprecision(2)<<"temp "<<temp<<std::endl;

    test_gemm_(reinterpret_cast<float*>(query.data_ptr()),reinterpret_cast<float*>(key.data_ptr()),reinterpret_cast<float*>(out.data_ptr()),m,n,k);
    

    // auto q = ;
    query = query.transpose(0,1).contiguous().to(at::kCUDA);
    cudaEvent_t start,stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop ) ;
    // test_gemm<float><<<1,dim3(11,32)>>>(a,b,c,m,n,k,64);
    cudaEventRecord( start, 0 ) ;
    torch::Tensor out1 = torch::mm(query,key);
    cudaEventRecord(stop,0);
    float elapsedTime;
    cudaEventSynchronize(stop);
    // cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time to generate:  %f ms\n", elapsedTime );
    
    std::cout<<check_value<float>(reinterpret_cast<float*>(out.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(out1.to(at::kCPU).data_ptr()),out.numel(),out1.numel())<<std::endl;

    std::cout<<out[0][0]<<std::endl;

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