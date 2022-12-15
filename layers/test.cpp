#include "./multi_headed_attention.h"
#include <torch/torch.h>
#include "./kernels/mat_mul.h"
#include "./kernels/kernel.h"
void test_GenerateSparseBlockIndex(){
    sparse_transformers::layers::MultiHeadedAttention multi_headed_attention = sparse_transformers::layers::MultiHeadedAttention();
    torch::Tensor select_index_tensor,select_index_position_tensor;
    multi_headed_attention.GenerateSparseBlockIndex(select_index_tensor,select_index_position_tensor,select_index_position_tensor,4096,64,3,1024);
    std::cout<<select_index_tensor<<std::endl;
    std::cout<<select_index_position_tensor<<std::endl;

}

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
        if(abs(A[i]-B[i])>1){
            std::cout<<std::setprecision(2)<<i<<" "<<A[i]<<" "<<B[i]<<" "<<abs(A[i]-B[i])<<std::endl;
            return false;
        }
        // return false;
    }
    return true;
}

void select_K_and_V(int block_num,std::vector<std::vector<int>> select_array, torch::Tensor k, torch::Tensor v, torch::Tensor& out_k, torch::Tensor& out_v,int select_len){
    for(int i=0;i<block_num;i++){
        auto temp = k.index_select(1,torch::from_blob(select_array[i].data(),{select_len},torch::kInt32).to(torch::kCUDA));
        out_k = torch::cat({out_k,temp},1);
        auto temp1 = v.index_select(1,torch::from_blob(select_array[i].data(),{select_len},torch::kInt32).to(torch::kCUDA));
        out_v = torch::cat({out_v,temp1},1);

    }
    return ;
}


void test_sparse_attention(){
    sparse_transformers::layers::MultiHeadedAttention multi_headed_attention = sparse_transformers::layers::MultiHeadedAttention();
    torch::Tensor select_index_tensor,select_index_position_tensor;
    multi_headed_attention.GenerateSparseBlockIndex(select_index_tensor,select_index_position_tensor,select_index_position_tensor,4096,64,3,1024);
    
    int seq_len = 4096, d_num = 768;
    torch::Tensor query = torch::zeros({seq_len,d_num},torch::kFloat);
    torch::Tensor key =torch::zeros({seq_len,d_num},torch::kFloat);
    torch::Tensor value =torch::zeros({seq_len,d_num},torch::kFloat);
    torch::Tensor out = torch::zeros({seq_len,d_num},torch::kFloat).to(at::kCUDA);

    generate_array(reinterpret_cast<float*>(query.data_ptr()),seq_len*d_num);
    generate_array(reinterpret_cast<float*>(key.data_ptr()),seq_len*d_num);
    generate_array(reinterpret_cast<float*>(value.data_ptr()),seq_len*d_num);

    int block_size = 64,head_num =12;
    int block_num = seq_len/block_size;
    int head_size = d_num / head_num;

    query = query.reshape({head_num,block_num,block_size,d_num/head_num}).to(at::kCUDA);
    key = key.reshape({head_num,block_num,block_size,d_num/head_num}).to(at::kCUDA);
    value = value.reshape({head_num,block_num,block_size,d_num/head_num}).to(at::kCUDA);
    out = out.reshape({head_num,block_num,block_size,d_num/head_num});

    // std::cout<<select_index_tensor<<select_index_position_tensor<<std::endl;

    sparse_transformers::layers::kernels::test_gemm_(reinterpret_cast<float*>(query.transpose(-2,-1).contiguous().data_ptr()),reinterpret_cast<float*>(key.data_ptr()),reinterpret_cast<float*>(value.data_ptr()),reinterpret_cast<float*>(out.data_ptr()),reinterpret_cast<int*>(select_index_tensor.data_ptr()),reinterpret_cast<int*>(select_index_position_tensor.data_ptr()),block_num,head_num,block_size,head_size);

    // first and last
    auto first_and_last = torch::cat({query.index({"...",torch::indexing::Slice(0, 1),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}),query.index({"...",torch::indexing::Slice(63, 64),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)})},1).reshape({12,128,64});
    auto first_and_last_score = torch::bmm(first_and_last,key.reshape({head_num,key.numel()/head_num/block_size,block_size}).transpose(-2,-1));
    first_and_last_score = first_and_last_score.softmax(-1);
    auto first_and_last_out = torch::bmm(first_and_last_score,value.reshape({head_num,key.numel()/head_num/block_size,block_size}));
    std::cout<<first_and_last_out.sizes()<<std::endl;

    std::cout<<"The first result is "<<check_value<float>(reinterpret_cast<float*>(first_and_last_out.index({"...",torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),reinterpret_cast<float*>(out.index({"...",torch::indexing::Slice(0, 1),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),first_and_last_out.numel()/2,first_and_last_out.numel()/2)<<std::endl;
    std::cout<<"The last result is "<<check_value<float>(reinterpret_cast<float*>(first_and_last_out.index({"...",torch::indexing::Slice(64, 128),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),reinterpret_cast<float*>(out.index({"...",torch::indexing::Slice(63, 64),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),first_and_last_out.numel()/2,first_and_last_out.numel()/2)<<std::endl;

    // middle part 
    std::vector<std::vector<int>> middle_index_tensor;
    auto select_index_tensor_cpu = select_index_tensor.to(torch::kCPU);
    std::vector<int> v(select_index_tensor_cpu.data_ptr<int>(),select_index_tensor_cpu.data_ptr<int>()+select_index_tensor_cpu.numel());
    for(int i=0;i<60;i++){
            std::vector<int> t(v.begin()+64+7+i*8,v.begin()+64+7+i*8+8);
            middle_index_tensor.push_back(t);
    }

    torch::Tensor k_out = torch::zeros({12,0,query.sizes()[2],query.sizes()[3]}).to(at::kCUDA);
    torch::Tensor v_out = torch::zeros({12,0,query.sizes()[2],query.sizes()[3]}).to(at::kCUDA);

    select_K_and_V(60,middle_index_tensor,key,value,k_out,v_out,8);

    auto middle_score = torch::bmm(query.index({"...",torch::indexing::Slice(2, 62),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).reshape({12,60*64,64}),k_out.reshape({head_num,k_out.numel()/head_num/block_size,block_size}).transpose(-2,-1));
    middle_score = middle_score.softmax(-1);
    auto middle_out = torch::bmm(middle_score,v_out.reshape({head_num,v_out.numel()/head_num/block_size,block_size}));

    // std::cout<<middle_out<<std::endl;
    std::cout<<"The middle result is "<<check_value<float>(reinterpret_cast<float*>(middle_out.to(at::kCPU).contiguous().data_ptr()),reinterpret_cast<float*>(out.index({"...",torch::indexing::Slice(2, 62),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),middle_out.numel(),middle_out.numel())<<std::endl;

    // the second first and last part
    std::vector<std::vector<int>> second_index_tensor;
    std::vector<int> t(v.begin()+64,v.begin()+64+7);
    middle_index_tensor.push_back(t);
    std::vector<int> t1(v.end()-64-7,v.end()-64);
    middle_index_tensor.push_back(t1);

    torch::Tensor k_out_1 = torch::zeros({12,0,query.sizes()[2],query.sizes()[3]}).to(at::kCUDA);
    torch::Tensor v_out_1 = torch::zeros({12,0,query.sizes()[2],query.sizes()[3]}).to(at::kCUDA);

    select_K_and_V(2,middle_index_tensor,key,value,k_out_1,v_out_1,7);

    auto second = torch::cat({query.index({"...",torch::indexing::Slice(1, 2),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}),query.index({"...",torch::indexing::Slice(62, 63),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)})},1).reshape({12,128,64});

    auto second_score = torch::bmm(second,k_out_1.reshape({12,k_out_1.numel()/64/12,64}).transpose(-2,-1));
    second_score = second_score.softmax(-1);
    auto second_out = torch::bmm(second_score,v_out_1.reshape({12,v_out_1.numel()/64/12,64}));
    std::cout<<"The first result is "<<check_value<float>(reinterpret_cast<float*>(second_out.index({"...",torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),reinterpret_cast<float*>(out.index({"...",torch::indexing::Slice(1, 2),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),second_out.numel()/2,second_out.numel()/2)<<std::endl;
    std::cout<<"The last result is "<<check_value<float>(reinterpret_cast<float*>(second_out.index({"...",torch::indexing::Slice(64, 128),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),reinterpret_cast<float*>(out.index({"...",torch::indexing::Slice(62, 63),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),second_out.numel()/2,second_out.numel()/2)<<std::endl;
}


void test_FuseGemm012AddBIasTranspose(){
    int seq_len = 4096, d_num = 768;

    torch::Tensor q_out = torch::zeros({seq_len,d_num},torch::kFloat).to(torch::kCUDA);
    torch::Tensor k_out = torch::zeros({seq_len,d_num},torch::kFloat).to(torch::kCUDA);
    torch::Tensor v_out = torch::zeros({seq_len,d_num},torch::kFloat).to(torch::kCUDA);

    torch::Tensor qkv_weight = torch::zeros({d_num,d_num*3},torch::kFloat);
    torch::Tensor qkv_bias = torch::zeros({d_num*3},torch::kFloat);

    generate_array(reinterpret_cast<float*>(qkv_weight.data_ptr()),d_num*d_num*3);
    generate_array(reinterpret_cast<float*>(qkv_bias.data_ptr()),3*d_num);
    qkv_weight = qkv_weight.to(torch::kCUDA);
    qkv_bias = qkv_bias.to(torch::kCUDA);


    torch::Tensor input_data = torch::zeros({seq_len,d_num},torch::kFloat);
    generate_array(reinterpret_cast<float*>(input_data.data_ptr()),seq_len*d_num);
    input_data = input_data.to(torch::kCUDA);

    sparse_transformers::layers::MultiHeadedAttention multi_headed_attention = sparse_transformers::layers::MultiHeadedAttention(torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),qkv_weight,qkv_bias,3);
    multi_headed_attention.FuseGemm012AddBIasTranspose(input_data,q_out,k_out,v_out,4096,768);

    auto qkv_temp = torch::matmul(input_data,qkv_weight);
    qkv_temp = qkv_temp + qkv_bias;
    auto query = qkv_temp.index({"...",torch::indexing::Slice(0, d_num)}).reshape({64,64,12,64}).permute({2,0,1,3}).transpose(-2,-1).contiguous();
    auto key = qkv_temp.index({"...",torch::indexing::Slice(d_num, d_num*2)}).reshape({64,64,12,64}).permute({2,0,1,3}).contiguous();
    auto value = qkv_temp.index({"...",torch::indexing::Slice(d_num*2, d_num*3)}).reshape({64,64,12,64}).permute({2,0,1,3}).contiguous();


    std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(query.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(q_out.to(at::kCPU).data_ptr()),q_out.numel(),q_out.numel())<<std::endl;
    std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(key.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(k_out.to(at::kCPU).data_ptr()),q_out.numel(),q_out.numel())<<std::endl;
    std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(value.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(v_out.to(at::kCPU).data_ptr()),q_out.numel(),q_out.numel())<<std::endl;

    


}

void test_mat_mul(){
    int seq_len = 4096, d_num = 768;
    torch::Tensor A =torch::zeros({seq_len,d_num},torch::kFloat);
    torch::Tensor B =torch::zeros({d_num,3*d_num},torch::kFloat);
    torch::Tensor out_data =torch::zeros({seq_len,3*d_num},torch::kFloat).to(torch::kCUDA);
    generate_array(reinterpret_cast<float*>(A.data_ptr()),seq_len*d_num);
    generate_array(reinterpret_cast<float*>(B.data_ptr()),3*d_num*d_num);

    A = A.to(torch::kCUDA);
    B = B.to(torch::kCUDA);

    sparse_transformers::layers::kernels::MatMul(A,false,B,false,1,out_data,0);

    auto out2 = torch::matmul(A,B);

    std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(out_data.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(out_data.to(at::kCPU).data_ptr()),out2.numel(),out_data.numel())<<std::endl;
}

void test_add_bias_and_transpose_(){
    int seq_len = 4096, d_num = 768;
    torch::Tensor input_data = torch::zeros({seq_len,d_num*3},torch::kFloat);
    torch::Tensor key =torch::zeros({seq_len,d_num},torch::kFloat).to(at::kCUDA);
    torch::Tensor value =torch::zeros({seq_len,d_num},torch::kFloat).to(at::kCUDA);
    torch::Tensor query =torch::zeros({seq_len,d_num},torch::kFloat).to(at::kCUDA);
    torch::Tensor bias = torch::zeros({d_num*3},torch::kFloat);

    generate_array(reinterpret_cast<float*>(input_data.data_ptr()),seq_len*d_num*3);
    generate_array(reinterpret_cast<float*>(bias.data_ptr()),3*d_num);

    int block_size = 64,head_num =12;
    int block_num = seq_len/block_size;
    int head_size = d_num / head_num;

    sparse_transformers::layers::kernels::test_add_bias_and_transpose(reinterpret_cast<float*>(bias.to(at::kCUDA).data_ptr()),reinterpret_cast<float*>(input_data.to(at::kCUDA).data_ptr()),reinterpret_cast<float*>(query.data_ptr()),reinterpret_cast<float*>(key.data_ptr()),reinterpret_cast<float*>(value.data_ptr()),0,d_num,d_num*2,1,4096,12,64,64,64);

    input_data = input_data + bias;


    query = query.reshape({head_num,block_num,block_size,d_num/head_num});
    key = key.reshape({head_num,block_num,block_size,d_num/head_num});
    value = value.reshape({head_num,block_num,block_size,d_num/head_num});

    auto query_1 = input_data.index({"...",torch::indexing::Slice(0, d_num)}).reshape({block_num,block_size,head_num,head_size}).permute({2,0,1,3}).transpose(-2,-1).contiguous();
    auto key_1 = input_data.index({"...",torch::indexing::Slice(d_num, d_num*2)}).reshape({block_num,block_size,head_num,head_size}).permute({2,0,1,3}).contiguous();
    auto value_1 = input_data.index({"...",torch::indexing::Slice(d_num*2, d_num*3)}).reshape({block_num,block_size,head_num,head_size}).permute({2,0,1,3}).contiguous();

    std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(query_1.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(query.to(at::kCPU).data_ptr()),key_1.numel(),key.numel())<<std::endl;

}

void test_MultiHeadedAttention_operator(){
    int seq_len = 4096, d_num = 768;

    torch::Tensor q_out = torch::zeros({seq_len,d_num},torch::kFloat).to(torch::kCUDA);
    torch::Tensor k_out = torch::zeros({seq_len,d_num},torch::kFloat).to(torch::kCUDA);
    torch::Tensor v_out = torch::zeros({seq_len,d_num},torch::kFloat).to(torch::kCUDA);

    torch::Tensor qkv_weight = torch::zeros({d_num,d_num*3},torch::kFloat);
    torch::Tensor qkv_bias = torch::zeros({d_num*3},torch::kFloat);

    generate_array(reinterpret_cast<float*>(qkv_weight.data_ptr()),d_num*d_num*3);
    generate_array(reinterpret_cast<float*>(qkv_bias.data_ptr()),3*d_num);
    qkv_weight = qkv_weight.to(torch::kCUDA);
    qkv_bias = qkv_bias.to(torch::kCUDA);


    torch::Tensor input_data = torch::zeros({seq_len,d_num},torch::kFloat);
    generate_array(reinterpret_cast<float*>(input_data.data_ptr()),seq_len*d_num);
    input_data = input_data.to(torch::kCUDA);

    sparse_transformers::layers::MultiHeadedAttention multi_headed_attention = sparse_transformers::layers::MultiHeadedAttention(torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),qkv_weight,qkv_bias,3);

    torch::Tensor out = torch::zeros({seq_len,d_num},torch::kFloat).to(torch::kCUDA);

    multi_headed_attention(input_data,torch::zeros(0),"self",out,torch::zeros(0),torch::zeros(0),4096,false,12,64,64,64,768);

    // auto qkv_temp = torch::matmul(input_data,qkv_weight);
    // qkv_temp = qkv_temp + qkv_bias;
    // auto query = qkv_temp.index({"...",torch::indexing::Slice(0, d_num)}).reshape({64,64,12,64}).permute({2,0,1,3}).transpose(-2,-1).contiguous();
    // auto key = qkv_temp.index({"...",torch::indexing::Slice(d_num, d_num*2)}).reshape({64,64,12,64}).permute({2,0,1,3}).contiguous();
    // auto value = qkv_temp.index({"...",torch::indexing::Slice(d_num*2, d_num*3)}).reshape({64,64,12,64}).permute({2,0,1,3}).contiguous();


    // std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(query.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(q_out.to(at::kCPU).data_ptr()),q_out.numel(),q_out.numel())<<std::endl;
    // std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(key.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(k_out.to(at::kCPU).data_ptr()),q_out.numel(),q_out.numel())<<std::endl;
    // std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(value.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(v_out.to(at::kCPU).data_ptr()),q_out.numel(),q_out.numel())<<std::endl;
}

int main(){
    test_MultiHeadedAttention_operator();
}