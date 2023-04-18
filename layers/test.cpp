// #include "./multi_headed_attention.h"
#include <torch/torch.h>
#include "./kernels/mat_mul.h"
#include "./kernels/kernel.h"
// #include "./bert_output.h"
// #include "./bert_intermediate.h"
#include "cuda_fp16.h"
// #include "./tensor_set.h"

// -3 - 3 的均匀分布
template<class type>
void generate_array(type* data, int len){
    for(int i=0;i<len;i++){
        data[i] = type(6.0/4095*(i%4096) - 2);
    }
}

//检查结果是否正确
// template<class type>
bool check_value(half* A,float *B, int len_a, int len_b){
    if(len_a != len_b){
        return false;
    }
    std::cout<<setiosflags(std::ios::fixed);
    for(int i=0;i<len_a;i++){
        if(abs(__half2float(A[i])-B[i])>0.01){
            std::cout<<std::setprecision(2)<<i<<" "<<__half2float(A[i])<<" "<<B[i]<<" "<<abs(A[i]-B[i])<<std::endl;
            return false;
        }
        // return false;
    }
    return true;
}

bool check_value(half* A,half *B, int len_a, int len_b){
    if(len_a != len_b){
        return false;
    }
    std::cout<<setiosflags(std::ios::fixed);
    for(int i=0;i<len_a;i++){
        if(abs(__half2float( A[i])-__half2float(B[i]))>0.1){
            std::cout<<std::setprecision(2)<<i<<" "<<__half2float(A[i])<<" "<<__half2float(B[i])<<" "<<abs(A[i]-B[i])<<std::endl;
            return false;
        }
        // return false;
    }
    return true;
}

bool check_value(float* A,float *B, int len_a, int len_b){
    if(len_a != len_b){
        return false;
    }
    std::cout<<setiosflags(std::ios::fixed);
    for(int i=0;i<len_a;i++){
        if(abs((A[i])-B[i])>1){
            std::cout<<std::setprecision(2)<<i<<" "<<(A[i])<<" "<<B[i]<<" "<<abs(A[i]-B[i])<<std::endl;
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
    // sparse_transformers::layers::MultiHeadedAttention multi_headed_attention = sparse_transformers::layers::MultiHeadedAttention();
    std::vector<int> seq_len_info = {0,64};
    // sparse_transformers::layers::TensorSet t(4096,622,65);
    // auto tensor_set  = sparse_transformers::layers::TensorSet::get_instance();
    // torch::Tensor to_select_index_ = tensor_set->get_tensor("to_select_index_tensor");
    //  torch::Tensor to_select_index_position_ = tensor_set->get_tensor("to_select_index_position_tensor");
    // multi_headed_attention.GenerateSparseBlockIndex(to_select_index_,to_select_index_position_,seq_len_info,4096,64,3);
    // std::cout<<to_select_index_<<std::endl;
    // std::cout<<to_select_index_position_<<std::endl;
    
    std::vector<int> to_select_index_position = {0,64,71,79,87,95,103,111,119,127,135,143,151,159,167,175,183,191,199,207,215,223,231,239,247,255,263,271,279,287,295,303,311,319,327,335,343,351,359,367,375,383,391,399,407,415,423,431,439,447,455,463,471,479,487,495,503,511,519,527,535,543,551,558,622};

    std::vector<int> to_select_index = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,0,1,2,63,45,22,30,1,2,3,63,0,61,32,20,2,3,4,0,63,13,40,47,3,4,5,0,63,55,57,24,4,5,6,0,63,41,33,14,5,6,7,0,63,52,37,36,6,7,8,0,63,23,20,45,7,8,9,0,63,39,3,51,8,9,10,0,63,3,39,27,9,10,11,0,63,27,60,25,10,11,12,0,63,41,14,49,11,12,13,0,63,53,48,60,12,13,14,0,63,49,18,8,13,14,15,0,63,24,16,52,14,15,16,0,63,32,61,10,15,16,17,0,63,60,46,3,16,17,18,0,63,58,51,8,17,18,19,0,63,43,60,4,18,19,20,0,63,29,24,62,19,20,21,0,63,55,34,9,20,21,22,0,63,59,13,27,21,22,23,0,63,19,39,52,22,23,24,0,63,6,55,37,23,24,25,0,63,51,15,32,24,25,26,0,63,22,51,19,25,26,27,0,63,3,31,34,26,27,28,0,63,51,14,36,27,28,29,0,63,39,56,50,28,29,30,0,63,15,40,45,29,30,31,0,63,17,60,21,30,31,32,0,63,14,55,60,31,32,33,0,63,62,17,51,32,33,34,0,63,19,42,53,33,34,35,0,63,14,42,25,34,35,36,0,63,30,53,50,35,36,37,0,63,1,10,45,36,37,38,0,63,43,8,19,37,38,39,0,63,41,19,5,38,39,40,0,63,27,26,60,39,40,41,0,63,5,30,60,40,41,42,0,63,62,15,24,41,42,43,0,63,20,31,51,42,43,44,0,63,59,21,37,43,44,45,0,63,12,19,6,44,45,46,0,63,5,12,57,45,46,47,0,63,5,48,38,46,47,48,0,63,34,56,52,47,48,49,0,63,5,29,32,48,49,50,0,63,17,4,5,49,50,51,0,63,2,3,12,50,51,52,0,63,28,2,8,51,52,53,0,63,43,4,49,52,53,54,0,63,29,37,13,53,54,55,0,63,50,33,45,54,55,56,0,63,19,21,41,55,56,57,0,63,25,8,34,56,57,58,0,63,15,40,27,57,58,59,0,63,29,55,17,58,59,60,0,63,37,55,46,59,60,61,0,63,21,49,41,60,61,62,0,63,23,59,2,61,62,63,0,33,2,49,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63};



    std::vector<int> from_select_index_position = {
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
         14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  30,  36,  42,
         48,  54,  60,  66,  72,  78,  84,  90,  96, 102, 108, 114, 120, 126,
        132, 138, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210,
        216, 222, 228, 234, 240, 246, 252, 258, 264, 270, 276, 282, 288, 294,
        300, 306, 312, 318, 324, 330, 336, 342, 348, 354, 360, 366, 372, 378,
        384, 390, 396, 402, 408, 413, 418, 423, 428, 433, 438, 443, 448, 453,
        458, 463, 468, 473, 478, 483, 488, 493, 498, 503, 508, 513, 518, 523,
        528, 533, 538, 543, 548, 553, 558, 563, 568, 573, 578, 583, 588, 593,
        598, 603, 608, 613, 618, 623, 628, 633, 638, 643, 648, 653, 658, 663,
        668, 673, 678, 683, 688, 693, 698, 703, 708, 713, 718, 723, 728, 733,
        738, 743, 748, 753, 758, 763, 768};
    
    std::vector<int> from_select_index ={
          0,  64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704,  63, 127,
        191, 255, 319, 383, 447, 511, 575, 639, 703, 767,   2, 269, 536,  36,
        303, 570,  66, 333, 600, 100, 367, 634, 130, 397, 664, 164, 431, 698,
        194, 461, 728, 228, 495, 762, 258, 525,  25, 292, 559,  59, 322, 589,
         89, 356, 623, 123, 386, 653, 153, 420, 687, 187, 450, 717, 217, 484,
        751, 251, 514,  14, 281, 548,  48, 315, 578,  78, 345, 612, 112, 379,
        642, 142, 409, 676, 176, 443, 706, 206, 473, 740, 240, 507,   3, 270,
        537,  37, 304, 571,  67, 334, 601, 101, 368, 635, 131, 398, 665, 165,
        432, 699, 195, 462, 729, 229, 496, 763, 259, 526,  26, 293, 560,  60,
        323, 590,  90, 357, 624, 124, 387, 654, 154, 421, 688, 188, 451, 718,
        218, 485, 752, 252, 515,  15, 282, 549,  49, 316, 579,  79, 346, 613,
        113, 380, 643, 143, 410, 677, 177, 444, 707, 207, 474, 741, 241, 508,
          4, 271, 538,  38, 305, 572,  68, 335, 602, 102, 369, 636, 132, 399,
        666, 166, 433, 700, 196, 463, 730, 230, 497, 764, 260, 527,  27, 294,
        561,  61, 324, 591,  91, 358, 625, 125, 388, 655, 155, 422, 689, 189,
        452, 719, 219, 486, 753, 253, 516,  16, 283, 550,  50, 317, 580,  80,
        347, 614, 114, 381, 644, 144, 411, 678, 178, 445, 708, 208, 475, 742,
        242, 509,   5, 272, 539,  39, 306, 573,  69, 336, 603, 103, 370, 637,
        133, 400, 667, 167, 434, 701, 197, 464, 731, 231, 498, 765, 261, 528,
         28, 295, 562,   1, 325, 592,  92, 359, 626,  65, 389, 656, 156, 423,
        690, 129, 453, 720, 220, 487, 754, 193, 517,  17, 284, 551,  51, 257,
        581,  81, 348, 615, 115, 321, 645, 145, 412, 679, 179, 385, 709, 209,
        476, 743, 243, 449,   6, 273, 540,  40, 307, 513,  70, 337, 604, 104,
        371, 577, 134, 401, 668, 168, 435, 641, 198, 465, 732, 232, 499, 705,
        262, 529,  29, 296, 563,  62, 326, 593,  93, 360, 627, 126, 390, 657,
        157, 424, 691, 190, 454, 721, 221, 488, 755, 254, 518,  18, 285, 552,
         52, 318, 582,  82, 349, 616, 116, 382, 646, 146, 413, 680, 180, 446,
        710, 210, 477, 744, 244, 510,   7, 274, 541,  41, 308, 574,  71, 338,
        605, 105, 372, 638, 135, 402, 669, 169, 436, 702, 199, 466, 733, 233,
        500, 766, 263, 530,  30, 297, 564, 327, 594,  94, 361, 628, 391, 658,
        158, 425, 692, 455, 722, 222, 489, 756, 519,  19, 286, 553,  53, 583,
         83, 350, 617, 117, 647, 147, 414, 681, 181, 711, 211, 478, 745, 245,
          8, 275, 542,  42, 309,  72, 339, 606, 106, 373, 136, 403, 670, 170,
        437, 200, 467, 734, 234, 501, 264, 531,  31, 298, 565, 328, 595,  95,
        362, 629, 392, 659, 159, 426, 693, 456, 723, 223, 490, 757, 520,  20,
        287, 554,  54, 584,  84, 351, 618, 118, 648, 148, 415, 682, 182, 712,
        212, 479, 746, 246,   9, 276, 543,  43, 310,  73, 340, 607, 107, 374,
        137, 404, 671, 171, 438, 201, 468, 735, 235, 502, 265, 532,  32, 299,
        566, 329, 596,  96, 363, 630, 393, 660, 160, 427, 694, 457, 724, 224,
        491, 758, 521,  21, 288, 555,  55, 585,  85, 352, 619, 119, 649, 149,
        416, 683, 183, 713, 213, 480, 747, 247,  10, 277, 544,  44, 311,  74,
        341, 608, 108, 375, 138, 405, 672, 172, 439, 202, 469, 736, 236, 503,
        266, 533,  33, 300, 567, 330, 597,  97, 364, 631, 394, 661, 161, 428,
        695, 458, 725, 225, 492, 759, 522,  22, 289, 556,  56, 586,  86, 353,
        620, 120, 650, 150, 417, 684, 184, 714, 214, 481, 748, 248,  11, 278,
        545,  45, 312,  75, 342, 609, 109, 376, 139, 406, 673, 173, 440, 203,
        470, 737, 237, 504, 267, 534,  34, 301, 568, 331, 598,  98, 365, 632,
        395, 662, 162, 429, 696, 459, 726, 226, 493, 760, 523,  23, 290, 557,
         57, 587,  87, 354, 621, 121, 651, 151, 418, 685, 185, 715, 215, 482,
        749, 249,  12, 279, 546,  46, 313,  76, 343, 610, 110, 377, 140, 407,
        674, 174, 441, 204, 471, 738, 238, 505, 268, 535,  35, 302, 569, 332,
        599,  99, 366, 633, 396, 663, 163, 430, 697, 460, 727, 227, 494, 761,
        524,  24, 291, 558,  58, 588,  88, 355, 622, 122, 652, 152, 419, 686,
        186, 716, 216, 483, 750, 250,  13, 280, 547,  47, 314,  77, 344, 611,
        111, 378, 141, 408, 675, 175, 442, 205, 472, 739, 239, 506};
        
    auto to_select_index_tensor =  torch::from_blob(to_select_index.data(),{int(to_select_index.size())},torch::kInt32).to(torch::kCUDA);
    auto to_select_index_position_tensor =  torch::from_blob(to_select_index_position.data(),{int(to_select_index_position.size())},torch::kInt32).to(torch::kCUDA);

    auto from_select_index_tensor =  torch::from_blob(from_select_index.data(),{int(from_select_index.size())},torch::kInt32).to(torch::kCUDA);
    auto from_select_index_position_tensor =  torch::from_blob(from_select_index_position.data(),{int(from_select_index_position.size())},torch::kInt32).to(torch::kCUDA);

    auto seq_len_info_tensor =  torch::from_blob(seq_len_info.data(),{3},torch::kInt32).to(torch::kCUDA);

    int seq_len = 4096, d_num = 768;
    int block_num = 64, head_num = 12, block_size = 64, head_size = 64;

    torch::Tensor query = torch::zeros({seq_len,d_num},torch::kFloat32);
    torch::Tensor key =torch::zeros({seq_len,d_num},torch::kFloat32);
    torch::Tensor value =torch::zeros({seq_len,d_num},torch::kFloat32);
    torch::Tensor out = torch::zeros({seq_len,d_num},torch::kFloat16).to(at::kCUDA).contiguous();
    torch::Tensor out_1 = torch::zeros({4096,d_num},torch::kFloat32).to(at::kCUDA).contiguous();


    generate_array<float>(reinterpret_cast<float*>(query.data_ptr()),seq_len*d_num);
    generate_array<float>(reinterpret_cast<float*>(key.data_ptr()),seq_len*d_num);
    generate_array<float>(reinterpret_cast<float*>(value.data_ptr()),seq_len*d_num);

    // std::cout<<value.reshape({head_num,64,block_size,d_num/head_num}).index({torch::indexing::Slice(0,1),torch::indexing::Slice(0,1),"..."})<<std::endl;

    query = query.to(at::kCUDA);
    key = key.to(at::kCUDA);
    value = value.to(at::kCUDA);

    // std::cout<<value.reshape({head_num,64,block_size,d_num/head_num}).index({torch::indexing::Slice(0,1),torch::indexing::Slice(0,1),"..."})<<std::endl;


    sparse_transformers::layers::kernels::test_gemm_1(reinterpret_cast<half*>(query.toType(torch::kFloat16).to(at::kCUDA).contiguous().data_ptr()),reinterpret_cast<half*>(key.toType(torch::kFloat16).to(at::kCUDA).contiguous().data_ptr()),reinterpret_cast<half*>(value.toType(torch::kFloat16).contiguous().to(at::kCUDA).data_ptr()),reinterpret_cast<half*>(out.data_ptr()),1,reinterpret_cast<int*>(seq_len_info_tensor.data_ptr()),reinterpret_cast<int*>(from_select_index_tensor.data_ptr()),reinterpret_cast<int*>(from_select_index_position_tensor.data_ptr()),reinterpret_cast<int*>(to_select_index_tensor.data_ptr()),reinterpret_cast<int*>(to_select_index_position_tensor.data_ptr()),160,block_num,head_num,block_size,head_size);

    // sparse_transformers::layers::kernels::test_gemm_(reinterpret_cast<float*>(query.index({torch::indexing::Slice(0,4096),"..."}).reshape({head_num,64,block_size,d_num/head_num}).transpose(-2,-1).contiguous().data_ptr()),reinterpret_cast<float*>(key.index({torch::indexing::Slice(0,4096),"..."}).reshape({head_num,64,block_size,d_num/head_num}).contiguous().data_ptr()),reinterpret_cast<float*>(value.index({torch::indexing::Slice(0,4096),"..."}).contiguous().data_ptr()),reinterpret_cast<float*>(out_1.contiguous().data_ptr()),reinterpret_cast<int*>(to_select_index_tensor.data_ptr()),reinterpret_cast<int*>(to_select_index_position_tensor.data_ptr()),64,12,64,64);


     sparse_transformers::layers::kernels::test_gemm_(reinterpret_cast<float*>(query.index({torch::indexing::Slice(0,4096),"..."}).reshape({head_num,64,block_size,d_num/head_num}).transpose(-2,-1).contiguous().data_ptr()),reinterpret_cast<float*>(key.index({torch::indexing::Slice(0,4096),"..."}).contiguous().data_ptr()),reinterpret_cast<float*>(value.index({torch::indexing::Slice(0,4096),"..."}).contiguous().data_ptr()),reinterpret_cast<float*>(out_1.data_ptr()),reinterpret_cast<int*>(to_select_index_tensor.data_ptr()),reinterpret_cast<int*>(to_select_index_position_tensor.data_ptr()),64,12,64,64);

   
    

    auto query_1 = query.index({torch::indexing::Slice(0,4096),"..."}).reshape({head_num,64,block_size,d_num/head_num}).index({torch::indexing::Slice(0,1),torch::indexing::Slice(0,1),"..."});
    auto key_1 = key.index({torch::indexing::Slice(0,4096),"..."}).reshape({head_num,64,block_size,d_num/head_num}).index({torch::indexing::Slice(0, 1),torch::indexing::Slice(0, 64),"..."});
    auto value_1 = value.index({torch::indexing::Slice(0,4096),"..."}).reshape({head_num,64,block_size,d_num/head_num}).index({torch::indexing::Slice(0, 1),torch::indexing::Slice(0, 64),"..."});

    auto temp_score = torch::bmm(query_1.reshape({1,64,64}),key_1.reshape({1,64*64,64}).transpose(-2,-1)).contiguous();

    // for(int i=0;i<64;i++){
    //     for(int j=0;j<64;j++){
    //         std::cout<<reinterpret_cast<float*>(temp_score.data_ptr())[i*4096+j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    std::tuple<torch::Tensor, torch::Tensor> max_test = torch::max(temp_score,-1);
    auto max_val = std::get<0>(max_test).unsqueeze(-1);
    // std::cout<<max_val<<std::endl;

    temp_score = temp_score.softmax(-1);
    // temp_score = torch::exp(temp_score - max_val);
    
    // auto sum_value = torch::sum(temp_score,-1);
    

    // torch::Tensor sum_test = torch::sum(temp_score,-1);
    // std::cout<<sum_test<<std::endl;
    auto out_2 = torch::bmm(temp_score,value_1.reshape({1,64*64,64}));

    // auto temp_score = torch::mm(query_1.index({"...",torch::indexing::Slice(0,32),torch::indexing::Slice(0,64)}).reshape({32,64}),key_1.index({"...",torch::indexing::Slice(0,1),torch::indexing::Slice(0,32),torch::indexing::Slice(0,64)}).reshape({32,64}).transpose(-2,-1));
    // std::cout<<temp_score;

    // auto out_2 = torch::mm(temp_score,value_1.index({"...",torch::indexing::Slice(0,1),torch::indexing::Slice(0,32),torch::indexing::Slice(0,64)}).reshape({32,64}));
    std::cout<<out[0][0];


    std::cout<<"The first result is "<<check_value(reinterpret_cast<half*>(out.index({torch::indexing::Slice(0,4096),"..."}).reshape({head_num,64,block_size,d_num/head_num}).to(at::kCPU).contiguous().data_ptr()),reinterpret_cast<half*>(out_1.toType(torch::kFloat16).to(at::kCPU).contiguous().data_ptr()),out_1.numel(),out_1.numel())<<std::endl;

}
// void test_sparse_attention_1(){

//     sparse_transformers::layers::MultiHeadedAttention multi_headed_attention = sparse_transformers::layers::MultiHeadedAttention();
//     std::vector<int> seq_len_info = {0,64};
//     torch::Tensor to_select_index,to_select_index_position;
//     multi_headed_attention.GenerateSparseBlockIndex(to_select_index,to_select_index_position,seq_len_info,4096,64,3);

//     int seq_len = 4096, d_num = 768;
//     torch::Tensor query = torch::zeros({seq_len,d_num},torch::kFloat);
//     torch::Tensor key =torch::zeros({seq_len,d_num},torch::kFloat);
//     torch::Tensor value =torch::zeros({seq_len,d_num},torch::kFloat);
//     torch::Tensor out = torch::zeros({seq_len,d_num},torch::kFloat).to(at::kCUDA);

//     generate_array<float>(reinterpret_cast<float*>(query.data_ptr()),seq_len*d_num);
//     generate_array<float>(reinterpret_cast<float*>(key.data_ptr()),seq_len*d_num);
//     generate_array<float>(reinterpret_cast<float*>(value.data_ptr()),seq_len*d_num);

//     int block_size = 64,head_num =12;
//     int block_num = seq_len/block_size;
//     int head_size = d_num / head_num;

//     query = query.reshape({head_num,block_num,block_size,d_num/head_num}).to(at::kCUDA);
//     key = key.reshape({head_num,block_num,block_size,d_num/head_num}).to(at::kCUDA);
//     value = value.reshape({head_num,block_num,block_size,d_num/head_num}).to(at::kCUDA);
//     out = out.reshape({head_num,block_num,block_size,d_num/head_num});

//     // std::cout<<select_index_tensor<<select_index_position_tensor<<std::endl;

//     sparse_transformers::layers::kernels::test_gemm_(reinterpret_cast<float*>(query.transpose(-2,-1).contiguous().data_ptr()),reinterpret_cast<float*>(key.data_ptr()),reinterpret_cast<float*>(value.data_ptr()),reinterpret_cast<float*>(out.data_ptr()),reinterpret_cast<int*>(to_select_index.data_ptr()),reinterpret_cast<int*>(to_select_index_position.data_ptr()),block_num,head_num,block_size,head_size);

//     // first and last
//     auto first_and_last = torch::cat({query.index({"...",torch::indexing::Slice(0, 1),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}),query.index({"...",torch::indexing::Slice(63, 64),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)})},1).reshape({12,128,64});
//     auto first_and_last_score = torch::bmm(first_and_last,key.reshape({head_num,key.numel()/head_num/block_size,block_size}).transpose(-2,-1));
//     first_and_last_score = first_and_last_score.softmax(-1);
//     auto first_and_last_out = torch::bmm(first_and_last_score,value.reshape({head_num,key.numel()/head_num/block_size,block_size}));
//     std::cout<<first_and_last_out.index({torch::indexing::Slice(0, 1),torch::indexing::Slice(0, 1),torch::indexing::Slice(0, 1)});
//     // std::cout<<first_and_last_out.sizes()<<std::endl;

//     std::cout<<"The first result is "<<check_value<float>(reinterpret_cast<float*>(first_and_last_out.index({"...",torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),reinterpret_cast<float*>(out.index({"...",torch::indexing::Slice(0, 1),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),first_and_last_out.numel()/2,first_and_last_out.numel()/2)<<std::endl;
//     std::cout<<"The last result is "<<check_value<float>(reinterpret_cast<float*>(first_and_last_out.index({"...",torch::indexing::Slice(64, 128),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),reinterpret_cast<float*>(out.index({"...",torch::indexing::Slice(63, 64),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),first_and_last_out.numel()/2,first_and_last_out.numel()/2)<<std::endl;

//     // middle part 
//     std::vector<std::vector<int>> middle_index_tensor;
//     auto select_index_tensor_cpu = to_select_index.to(torch::kCPU);
//     std::vector<int> v(select_index_tensor_cpu.data_ptr<int>(),select_index_tensor_cpu.data_ptr<int>()+select_index_tensor_cpu.numel());
//     for(int i=0;i<60;i++){
//             std::vector<int> t(v.begin()+64+7+i*8,v.begin()+64+7+i*8+8);
//             middle_index_tensor.push_back(t);
//     }

//     torch::Tensor k_out = torch::zeros({12,0,query.sizes()[2],query.sizes()[3]}).to(at::kCUDA);
//     torch::Tensor v_out = torch::zeros({12,0,query.sizes()[2],query.sizes()[3]}).to(at::kCUDA);

//     select_K_and_V(60,middle_index_tensor,key,value,k_out,v_out,8);

//     auto middle_score = torch::bmm(query.index({"...",torch::indexing::Slice(2, 62),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).reshape({12,60*64,64}),k_out.reshape({head_num,k_out.numel()/head_num/block_size,block_size}).transpose(-2,-1));
//     middle_score = middle_score.softmax(-1);
//     auto middle_out = torch::bmm(middle_score,v_out.reshape({head_num,v_out.numel()/head_num/block_size,block_size}));

//     // std::cout<<middle_out<<std::endl;
//     std::cout<<"The middle result is "<<check_value<float>(reinterpret_cast<float*>(middle_out.to(at::kCPU).contiguous().data_ptr()),reinterpret_cast<float*>(out.index({"...",torch::indexing::Slice(2, 62),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),middle_out.numel(),middle_out.numel())<<std::endl;

//     // the second first and last part
//     std::vector<std::vector<int>> second_index_tensor;
//     std::vector<int> t(v.begin()+64,v.begin()+64+7);
//     middle_index_tensor.push_back(t);
//     std::vector<int> t1(v.end()-64-7,v.end()-64);
//     middle_index_tensor.push_back(t1);

//     torch::Tensor k_out_1 = torch::zeros({12,0,query.sizes()[2],query.sizes()[3]}).to(at::kCUDA);
//     torch::Tensor v_out_1 = torch::zeros({12,0,query.sizes()[2],query.sizes()[3]}).to(at::kCUDA);

//     select_K_and_V(2,middle_index_tensor,key,value,k_out_1,v_out_1,7);

//     auto second = torch::cat({query.index({"...",torch::indexing::Slice(1, 2),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}),query.index({"...",torch::indexing::Slice(62, 63),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)})},1).reshape({12,128,64});

//     auto second_score = torch::bmm(second,k_out_1.reshape({12,k_out_1.numel()/64/12,64}).transpose(-2,-1));
//     second_score = second_score.softmax(-1);
//     auto second_out = torch::bmm(second_score,v_out_1.reshape({12,v_out_1.numel()/64/12,64}));
//     std::cout<<"The first result is "<<check_value<float>(reinterpret_cast<float*>(second_out.index({"...",torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),reinterpret_cast<float*>(out.index({"...",torch::indexing::Slice(1, 2),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),second_out.numel()/2,second_out.numel()/2)<<std::endl;
//     std::cout<<"The last result is "<<check_value<float>(reinterpret_cast<float*>(second_out.index({"...",torch::indexing::Slice(64, 128),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),reinterpret_cast<float*>(out.index({"...",torch::indexing::Slice(62, 63),torch::indexing::Slice(0, 64),torch::indexing::Slice(0, 64)}).to(at::kCPU).contiguous().data_ptr()),second_out.numel()/2,second_out.numel()/2)<<std::endl;
// }


// void test_FuseGemm012AddBIasTranspose(){
//     int seq_len = 4096, d_num = 768;

//     torch::Tensor q_out = torch::zeros({seq_len,d_num},torch::kFloat).to(torch::kCUDA);
//     torch::Tensor k_out = torch::zeros({seq_len,d_num},torch::kFloat).to(torch::kCUDA);
//     torch::Tensor v_out = torch::zeros({seq_len,d_num},torch::kFloat).to(torch::kCUDA);

//     torch::Tensor qkv_weight = torch::zeros({d_num,d_num*3},torch::kFloat);
//     torch::Tensor qkv_bias = torch::zeros({d_num*3},torch::kFloat);

//     generate_array<float>(reinterpret_cast<float*>(qkv_weight.data_ptr()),d_num*d_num*3);
//     generate_array<float>(reinterpret_cast<float*>(qkv_bias.data_ptr()),3*d_num);
//     qkv_weight = qkv_weight.to(torch::kCUDA);
//     qkv_bias = qkv_bias.to(torch::kCUDA);


//     torch::Tensor input_data = torch::zeros({seq_len,d_num},torch::kFloat);
//     generate_array<float>(reinterpret_cast<float*>(input_data.data_ptr()),seq_len*d_num);
//     input_data = input_data.to(torch::kCUDA);

//     sparse_transformers::layers::MultiHeadedAttention multi_headed_attention = sparse_transformers::layers::MultiHeadedAttention(torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),qkv_weight,qkv_bias,3);
//     multi_headed_attention.FuseGemm012AddBIasTranspose(input_data,q_out,k_out,v_out,4096,768);

//     auto qkv_temp = torch::matmul(input_data,qkv_weight);
//     qkv_temp = qkv_temp + qkv_bias;
//     auto query = qkv_temp.index({"...",torch::indexing::Slice(0, d_num)}).reshape({64,64,12,64}).permute({2,0,1,3}).transpose(-2,-1).contiguous();
//     auto key = qkv_temp.index({"...",torch::indexing::Slice(d_num, d_num*2)}).reshape({64,64,12,64}).permute({2,0,1,3}).contiguous();
//     auto value = qkv_temp.index({"...",torch::indexing::Slice(d_num*2, d_num*3)}).reshape({64,64,12,64}).permute({2,0,1,3}).contiguous();


//     std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(query.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(q_out.to(at::kCPU).data_ptr()),q_out.numel(),q_out.numel())<<std::endl;
//     std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(key.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(k_out.to(at::kCPU).data_ptr()),q_out.numel(),q_out.numel())<<std::endl;
//     std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(value.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(v_out.to(at::kCPU).data_ptr()),q_out.numel(),q_out.numel())<<std::endl;

    


// }

// void test_mat_mul(){
//     int seq_len = 4096, d_num = 768;
//     torch::Tensor A =torch::zeros({seq_len,d_num},torch::kFloat);
//     torch::Tensor B =torch::zeros({d_num,3*d_num},torch::kFloat);
//     torch::Tensor out_data =torch::zeros({seq_len,3*d_num},torch::kFloat).to(torch::kCUDA);
//     generate_array<float>(reinterpret_cast<float*>(A.data_ptr()),seq_len*d_num);
//     generate_array<float>(reinterpret_cast<float*>(B.data_ptr()),3*d_num*d_num);

//     A = A.to(torch::kCUDA);
//     B = B.to(torch::kCUDA);

//     sparse_transformers::layers::kernels::MatMul(A,false,B,false,1,out_data,0);

//     auto out2 = torch::matmul(A,B);

//     std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(out_data.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(out_data.to(at::kCPU).data_ptr()),out2.numel(),out_data.numel())<<std::endl;
// }

// void test_add_bias_and_transpose_(){
//     int seq_len = 4096, d_num = 768;
//     torch::Tensor input_data = torch::zeros({seq_len,d_num*3},torch::kFloat);
//     torch::Tensor key =torch::zeros({seq_len,d_num},torch::kFloat).to(at::kCUDA);
//     torch::Tensor value =torch::zeros({seq_len,d_num},torch::kFloat).to(at::kCUDA);
//     torch::Tensor query =torch::zeros({seq_len,d_num},torch::kFloat).to(at::kCUDA);
//     torch::Tensor bias = torch::zeros({d_num*3},torch::kFloat);

//     generate_array<float>(reinterpret_cast<float*>(input_data.data_ptr()),seq_len*d_num*3);
//     generate_array<float>(reinterpret_cast<float*>(bias.data_ptr()),3*d_num);

//     int block_size = 64,head_num =12;
//     int block_num = seq_len/block_size;
//     int head_size = d_num / head_num;

//     sparse_transformers::layers::kernels::test_add_bias_and_transpose(reinterpret_cast<float*>(bias.to(at::kCUDA).data_ptr()),reinterpret_cast<float*>(input_data.to(at::kCUDA).data_ptr()),reinterpret_cast<float*>(query.data_ptr()),reinterpret_cast<float*>(key.data_ptr()),reinterpret_cast<float*>(value.data_ptr()),0,d_num,d_num*2,1,4096,12,64,64,64);

//     input_data = input_data + bias;


//     query = query.reshape({head_num,block_num,block_size,d_num/head_num});
//     key = key.reshape({head_num,block_num,block_size,d_num/head_num});
//     value = value.reshape({head_num,block_num,block_size,d_num/head_num});

//     auto query_1 = input_data.index({"...",torch::indexing::Slice(0, d_num)}).reshape({block_num,block_size,head_num,head_size}).permute({2,0,1,3}).transpose(-2,-1).contiguous();
//     auto key_1 = input_data.index({"...",torch::indexing::Slice(d_num, d_num*2)}).reshape({block_num,block_size,head_num,head_size}).permute({2,0,1,3}).contiguous();
//     auto value_1 = input_data.index({"...",torch::indexing::Slice(d_num*2, d_num*3)}).reshape({block_num,block_size,head_num,head_size}).permute({2,0,1,3}).contiguous();

//     std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(query_1.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(query.to(at::kCPU).data_ptr()),key_1.numel(),key.numel())<<std::endl;

// }

// void test_MultiHeadedAttention_operator(){
//     int seq_len = 4096, d_num = 768;

//     torch::Tensor q_out = torch::zeros({seq_len,d_num},torch::kFloat).to(torch::kCUDA);
//     torch::Tensor k_out = torch::zeros({seq_len,d_num},torch::kFloat).to(torch::kCUDA);
//     torch::Tensor v_out = torch::zeros({seq_len,d_num},torch::kFloat).to(torch::kCUDA);

//     torch::Tensor dense_weight = torch::zeros({d_num,d_num},torch::kFloat);
//     torch::Tensor dense_bias = torch::zeros({d_num},torch::kFloat);
//     torch::Tensor layernorm_weight = torch::ones({d_num},torch::kFloat).to(torch::kCUDA);
//     torch::Tensor layernorm_bias = torch::zeros({d_num},torch::kFloat).to(torch::kCUDA);


//     torch::Tensor qkv_weight = torch::zeros({d_num,d_num*3},torch::kFloat);
//     torch::Tensor qkv_bias = torch::zeros({d_num*3},torch::kFloat);

//     generate_array(reinterpret_cast<float*>(qkv_weight.data_ptr()),d_num*d_num*3);
//     generate_array(reinterpret_cast<float*>(qkv_bias.data_ptr()),3*d_num);
//     generate_array(reinterpret_cast<float*>(dense_weight.data_ptr()),d_num*d_num);
//     generate_array(reinterpret_cast<float*>(dense_bias.data_ptr()),d_num);
//     qkv_weight = qkv_weight.to(torch::kCUDA);
//     qkv_bias = qkv_bias.to(torch::kCUDA);
//     dense_weight = dense_weight.to(torch::kCUDA);
//     dense_bias = dense_bias.to(torch::kCUDA);


//     torch::Tensor input_data = torch::zeros({seq_len,d_num},torch::kFloat);
//     generate_array(reinterpret_cast<float*>(input_data.data_ptr()),seq_len*d_num);
//     input_data = input_data.to(torch::kCUDA);

//     sparse_transformers::layers::MultiHeadedAttention multi_headed_attention = sparse_transformers::layers::MultiHeadedAttention(torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),dense_weight,dense_bias,qkv_weight,qkv_bias,layernorm_weight,layernorm_bias,3);

//     torch::Tensor out = torch::zeros({seq_len,d_num},torch::kFloat).to(torch::kCUDA);

//     multi_headed_attention(input_data,torch::zeros(0),"self",out,torch::zeros(0),torch::zeros(0),4096,12,64,64,64,768);

//     // auto qkv_temp = torch::matmul(input_data,qkv_weight);
//     // qkv_temp = qkv_temp + qkv_bias;
//     // auto query = qkv_temp.index({"...",torch::indexing::Slice(0, d_num)}).reshape({64,64,12,64}).permute({2,0,1,3}).transpose(-2,-1).contiguous();
//     // auto key = qkv_temp.index({"...",torch::indexing::Slice(d_num, d_num*2)}).reshape({64,64,12,64}).permute({2,0,1,3}).contiguous();
//     // auto value = qkv_temp.index({"...",torch::indexing::Slice(d_num*2, d_num*3)}).reshape({64,64,12,64}).permute({2,0,1,3}).contiguous();


//     // std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(query.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(q_out.to(at::kCPU).data_ptr()),q_out.numel(),q_out.numel())<<std::endl;
//     // std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(key.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(k_out.to(at::kCPU).data_ptr()),q_out.numel(),q_out.numel())<<std::endl;
//     // std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(value.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(v_out.to(at::kCPU).data_ptr()),q_out.numel(),q_out.numel())<<std::endl;
// }

// void test_bert_intermediate(){
//     torch::Tensor weight = torch::rand({768,3072});
//     torch::Tensor bias = torch::rand({768});
//     sparse_transformers::layers::BertIntermediate(weight,bias);
// }

// void test_bert_output(){
//     int seq_len = 4096, d_num = 768;
//     torch::Tensor hidden_states =torch::zeros({seq_len,d_num},torch::kFloat);
//     torch::Tensor input_data =torch::zeros({seq_len,d_num},torch::kFloat);
//     torch::Tensor out_data =torch::zeros({seq_len,d_num},torch::kFloat);
//     torch::Tensor bias = torch::zeros({d_num},torch::kFloat);
//     torch::Tensor dense_weight = torch::zeros({d_num,d_num},torch::kFloat);
//     torch::Tensor dense_bias = torch::zeros({d_num},torch::kFloat);
//     torch::Tensor layernorm_weight = torch::ones({d_num},torch::kFloat).to(torch::kCUDA);
//     torch::Tensor layernorm_bias = torch::zeros({d_num},torch::kFloat).to(torch::kCUDA);

//     generate_array(reinterpret_cast<float*>(out_data.data_ptr()),seq_len*d_num);
//     generate_array(reinterpret_cast<float*>(input_data.data_ptr()),seq_len*d_num);
//     generate_array(reinterpret_cast<float*>(bias.data_ptr()),d_num);
//     generate_array(reinterpret_cast<float*>(dense_weight.data_ptr()),d_num*d_num);
//     generate_array(reinterpret_cast<float*>(dense_bias.data_ptr()),d_num);
//     generate_array(reinterpret_cast<float*>(hidden_states.data_ptr()),seq_len*d_num);

//     out_data = out_data.to(torch::kCUDA);
//     input_data = input_data.to(torch::kCUDA);
//     bias = bias.to(torch::kCUDA);
//     dense_weight = dense_weight.to(torch::kCUDA);
//     dense_bias = dense_bias.to(torch::kCUDA);
//     hidden_states = hidden_states.to(torch::kCUDA);

//     sparse_transformers::layers::kernels::MatMul(hidden_states, false, dense_weight, false, 1.0,
//                   out_data, 0.0);
//     // std::cout<<hidden_states.to(torch::kCPU).index({torch::indexing::Slice(0,1),torch::indexing::Slice(0,10)})<<std::endl;
//     // std::cout<<dense_weight.to(torch::kCPU).index({torch::indexing::Slice(0,1),torch::indexing::Slice(0,10)})<<std::endl;
//     // std::cout<<out_data.to(torch::kCPU).index({torch::indexing::Slice(0,1),torch::indexing::Slice(0,10)})<<std::endl;

//     sparse_transformers::layers::kernels::test_add_bias_and_layernorm(reinterpret_cast<float*>(out_data.data_ptr()),
//     reinterpret_cast<float*>(input_data.data_ptr()),reinterpret_cast<float*>(dense_bias.data_ptr()),
//     seq_len,2,768,float(1e-5),reinterpret_cast<float*>(layernorm_weight.data_ptr()),
//     reinterpret_cast<float*>(layernorm_bias.data_ptr()));

//     torch::Tensor out = torch::mm(hidden_states,dense_weight);
//     // std::cout<<hidden_states.to(torch::kCPU).index({torch::indexing::Slice(0,1),torch::indexing::Slice(0,10)})<<std::endl;
//     // std::cout<<dense_weight.to(torch::kCPU).index({torch::indexing::Slice(0,1),torch::indexing::Slice(0,10)})<<std::endl;
//     // std::cout<<out.to(torch::kCPU).index({torch::indexing::Slice(0,1),torch::indexing::Slice(0,10)})<<std::endl;
//     out = out + dense_bias;
//     out = out + input_data;

//     torch::nn::LayerNorm layernorm(torch::nn::LayerNormOptions({768}).elementwise_affine(false).eps(1e-5));

//     out = layernorm(out);
//     out = out * layernorm_weight + layernorm_bias;

//     // std::cout<<out<<std::endl;

//     std::cout<<out.to(torch::kCPU).index({torch::indexing::Slice(0,1),torch::indexing::Slice(0,10)})<<std::endl;
//     std::cout<<out_data.to(torch::kCPU).index({torch::indexing::Slice(0,1),torch::indexing::Slice(0,10)})<<std::endl;

    

//     std::cout<<"The result is "<<check_value<float>(reinterpret_cast<float*>(out.to(at::kCPU).data_ptr()),reinterpret_cast<float*>(out_data.to(at::kCPU).data_ptr()),out_data.numel(),out_data.numel())<<std::endl;

// }


#include <chrono>   
using namespace std;
using namespace chrono;

void test_add_bias_act(){
    int seq_len = 4096;
    int d_num = 3072;
    torch::Tensor input = torch::zeros({seq_len,d_num},torch::kFloat32);
    torch::Tensor bias = torch::zeros({d_num},torch::kFloat32);
    torch::Tensor out1 = torch::zeros({seq_len,d_num},torch::kFloat32);
    torch::Tensor out2 = torch::zeros({seq_len,d_num},torch::kFloat32);


    generate_array<float>(reinterpret_cast<float*>(input.data_ptr()),seq_len*d_num);
    generate_array<float>(reinterpret_cast<float*>(bias.data_ptr()),d_num);
    input = input.to(torch::kCUDA);
    bias = bias.to(torch::kCUDA);
    out1 = out1.to(torch::kCUDA);
    out2 = out2.to(torch::kCUDA);

    auto gule = torch::nn::GELU();

    for(int i=0;i<3;i++){
        out1 = input + bias;
        out1 = gule(out1);
    }
    for(int i=0;i<10;i++){
        out1 = input + bias;
        out1 = gule(out1);
    }

    for(int i=0;i<3;i++){
        sparse_transformers::layers::kernels::test_add_bias_act(reinterpret_cast<float*>(bias.data_ptr()),reinterpret_cast<float*>(input.data_ptr()),seq_len,d_num);
    }
    for(int i=0;i<10;i++){
        sparse_transformers::layers::kernels::test_add_bias_act(reinterpret_cast<float*>(bias.data_ptr()),reinterpret_cast<float*>(input.data_ptr()),seq_len,d_num);
    }
}

void test_add_bias_and_transpose(){
    int seq_len = 4096;
    int d_num = 2304;
    torch::Tensor input = torch::zeros({seq_len,d_num},torch::kFloat32);
    torch::Tensor bias = torch::zeros({d_num},torch::kFloat32);
    torch::Tensor out1 = torch::zeros({seq_len,d_num},torch::kFloat32);
    torch::Tensor q = torch::zeros({seq_len,d_num/3},torch::kFloat16);
    torch::Tensor k = torch::zeros({seq_len,d_num/3},torch::kFloat16);
    torch::Tensor v = torch::zeros({seq_len,d_num/3},torch::kFloat16);



    generate_array<float>(reinterpret_cast<float*>(input.data_ptr()),seq_len*d_num);
    generate_array<float>(reinterpret_cast<float*>(bias.data_ptr()),d_num);
    input = input.to(torch::kCUDA).toType(torch::kFloat16);
    bias = bias.to(torch::kCUDA).toType(torch::kFloat16);
    out1 = out1.to(torch::kCUDA).toType(torch::kFloat16);
    q = q.to(torch::kCUDA).toType(torch::kFloat16);
    v = v.to(torch::kCUDA).toType(torch::kFloat16);
    k = k.to(torch::kCUDA).toType(torch::kFloat16);
    

    auto gule = torch::nn::GELU();
    for(int i=0;i<3;i++){
        out1 = input + bias;
        out1 = out1.reshape({64,64,3,12,64}).permute({2,3,0,1,4}).contiguous();
    }
    for(int i=0;i<10;i++){
        out1 = input + bias;
        out1 = out1.reshape({64,64,3,12,64}).permute({2,3,0,1,4}).contiguous();
    }

    // sparse_transformers::layers::


    std::vector<int> seq_len_info = {0,64};
    auto seq_len_info_tensor =  torch::from_blob(seq_len_info.data(),{3},torch::kInt32).to(torch::kCUDA);
    for(int i=0;i<3;i++){
        sparse_transformers::layers::kernels::test_add_bias_and_transpose(reinterpret_cast<half*>(bias.data_ptr()),reinterpret_cast<half*>(input.data_ptr()),reinterpret_cast<half*>(q.data_ptr()),reinterpret_cast<half*>(k.data_ptr()),reinterpret_cast<half*>(v.data_ptr()),0,768,768*2,reinterpret_cast<int*>(seq_len_info_tensor.data_ptr()),1,12,64,64,64);
    }
    for(int i=0;i<10;i++){
        sparse_transformers::layers::kernels::test_add_bias_and_transpose(reinterpret_cast<half*>(bias.data_ptr()),reinterpret_cast<half*>(input.data_ptr()),reinterpret_cast<half*>(q.data_ptr()),reinterpret_cast<half*>(k.data_ptr()),reinterpret_cast<half*>(v.data_ptr()),0,768,768*2,reinterpret_cast<int*>(seq_len_info_tensor.data_ptr()),1,12,64,64,64);
    }

}

void test_add_bias_and_layernorm(){
    int seq_len = 4096;
    int d_num = 768;
    torch::Tensor input = torch::zeros({seq_len,d_num},torch::kFloat32);
    torch::Tensor bias = torch::zeros({d_num},torch::kFloat32);
    torch::Tensor res_bias = torch::zeros({seq_len,d_num},torch::kFloat32);
    torch::Tensor out1 = torch::zeros({seq_len,d_num},torch::kFloat32);
    torch::Tensor out2 = torch::zeros({seq_len,d_num},torch::kFloat32);
    torch::Tensor layer_norm_weight_ = torch::ones({d_num});
    torch::Tensor layer_norm_bias_ = torch::zeros({d_num});


    generate_array<float>(reinterpret_cast<float*>(input.data_ptr()),seq_len*d_num);
    generate_array<float>(reinterpret_cast<float*>(bias.data_ptr()),d_num);
    generate_array<float>(reinterpret_cast<float*>(res_bias.data_ptr()),seq_len*d_num);

    input = input.to(torch::kCUDA).toType(torch::kFloat16);
    bias = bias.to(torch::kCUDA).toType(torch::kFloat16);
    res_bias = res_bias.to(torch::kCUDA).toType(torch::kFloat16);
    out1 = out1.to(torch::kCUDA).toType(torch::kFloat16);
    out2 = out2.to(torch::kCUDA).toType(torch::kFloat16);
    layer_norm_weight_ = layer_norm_weight_.to(torch::kCUDA).toType(torch::kFloat16);
    layer_norm_bias_ = layer_norm_bias_.to(torch::kCUDA).toType(torch::kFloat16);



    torch::nn::LayerNorm layer_norm(torch::nn::LayerNormOptions({768}).elementwise_affine(false).eps(1e-5));
    for(int i=0;i<3;i++){
        out1 = input + bias + res_bias;
        out1 = layer_norm(out1);
    }
    for(int i=0;i<10;i++){
        out1 = input + bias + res_bias;
        out1 = layer_norm(out1);
    }

    std::vector<int> seq_len_info = {0,64};
    auto seq_len_info_tensor =  torch::from_blob(seq_len_info.data(),{3},torch::kInt32).to(torch::kCUDA);
    for(int i=0;i<3;i++){
        sparse_transformers::layers::kernels::test_add_bias_and_layernorm(reinterpret_cast<half*>(input.data_ptr()),reinterpret_cast<half*>(res_bias.data_ptr()),reinterpret_cast<half*>(bias.data_ptr()),seq_len,2,768,float(1e-5),reinterpret_cast<half*>(layer_norm_weight_.data_ptr()),reinterpret_cast<half*>(layer_norm_bias_.data_ptr()));
    }
    for(int i=0;i<10;i++){
        sparse_transformers::layers::kernels::test_add_bias_and_layernorm(reinterpret_cast<half*>(input.data_ptr()),reinterpret_cast<half*>(res_bias.data_ptr()),reinterpret_cast<half*>(bias.data_ptr()),seq_len,2,768,float(1e-5),reinterpret_cast<half*>(layer_norm_weight_.data_ptr()),reinterpret_cast<half*>(layer_norm_bias_.data_ptr()));

    }

}

int main(){
    test_sparse_attention();
    return 0;
}