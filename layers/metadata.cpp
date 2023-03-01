#include "metadata.h"

namespace sparse_transformers {
namespace layers {

    void MetaData::update_meta_data(int total_seq_len, int to_select_index_len, int to_select_index_position_len,std::vector<int> seq_len_info){
        tensor_set->update_tensor_set(total_seq_len,to_select_index_len,to_select_index_position_len,seq_len_info);
        semaphore->Signal();
        // std::cout<<"update success"<<std::endl;
        // std::cout<<tensor_set->seq_len_info_<<std::endl;
        // std::cout<<tensor_set->get_tensor("tmp_qkv_out1").sizes() <<std::endl;
        // std::cout<<tensor_set->to_select_index_position_len_ <<std::endl;
    }

    void MetaData::terminate_thread(){
        semaphore->terminate_single = true;
        semaphore->Signal();
    }

}
}