#pragma once
#include "tensor_set.h"
#include "semaphore.h"
namespace sparse_transformers {
namespace layers {
class MetaData
{
private:
    TensorSet::Ptr tensor_set;
    Semaphore::Ptr semaphore;

public:
    void terminate_thread();

    MetaData(){
        tensor_set = TensorSet::get_instance();
        semaphore = Semaphore::get_instance();
    }
    void update_meta_data(int total_seq_len, int to_select_index_len, int to_select_index_position_len,std::vector<int> seq_len_info);
    ~MetaData(){}
};
}
}