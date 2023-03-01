#include "semaphore.h"
namespace sparse_transformers {
namespace layers {

    Semaphore::Ptr Semaphore::m_instance_ptr = nullptr;
    std::mutex Semaphore::m_mutex;

    void Semaphore::register_layer(int layer_idx){
        if(layer_cond.find(layer_idx) != layer_cond.end())
        {
            layer_cond[layer_idx] = 0;
            std::cout<<layer_idx<<" register success"<<std::endl;
        }
    }

    void Semaphore::Wait(int layer_idx) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [=] { return layer_cond[layer_idx] > 0; });
        layer_cond[layer_idx] = layer_cond[layer_idx] - 1;
    }

    bool Semaphore::get_terminate_single(){
        return terminate_single;    
    }

    void Semaphore::Signal() {
        std::unique_lock<std::mutex> lock(mutex_);
        for(auto it:layer_cond){
            layer_cond[it.first] = it.second + 1;
        }
        // std::map<int, int>::iterator iter = layer_cond.begin();
        // while(iter != layer_cond.end()) {
        //     std::cout << iter->first << " : " << iter->second << std::endl;
        //     iter++;
        // }
        cv_.notify_all();
    }
}
}