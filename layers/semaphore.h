#pragma once
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <map>
namespace sparse_transformers {
namespace layers {
class Semaphore {
   public:
    typedef std::shared_ptr<Semaphore> Ptr;
    static Ptr m_instance_ptr;
    static std::mutex m_mutex;
    std::map<int,int> layer_cond;
    bool terminate_single;

    bool get_terminate_single();

    void Signal();

    void Wait(int layer_idx);

    void register_layer(int layer_idx);

    static Ptr get_instance(){
        // "double checked lock"
        if(m_instance_ptr==nullptr){
            std::lock_guard<std::mutex> lk(m_mutex);
            if(m_instance_ptr == nullptr){
              m_instance_ptr = std::shared_ptr<Semaphore>(new Semaphore());
            }
        }
        return m_instance_ptr;
    }

   private:
    std::mutex mutex_;
    std::condition_variable cv_;
    explicit Semaphore() {
        terminate_single = false;
    }
};
}
}