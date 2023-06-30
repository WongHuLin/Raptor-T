#pragma once
#include <torch/torch.h>
#include <chrono>  
#include <map>
#include <string>
#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <numeric>
#include <cooperative_groups/memcpy_async.h>
#include <thrust/extrema.h>
#include <mma.h>
#include <cub/cub.cuh>
// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>
// Optionally include for memcpy_async() collective
#include <cooperative_groups/memcpy_async.h>
// Optionally include for reduce() collective
#include <cooperative_groups/reduce.h>
// Optionally include for inclusive_scan() and exclusive_scan() collectives
#include <cuda/barrier>


using namespace std::chrono;

namespace sparse_transformers {
namespace layers {
namespace kernels {
__global__ void sparse_attention_test(half *a,  half *b,  half *c, 
    half *out, const int *seq_len_info,const int *from_block_index, 
    const int *from_block_index_position, const int *to_select_index,
    const int *to_select_index_position, const int batch_size,
    const int block_size, const int head_size);
}
}
}