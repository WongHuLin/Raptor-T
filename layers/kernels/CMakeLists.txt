# #add_executable(kernel_test test.cpp)
# add_library(kernels OBJECT
#     add_bias_act.cu
#     add_bias_and_layernorm.cu
#     add_bias_and_transpose.cu
#     sparse_attention_new.cu
#     sparse_attention.cu
#     mat_mul.cpp
#     )
# target_link_libraries(kernels "${TORCH_LIBRARIES}")
# # add_library(test_lib OBJECT
# #     mat_mul.cpp
# #     )

# #target_link_libraries(kernel_test kernels)
# # set_property(TARGET test PROPERTY CXX_STANDARD 14)


cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_COMMON_GPU_ARCHITECTURES "8.0" "8.6")
set(CUDA_ALL_GPU_ARCHITECTURES "8.0" "8.6")

# MESSAGE( STATUS "this var key = ${CUDA_ALL_GPU_ARCHITECTURES}.")


# # project(attention_layers)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")
set(CUDA_NVCC_FLAGS -lineinfo;)
find_package(Torch REQUIRED
            NO_MODULE
            PATHS /home/wong/test/Raptor-T/libtorch/
            NO_DEFAULT_PATH)


set(CUDA_COMMON_GPU_ARCHITECTURES "8.0" "8.6")
set(CUDA_ALL_GPU_ARCHITECTURES "8.0" "8.6")

MESSAGE( STATUS "this var key = ${CUDA_ALL_GPU_ARCHITECTURES}.")



# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.1)
# project(attention_layers)
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")

# set(CMAKE_CUDA_ARCHITECTURES 80)

# set(CUDA_ARCHITECTURES 80)
# set(CMAKE_CUDA_ARCHITECTURES 80)



string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_80,code=sm_80")

add_library(kernels OBJECT
    add_bias_act.cu
    add_bias_and_layernorm.cu
    add_bias_and_transpose.cu
    sparse_attention.cu
    pipeline_mha.cu
    test.cu
    fine_grain_kernel.cu
    )
target_link_libraries(kernels "${TORCH_LIBRARIES}")

add_executable(kernel_test test.cpp)
target_link_libraries(kernel_test kernels)
set_property(TARGET kernel_test PROPERTY CXX_STANDARD 17)