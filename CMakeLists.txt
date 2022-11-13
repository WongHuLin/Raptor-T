cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(libtorch_test)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.3/bin/nvcc")
set(CUDA_NVCC_FLAGS -lineinfo;)
find_package(Torch REQUIRED
            NO_MODULE
            PATHS /home/wong/TurboTransformers/libtorch
            NO_DEFAULT_PATH)
add_library(test_lib OBJECT
    attention_2.cu
    )
add_executable(test test.cpp)
target_link_libraries(test "${TORCH_LIBRARIES}")
target_link_libraries(test test_lib)
set_property(TARGET test PROPERTY CXX_STANDARD 14)