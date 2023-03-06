cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(attention_layers)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.3/bin/nvcc")
set(CUDA_NVCC_FLAGS -lineinfo;)
find_package(Torch REQUIRED
            NO_MODULE
            PATHS /home/wong/TurboTransformers/libtorch
            NO_DEFAULT_PATH)
add_subdirectory(kernels)
# add_library(layers OBJECT multi_headed_attention.cpp tensor_set.cpp semaphore.cpp tensor_set.cpp metadata.cpp)
# target_link_libraries(layers PUBLIC "${TORCH_LIBRARIES}" kernels)
# add_executable(layers_test test.cpp)
# target_link_libraries(layers_test layers kernels)
# add_library(layers OBJECT multi_headed_attention.cpp tensor_set.cpp semaphore.cpp tensor_set.cpp metadata.cpp)
# target_link_libraries(layers PUBLIC "${TORCH_LIBRARIES}" kernels)
add_executable(layers_test test.cpp)
target_link_libraries(layers_test kernels)