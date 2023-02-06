#add_executable(kernel_test test.cpp)
add_library(kernels OBJECT
    attention.cu mat_mul.cpp
    )
target_link_libraries(kernels "${TORCH_LIBRARIES}")
# add_library(test_lib OBJECT
#     mat_mul.cpp
#     )

#target_link_libraries(kernel_test kernels)
# set_property(TARGET test PROPERTY CXX_STANDARD 14)


# cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
# # project(attention_layers)
# # set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.3/bin/nvcc")
# # set(CUDA_NVCC_FLAGS -lineinfo;)
# find_package(Torch REQUIRED
#             NO_MODULE
#             PATHS /home/wong/TurboTransformers/libtorch
#             NO_DEFAULT_PATH)
# add_library(kernels OBJECT
#     attention.cu
#     )
# target_link_libraries(kernels "${TORCH_LIBRARIES}")

# add_executable(kernel_test test.cpp)
# target_link_libraries(kernel_test kernels)
# set_property(TARGET kernel_test PROPERTY CXX_STANDARD 14)