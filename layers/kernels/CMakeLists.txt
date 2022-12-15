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