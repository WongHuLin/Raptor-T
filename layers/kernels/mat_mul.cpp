
#include "mat_mul.h"

// #define ENFORCE_EQ(a, b, ...) TT_ENFORCE((a) == (b), __VA_ARGS__)

namespace sparse_transformers {
namespace layers {
namespace kernels {
    void MatMul(const torch::Tensor& A, bool a_trans, const torch::Tensor& B,
        bool b_trans, float alpha, torch::Tensor& out, float beta,const cublasHandle_t headle_, const std::string name)
    {
        int a_cols = A.sizes()[A.sizes().size() - 1];
        int a_rows = A.numel() / a_cols;
        int b_cols = B.sizes()[B.sizes().size() - 1];
        int b_rows = B.numel() / b_cols;

        // printf("%d %d %d %d\n",a_cols,a_rows,b_cols,b_rows);


        
        int M = a_trans ? a_cols : a_rows;
        int N = b_trans ? b_rows : b_cols;
        
        int K_a = a_trans ? a_rows : a_cols;
        int K_b = b_trans ? b_cols : b_rows;

        // ENFORCE_EQ(K_a, K_b, "matrix shape mismatch %d vs %d", K_a, K_b);

        if(A.device() != B.device())
            printf("MatMul error: the device of A and B is different.\n");
        if(A.device() != out.device())
            printf("MatMul error: the device of A and out is different.\n");

        // if(A.device() == torch::kCUDA && B.device() == torch::kCUDA && out.device() == torch::kCUDA){
        if(true){
            cublasOperation_t transA = a_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t transB = b_trans ? CUBLAS_OP_T : CUBLAS_OP_N;

            int lda = (transA == CUBLAS_OP_N) ? K_a : M;
            int ldb = (transB == CUBLAS_OP_N) ? N : K_a;
            int ldc = N;

#if defined(WITH_TENSOR_CORE)      
            auto cublas_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
            auto math_algo = CUBLAS_TENSOR_OP_MATH;
#else
            auto cublas_algo = CUBLAS_GEMM_DEFAULT;
            auto math_algo = CUBLAS_DEFAULT_MATH;
#endif         
            if(cublasSetMathMode(headle_, math_algo) != CUBLAS_STATUS_SUCCESS)
            {
                printf("1111\n");
                printf("CUDA runtime error\n");
                exit(0);
            }

            if(A.dtype() == torch::kFloat){
                auto result = cublasGemmEx(headle_, transB, transA, N, M, K_a, &alpha,
                reinterpret_cast<float*>(B.data_ptr()), CUDA_R_32F, ldb, reinterpret_cast<float*>(A.data_ptr()), CUDA_R_32F, lda,&beta, reinterpret_cast<float*>(out.data_ptr()), CUDA_R_32F, ldc, CUDA_R_32F, cublas_algo);
                if (result != CUBLAS_STATUS_SUCCESS){
                    std::cout<<result<<std::endl;
                    std::cout<<A.sizes()<<std::endl;
                    std::cout<<B.sizes()<<std::endl;
                    std::cout<<out.sizes()<<std::endl;
                    std::cout<<A.dtype()<<std::endl;
                    std::cout<<B.dtype()<<std::endl;
                    std::cout<<out.dtype()<<std::endl;
                    printf("CUDA runtime error\n");
                    exit(0);
                }
            }
            else if(A.dtype() == torch::kHalf){
                half alpha_h = __float2half(alpha);
                half beta_h = __float2half(beta);

                auto result = cublasGemmEx(headle_, transB, transA, N, M, K_a, &alpha_h,reinterpret_cast<half*>(B.data_ptr()), CUDA_R_16F, ldb, reinterpret_cast<half*>(A.data_ptr()), CUDA_R_16F, lda,&beta_h, reinterpret_cast<half*>(out.data_ptr()), CUDA_R_16F, ldc, CUBLAS_COMPUTE_16F, cublas_algo);
                if (result != CUBLAS_STATUS_SUCCESS){
                    std::cout<<result<<std::endl;
                    std::cout<<name<<std::endl;
                    std::cout<<A.sizes()<<std::endl;
                    std::cout<<B.sizes()<<std::endl;
                    std::cout<<out.sizes()<<std::endl;
                    std::cout<<A.dtype()<<std::endl;
                    std::cout<<B.dtype()<<std::endl;
                    std::cout<<out.dtype()<<std::endl;
                    printf("CUDA runtime error\n");
                    exit(0);
                }
            }

            if(cublasSetMathMode(headle_, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS){
                printf("CUDA runtime error\n");
                exit(0);
            }
        }
    }
}
}
}