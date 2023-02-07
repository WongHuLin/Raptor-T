#include <torch/extension.h>
#include "../layers/bert_attention.h"
#include "../layers/bert_intermediate.h"
#include "../layers/bert_output.h"
#include "../layers/bert_pooler.h"


namespace sparse_transformers {
namespace python {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    pybind11::class_<layers::BertAttention>(m,"BertAttention")
        .def(pybind11::init([](torch::Tensor &qkv_weight, torch::Tensor &qkv_bias, torch::Tensor &dense_weight, torch::Tensor &dense_bias,
                torch::Tensor &layer_norm_weight, torch::Tensor &layer_norm_bias,
                int64_t num_attention_heads) -> layers::BertAttention * {
                    return new layers::BertAttention(
                        std::move(qkv_weight), std::move(qkv_bias), std::move(dense_weight),
                        std::move(dense_bias), std::move(layer_norm_weight),
                        std::move(layer_norm_bias), num_attention_heads);
        }))
        .def("__call__", &layers::BertAttention::operator());
    
    pybind11::class_<layers::MultiHeadedAttention>(m,"MultiHeadedAttention")
        .def(pybind11::init([](torch::Tensor &k_weight, torch::Tensor &k_bias,
        torch::Tensor &v_weight, torch::Tensor &v_bias,
        torch::Tensor &q_weight, torch::Tensor &q_bias, torch::Tensor &dense_weight, torch::Tensor &dense_bias,
                torch::Tensor &qkv_weight, torch::Tensor &qkv_bias,
                int64_t num_attention_heads) -> layers::MultiHeadedAttention * {
                    return new layers::MultiHeadedAttention(
                        std::move(k_weight), std::move(k_bias),
                        std::move(v_weight), std::move(v_bias),
                        std::move(q_weight), std::move(q_bias),
                        std::move(dense_weight),std::move(dense_bias),
                        std::move(qkv_weight), std::move(qkv_bias),
                        num_attention_heads);
        }))
        .def("__call__", &layers::MultiHeadedAttention::operator());

    pybind11::class_<layers::BertIntermediate>(m,"BertIntermediate")
        .def(pybind11::init([](torch::Tensor &dense_weight, 
        torch::Tensor &dense_bias) -> layers::BertIntermediate * {
                    return new layers::BertIntermediate(
                        std::move(dense_weight), std::move(dense_bias));
        }))
        .def("__call__", &layers::BertIntermediate::operator());

    pybind11::class_<layers::BertOutput>(m,"BertOutput")
        .def(pybind11::init([](torch::Tensor &dense_weight, 
        torch::Tensor &dense_bias,torch::Tensor &layer_norm_weight, 
        torch::Tensor &layer_norm_bias) -> layers::BertOutput * {
                    return new layers::BertOutput(
                        std::move(dense_weight), std::move(dense_bias),
                        std::move(layer_norm_weight), std::move(layer_norm_bias));
        }))
        .def("__call__", &layers::BertOutput::operator());

    pybind11::class_<layers::BertPooler>(m,"BertPooler")
        .def(pybind11::init([](torch::Tensor &dense_weight, 
        torch::Tensor &dense_bias) -> layers::BertPooler * {
                    return new layers::BertPooler(
                        std::move(dense_weight), std::move(dense_bias));
        }))
        .def("__call__", &layers::BertPooler::operator());
}
}
}
