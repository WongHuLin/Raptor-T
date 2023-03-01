from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='lltm_cpp',
      ext_modules=[CUDAExtension('lltm_cpp', [
            'pybind.cpp','../layers/bert_attention.cpp',
            '../layers/bert_intermediate.cpp',
            '../layers/bert_output.cpp',
            '../layers/bert_pooler.cpp',
            '../layers/multi_headed_attention.cpp',
            '../layers/kernels/mat_mul.cpp',
            '../layers/kernels/add_bias_act.cu',
            '../layers/kernels/add_bias_and_layernorm.cu',
            '../layers/kernels/add_bias_and_transpose.cu',
            '../layers/kernels/sparse_attention.cu',
            '../layers/tensor_set.cpp',
            '../layers/metadata.cpp',
            '../layers/semaphore.cpp'])],
      cmdclass={'build_ext': BuildExtension})