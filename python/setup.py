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
            '../layers/kernels/attention.cu'])],
      cmdclass={'build_ext': BuildExtension})