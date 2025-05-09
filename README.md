# Raptor-t: A fused and memory-efficient sparse transformer for long and variable-length sequences

This repo is for TC 2024 artifacts evaluation.

### 
We run all experiments on NVIDIA A100 GPUs. We use nvidia-docker to run the container on the GPU. Reproducing the artifact is divided into the following 3 main steps.

**Step 1: Download Source code and run the container.**
```bash
git clone --recursive https://github.com/WongHuLin/Raptor-T.git
cd Raptor-T
apt-get update && apt-get install git-lfs && git lfs install
mkdir models && cd models
git clone https://huggingface.co/allenai/longformer-base-4096 ./longformer  
&&  git clone  https://huggingface.co/google/bigbird-roberta-base ./bigbird  
&&  cd ..
docker build -t raptor_t:v1.0 .
nvidia-docker run -it -v $PWD:/workspace raptor_t:v1.0 /bin/bash
```

**Step 2: Deploy FlashAttention, FastTransformer, Pytorch and Raptor-T**

FlashAttention
```bash
pip install flash_attn==1.0.1 
```

FashTransformer
```bash
cd /workspace/Raptor-T/FastTransformer 
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON ..
make -j12
```

Raptor-T
```bash
cd /workspace/Raptor-T/python
python setup.py install
```

Pytorch
```
Replace the modeling_big_bird.py file in the Huggingface library with the modeling_big_bird.py in the Raptor-T directory.
```

**Step 3: Run the experiments.**

bash /workspace/python/benchmark/end2end.sh $\rightarrow$ Figs 9-10

bash /workspace/python/benchmark/mem\_bound\_op.sh $\rightarrow$ Fig 11

bash /workspace/python/benchmark/async\_.sh $\rightarrow$ Fig 12

bash /workspace/python/benchmark/attention\_test.sh $\rightarrow$ Fig 13

bash /workspace/python/benchmark/CTAs\_num\_test.sh $\rightarrow$ Fig 14



###  Citation
If you use Eco-Rec in your research, please consider citing our paper:
```bash
@article{wang2024raptor,
  title={Raptor-t: A fused and memory-efficient sparse transformer for long and variable-length sequences},
  author={Wang, Hulin and Yang, Donglin and Xia, Yaqi and Zhang, Zheng and Wang, Qigang and Fan, Jianping and Zhou, Xiaobo and Cheng, Dazhao},
  journal={IEEE Transactions on Computers},
  year={2024},
  publisher={IEEE}
}


```
