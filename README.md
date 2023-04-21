
### 
We run all experiments on NVIDIA A100 GPUs. We use nvidia-docker to run the container on the GPU. Reproducing the artifact is divided into the following 3 main steps.

Step 1: Download Source code and run the container.
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

Step 2: Deploy FlashAttention, FastTransformer and Pytorch

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

Step 3: Run the experiments.

bash /workspace/python/benchmark/end2end.sh $\rightarrow$ Figs~\ref{fig:end2end_time}-\ref{fig:end2end_memory}

bash /workspace/python/benchmark/mem\_bound\_op.sh $\rightarrow$ Fig~\ref{fig:mem_bound_op}

bash /workspace/python/benchmark/async\_.sh $\rightarrow$ Fig~\ref{fig:asyc_generation}

bash /workspace/python/benchmark/attention\_test.sh $\rightarrow$ Fig~\ref{fig:attn}

bash /workspace/python/benchmark/CTAs\_num\_test.sh $\rightarrow$ Fig~\ref{fig:balanced_compute}
