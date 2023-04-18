FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel
RUN  apt update \
    && apt install -y --no-install-recommends wget git \
    && wget https://cmake.org/files/v3.21/cmake-3.21.0.tar.gz  \
    && tar -zxvf cmake-3.21.0.tar.gz -C  /usr/local/  \
    && ln -sf /usr/local/cmake-3.21.0-linux-x86_64/bin/* /usr/bin   \
    && rm cmake-3.21.0.tar.gz  \
    && rm -rf /var/lib/apt/lists/* \ 
    && pip install packaging \
    && pip install einops==0.6.0 numpy==1.21.3 nvidia_ml_py3==7.352.0 setuptools==67.6.0 transformers==4.27.3 flash_attn==1.0.1 