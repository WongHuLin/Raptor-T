#!/bin/bash
CUDA_VISIBLE_DEVICES=1
cd /home/wong/TurboTransformers/Attention/python
source /home/wong/miniconda3/bin/activate transformer
# /home/wong/miniconda3/bin/conda activate transformer

python ./async.py -s 3840 -m "raptor_t" -b 4 -a true