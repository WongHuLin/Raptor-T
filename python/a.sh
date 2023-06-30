#!/bin/bash
cd /home/wong/test/Raptor-T/python/benchmark
source /home/wong/miniconda3/bin/activate cuda12
# /home/wong/miniconda3/bin/conda activate transformer
CUDA_VISIBLE_DEVICES=2
python ../benchmark.py -s 4096 -m "raptor_t" -b 1 -be "attention_test"