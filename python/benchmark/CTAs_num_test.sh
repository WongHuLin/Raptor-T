#!/bin/bash
batch_sizes=(1 2 4 8 16 32)
seqlens=(1024 1280 1536 1792 2048 2304 2560 2816 3072 3328 3584 3840)
thread_blocks=(160 240 320 400 480 560 640 720 800 880)


models=("raptor_t")

gpu_card=0
export CUDA_VISIBLE_DEVICES=${gpu_card}

source /home/wong/miniconda3/bin/activate transformer
cd /home/wong/TurboTransformers/Attention/python/benchmark

rm ./metrics/CTAs_num_test

echo "model_name thread_block batch_size max_seqlen seq_len total_time(ms)" >> ./metrics/CTAs_num_test


for thread_block in ${thread_blocks[@]};do
    for batch_size in ${batch_sizes[@]};do
        for seqlen in ${seqlens[@]};do
            tmp_log=tmp.log
            python ../benchmark.py -s ${seqlen} -m "raptor_t" -b ${batch_size} -t ${thread_block} -be "CTAs_num_test"  2>&1 > $tmp_log
            cat $tmp_log | grep "raptor_t" >> ./metrics/CTAs_num_test
        done
    done
done

rm ./tmp.log
