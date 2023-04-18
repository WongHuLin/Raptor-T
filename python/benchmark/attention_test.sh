#!/bin/bash
batch_sizes=(1 2 4 8 16 32)
seqlens=(1024 3840)
balanceds=(true false)

models=("raptor_t" "pytorch")

gpu_card=0
export CUDA_VISIBLE_DEVICES=${gpu_card}

source /home/wong/miniconda3/bin/activate transformer
cd /home/wong/TurboTransformers/Attention/python/benchmark

rm ./metrics/attention_test

echo "model_name batch_size max_seqlen seq_len balanced total_time(ms) attention_time(ms)" >> ./metrics/attention_test


for batch_size in ${batch_sizes[@]};do
    for seqlen in ${seqlens[@]};do
        tmp_log=tmp.log

        python ../benchmark.py -s ${seqlen} -m "raptor_t" -b ${batch_size} -be "attention_test"  2>&1 > $tmp_log
        cat $tmp_log | grep "raptor_t" >> ./metrics/attention_test
    done
done


for batch_size in ${batch_sizes[@]};do
    for seqlen in ${seqlens[@]};do
        tmp_log=tmp.log
        python ../benchmark.py -s ${seqlen} -m "pytorch" -b ${batch_size} -be "attention_test"  2>&1 > $tmp_log
        cat $tmp_log | grep "pytorch" >> ./metrics/attention_test
    done
done

rm ./tmp.log
