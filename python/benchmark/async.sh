#!/bin/bash
batch_sizes=(1 2 4 8 16 32)
seqlens=(1024 3840)

models=("raptor_t" "pytorch")

gpu_card=0
export CUDA_VISIBLE_DEVICES=${gpu_card}

source /home/wong/miniconda3/bin/activate transformer
cd /home/wong/TurboTransformers/Attention/python/benchmark

rm ./metrics/async_

echo "model_name batch_size max_seqlen seq_len async total_time(ms) attention_time(ms)" >> ./metrics/async_


for batch_size in ${batch_sizes[@]};do
    for seqlen in ${seqlens[@]};do
        tmp_log=tmp.log
        python ../benchmark.py -s ${seqlen} -m "raptor_t" -b ${batch_size} -a true -be "async_"  2>&1 > $tmp_log

        # python ../async.py -s ${seqlen} -m "raptor_t" -b ${batch_size}  2>&1 > $tmp_log

        cat $tmp_log | grep "raptor_t" >> ./metrics/async_
    done
done

for batch_size in ${batch_sizes[@]};do
    for seqlen in ${seqlens[@]};do
        tmp_log=tmp.log

        python ../benchmark.py -s ${seqlen} -m "raptor_t" -b ${batch_size} -be "async_" -k true -ba true   2>&1 > $tmp_log

        cat $tmp_log | grep "raptor_t" >> ./metrics/async_
    done
done


for batch_size in ${batch_sizes[@]};do
    for seqlen in ${seqlens[@]};do
        tmp_log=tmp.log
        python ../benchmark.py -s ${seqlen} -m "pytorch" -b ${batch_size} -be "async_"  2>&1 > $tmp_log
        cat $tmp_log | grep "pytorch" >> ./metrics/async_
    done
done

rm ./tmp.log
