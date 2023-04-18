#!/bin/bash
batch_sizes=(1 2 4 8 16 32)
seqlens=(1024 3840)

models=("raptor_t")
kernel_fusions=(true false)

gpu_card=0
export CUDA_VISIBLE_DEVICES=${gpu_card}

source /home/wong/miniconda3/bin/activate transformer
cd /home/wong/TurboTransformers/Attention/python/benchmark

rm ./metrics/kernel_fusion_mem_op

echo "model_name batch_size max_seqlen seq_len kernel_fusion total_time(ms) mem_op_time(ms)" >> ./metrics/kernel_fusion_mem_op

for model_name in ${models[@]};do
    for batch_size in ${batch_sizes[@]};do
        for seqlen in ${seqlens[@]};do
            for kernel_fusion in ${kernel_fusions[@]};do
                tmp_log=tmp.log
                if ${kernel_fusion};then
                    python ../benchmark.py -s ${seqlen} -m ${model_name} -b ${batch_size} -k true -be "kernel_fusion_mem_op" 2>&1 > $tmp_log
                    cat $tmp_log | grep "${model_name}" >> ./metrics/kernel_fusion_mem_op
                else
                    python ../benchmark.py -s ${seqlen} -m ${model_name} -b ${batch_size} -be "kernel_fusion_mem_op" 2>&1 > $tmp_log
                    cat $tmp_log | grep "${model_name}" >> ./metrics/kernel_fusion_mem_op
                fi
            done
        done
    done
done

rm ./tmp.log
