# !/bin/bash
batch_sizes=(1 2 4 8 16 32)
seqlens=(1024 1536 2048 2560 3072 3584)

models=("raptor_t" "flash_attn" "pytorch" "fasttransformer" "bert_like")

# batch_sizes=(1 2 4 8 16 32)
# seqlens=(2048)

# models=("raptor_t" "flash_attn")

gpu_card=2
export CUDA_VISIBLE_DEVICES=${gpu_card}



source /home/wong/miniconda3/bin/activate transformer
cd /home/wong/test/Raptor-T/python/benchmark

rm ./metrics/end2end

echo "model_name batch_size max_seqlen seq_len total_time(ms) memory(MB)" >> ./metrics/end2end

for model_name in ${models[@]};do
    for batch_size in ${batch_sizes[@]};do
        for seqlen in ${seqlens[@]};do
            tmp_log=tmp.log
            echo "start" >> ./tmp.log
            python ../benchmark.py -s ${seqlen} -m ${model_name} -b ${batch_size} -be "end2end" -g ${gpu_card}  2>&1 > $tmp_log
            cat $tmp_log | grep "${model_name}" >> ./metrics/end2end
        done
    done
done

rm ./tmp.log
