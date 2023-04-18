import torch
import heapq
import operator
import random
import numpy as np
from functools import reduce

device = torch.device('cuda')


def generate_parition_plan(block_num_array:list,seq_start_index:list,thread_block_num:int,head_num:int):
        sorted_nums = sorted(enumerate(block_num_array), key=lambda x: x[1],reverse=True)
        index = [i[0] for i in sorted_nums]
        sorted_value = [i[1] for i in sorted_nums]
        partition_part = [[] for i in range(0,thread_block_num)]
        min_heap = [(0,i) for i in range(0,thread_block_num)]
        for i in range(0,len(sorted_value)):
            for j in range(0,head_num):
                pop_element = heapq.heappop(min_heap)
                start_index = (0,0)
                for k in seq_start_index:
                    if(index[i] >= k[0]):
                        start_index = k
                    else:
                        break
                partition_part[pop_element[1]].append(index[i] - start_index[0] +j*start_index[1]+start_index[0]*head_num)
                pop_element = (pop_element[0]+sorted_value[i],pop_element[1])
                heapq.heappush(min_heap,pop_element)
        partition_part_index = [len(it) for it in partition_part]
        partition_part_index = [sum(partition_part_index[0:i]) for i in range(0,len(partition_part_index)+1)]
        partition_part_index_tensor = torch.tensor(partition_part_index,dtype=torch.int,device="cuda")
        partition_part = reduce(operator.add, partition_part)
        partition_part_tensor = torch.tensor(partition_part,dtype=torch.int,device="cuda")
        return (partition_part_index_tensor,partition_part_tensor)

def generate_data_metainfo(input_datas:list, block_size:int, thread_block_limit:int):
    seq_position_info = [int(len(seq)/block_size) for seq in input_datas]
    block_num_array = []
    for i in seq_position_info:
        block_num_array.append(i)
        block_num_array.append(7)
        block_num_array.extend([8 for i in range(i-4)])
        block_num_array.append(7)
        block_num_array.append(i)
    seq_position_info = [sum(seq_position_info[0:i]) for i in range(len(seq_position_info)+1)]
    total_comp_block_num = sum(block_num_array)
    seq_start_index = [(seq_position_info[i],seq_position_info[i+1] - seq_position_info[i])for i in range(len(seq_position_info)- 1)]
    (partition_part_index_tensor,partition_part_tensor) = generate_parition_plan(block_num_array,seq_start_index,thread_block_limit,12)
    seq_position_info_tensor = torch.tensor(seq_position_info,dtype=torch.int,device="cuda")
    return (seq_position_info,seq_position_info_tensor,total_comp_block_num,partition_part_index_tensor,partition_part_tensor)



def pre_process_input_data(queue1,block_size,output_layer_temp:torch.Tensor,intermediate_temp,attention_output):
    if queue1.empty():
        return (None,None,None,None, None, None,None,None,None,None)
    input_data = queue1.get()
    position_ids = [np.arange(len(it)).tolist() for it in input_data]
    position_ids = reduce(operator.add, position_ids)
    seq_position_info,seq_position_info_tensor,total_comp_block_num,partition_part_index_tensor,partition_part_tensor = generate_data_metainfo(input_data,64,160)
    total_seq_len = seq_position_info[-1]*block_size
    input_data = reduce(operator.add, input_data)
    hidden_states = torch.tensor(input_data,dtype=torch.int,device = device)
    token_type_ids = torch.zeros_like(hidden_states)
    position_ids = torch.tensor(position_ids,dtype=torch.int,device = device)
    attention_masks = torch.zeros_like(hidden_states)
    torch.cuda.nvtx.range_push("11111")
    output_layer_temp = output_layer_temp.resize_(total_seq_len,768)
    torch.cuda.nvtx.range_pop()

    
    intermediate_temp = intermediate_temp.resize_(total_seq_len,3072)
    attention_output = attention_output.resize_(total_seq_len,768)
    # queue2.put((hidden_states,total_seq_len,seq_position_info))
    return (hidden_states,total_seq_len,seq_position_info,seq_position_info_tensor, partition_part_index_tensor, partition_part_tensor,attention_masks,token_type_ids,position_ids,total_comp_block_num)


def generate_array_with_avg(seq_len, batch_size,max_len):
    # 生成一个随机数组，数组长度为 length
    seq_lens = [random.randint(768, max_len+1) for i in range(batch_size)]
    
    # 计算数组的当前平均值
    cur_avg = sum(seq_lens) / batch_size
    
    # 根据给定的平均值和当前平均值的差，对数组中的元素进行加/减操作
    for i in range(batch_size):
        diff = seq_len - cur_avg
        if seq_lens[i] + diff > max_len:
            diff = seq_lens[i] - max_len
        elif seq_lens[i] + diff < 768:
            diff = seq_lens[i] - 768
        seq_lens[i] += diff
        cur_avg = sum(seq_lens) / batch_size
    
    diff_sum = (seq_len - cur_avg)*batch_size
    if diff_sum < 0:
        seq_lens.sort()
    else:
        seq_lens.sort(reverse=True)
    for i in range(batch_size):
        diff = diff_sum/(batch_size-i)
        if seq_lens[i] + diff > max_len:
            diff = max_len - seq_lens[i]
        elif seq_lens[i] + diff < 768:
            diff = 768 - seq_lens[i]
        seq_lens[i] += diff
        diff_sum -= diff
    result = []
    for val in seq_lens:
        nearest_multiple = round(val/64)
        result.append(nearest_multiple*64)
    return result

def generate_input_data(batch_size:int,seq_len_start:int,seq_len_end:int,block_size:int, seq_lens):
    result = []
    # seq_lens = [1024,4096,3072,2048,2048,3072,1024,3072,2048,2048]

    for i in range(batch_size):
        seq_len = random.randint(seq_len_start,seq_len_end)
        seq_len = seq_len - seq_len%block_size
        seq_len = seq_lens[i]
        result.append(np.random.randint(0,30000,(seq_len)).tolist())
    return result