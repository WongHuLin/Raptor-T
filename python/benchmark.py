import torch
import numpy as np
import random
import heapq
from functools import reduce
import operator
import time

import lltm_cpp as cxx
from modeling_bert import BertIntermediate,BertOutput,BertAttention,BertLayer,BertEncoder,BertModel,BertModelNoPooler
from transformers import BertModel,BigBirdModel
import os
import torch.cuda.nvtx as nvtx

import torch
from transformers import BigBirdConfig,BigBirdModel
from flash_attn.models.bert import BertModel, BertForPreTraining
from flash_attn.models.bert import remap_state_dict
from flash_attn.utils.pretrained import state_dict_from_pretrained

def generate_input_data(batch_size:int,seq_len_start:int,seq_len_end:int,block_size:int):
    result = []
    seq_lens = [1024,4096,3072,2048,2048,3072,1024,3072,2048,2048]
    for i in range(batch_size):
        seq_len = random.randint(seq_len_start,seq_len_end)
        seq_len = seq_len - seq_len%block_size
        seq_len = seq_lens[i]
        result.append(np.random.randint(0,30000,(seq_len)).tolist())
    return result

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
        print(min_heap)
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

# input_data = generate_input_data(5,1024,4096,block_size)
# position_ids = [np.arange(len(it)).tolist() for it in input_data]
# position_ids = reduce(operator.add, position_ids)
# seq_position_info,seq_position_info_tensor,total_comp_block_num,partition_part_index_tensor,partition_part_tensor = generate_data_metainfo(input_data,64,160)
# total_seq_len = seq_position_info[-1]*block_size
# input_data = reduce(operator.add, input_data)



# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


device = torch.device('cuda')


def generate_batch(queue,batch_num,block_size):
    for i in range(batch_num):
        input_data = generate_input_data(10,3077,3090,block_size)
        queue.put(input_data)
        # print("generate data {}".format(i))

def pre_process_input_data(queue1,block_size,output_layer_temp:torch.Tensor,intermediate_temp,attention_output):
    if queue1.empty():
        return (None,None,None,None, None, None,None,None,None,None)
    # print("get input_data from queue queue_size {}".format(queue1.qsize()))
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


def handle_data(model:BertModelNoPooler,queue,event):

    metadata.terminate_thread()
    print("t3 end")
    event.set()

def new_stream():
    return torch.cuda.Stream(torch.device("cuda"))


def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.5)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x

# cuda_device = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
# occumpy_mem(cuda_device)



import multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn')
    batch_num = 10
    block_size = 64


    queue1 = mp.Queue(10)
    queue2 = mp.Queue(10)
    e = mp.Event()

    t1 = mp.Process(target=generate_batch,args=(queue1,batch_num,64))
    # t2 = mp.Process(target=pre_process_input_data,args=(queue1,queue2,e))

    # t1.start()
    n_buffer = 2
    pre_process_stream = new_stream()
    comp_stream = new_stream()
    pre_process_end_event =  [torch.cuda.Event() for i in range(n_buffer)]
    comp_end_event = [torch.cuda.Event() for i in range(n_buffer)]




    # hidden_states = [None,None]
    # total_seq_len = [None,None]
    # seq_position_info = [None,None]
    # seq_position_info_tensor = [None,None]
    # partition_part_index_tensor = [None,None]
    # partition_part_tensor = [None,None]
    # attention_masks = [None,None]
    # token_type_ids = [None,None]
    # position_ids = [None,None]
    # total_comp_block_num = [None,None]
    # output_layer_temp = [torch.tensor(0,dtype=float,device=device),torch.tensor(0,dtype=float,device=device)]
    # intermediate_temp = [torch.tensor(0,dtype=float,device=device),torch.tensor(0,dtype=float,device=device)]
    # attention_output = [torch.tensor/home/wong/TurboTransformers/Attention/python/build(0,dtype=float,device=device),torch.tensor(0,dtype=float,device=device)]


    # end_flag = False


    # for i in range(batch_num):
    #     p_id = i % n_buffer
    #     with torch.cuda.stream(pre_process_stream):
    #         if i >= n_buffer:
    #             pre_process_stream.wait_event(comp_end_event[p_id])
    #         torch.cuda.nvtx.range_push("generate_data")
    #         (hidden_states[p_id], total_seq_len[p_id], seq_position_info[p_id], seq_position_info_tensor[p_id],  partition_part_index_tensor[p_id],  partition_part_tensor[p_id], attention_masks[p_id], token_type_ids[p_id], position_ids[p_id],total_comp_block_num[p_id]) = pre_process_input_data(queue1,block_size,output_layer_temp[p_id],intermediate_temp[p_id],attention_output[p_id])
    #         # if hidden_states[pre_process_cnt] == None:
    #         #     end_flag = True
    #         # print("release pre_process_end_event {}".format(pre_process_cnt))
    #         torch.cuda.nvtx.range_pop()
    #         pre_process_end_event[p_id].record()

    #     with torch.cuda.stream(comp_stream):
    #         # print("wait pre_process_end_event {}".format(comp_cnt))
    #         comp_stream.wait_event(pre_process_end_event[p_id])
    #         torch.cuda.nvtx.range_push("Bert comp")
    #         metadata.update_meta_data(total_seq_len[p_id],seq_position_info[p_id][-1]+1,total_comp_block_num[p_id],seq_position_info[p_id])
    
    #         output = bertModel(hidden_states[p_id], total_seq_len[p_id], seq_position_info[p_id], seq_position_info_tensor[p_id],  partition_part_index_tensor[p_id],  partition_part_tensor[p_id], attention_masks[p_id], token_type_ids[p_id], position_ids[p_id],output_layer_temp[p_id],intermediate_temp[p_id],attention_output[p_id])
    #         torch.cuda.nvtx.range_pop()
    #         # print("release comp_end_event {}".format(comp_cnt))
    #         comp_end_event[p_id].record()

    model_name = 'sparse_attn'
    if model_name == 'sparse_attn':
        model = BigBirdModel.from_pretrained("google/bigbird-roberta-base").to(device).half()
        bertModel = BertModelNoPooler.from_torch(model)
        metadata = cxx.MetaData()
        thread_block_limit = 240
        input_data = generate_input_data(10,1024,4096,block_size)
        position_ids = [np.arange(len(it)).tolist() for it in input_data]
        position_ids = reduce(operator.add, position_ids)
        seq_position_info,seq_position_info_tensor,total_comp_block_num,partition_part_index_tensor,partition_part_tensor = generate_data_metainfo(input_data,64,thread_block_limit)
        total_seq_len = seq_position_info[-1]*block_size
        input_data = reduce(operator.add, input_data)
        hidden_states = torch.tensor(input_data,dtype=torch.int,device = device)
        token_type_ids = torch.zeros_like(hidden_states)
        position_ids = torch.tensor(position_ids,dtype=torch.int,device = device)
        attention_masks = torch.zeros_like(hidden_states)
        output_layer_temp = torch.zeros(total_seq_len,768,dtype=torch.half,device=device)
        intermediate_temp = torch.zeros(total_seq_len,3072,dtype=torch.half,device=device)
        attention_output = torch.zeros(total_seq_len,768,dtype=torch.half,device=device)

        for i in range(3):
            metadata.update_meta_data(total_seq_len,seq_position_info[-1]+1,total_comp_block_num,seq_position_info)
            output = bertModel(hidden_states, total_seq_len,thread_block_limit, seq_position_info, seq_position_info_tensor,  partition_part_index_tensor,  partition_part_tensor, attention_masks, token_type_ids, position_ids,output_layer_temp,intermediate_temp,attention_output)

        for i in range(20):
            # print(i)
            metadata.update_meta_data(total_seq_len,seq_position_info[-1]+1,total_comp_block_num,seq_position_info)
            torch.cuda.nvtx.range_push("Bert")
            output = bertModel(hidden_states, total_seq_len,thread_block_limit, seq_position_info, seq_position_info_tensor,  partition_part_index_tensor,  partition_part_tensor, attention_masks, token_type_ids, position_ids,output_layer_temp,intermediate_temp,attention_output)
            torch.cuda.nvtx.range_pop()

        metadata.terminate_thread()
    elif model_name == "big_bird":
        model_name = "google/bigbird-roberta-base"
        model = BigBirdModel.from_pretrained("google/bigbird-roberta-base").to(device)
        config = model.config
        seq_lens = [1024,4096,3072,2048,2048,3072,1024,3072,2048,2048]
        seqlens = torch.tensor(seq_lens,dtype=torch.int,device = device)
        attention_mask = torch.arange(4096, device='cuda')[None, :] < seqlens[:, None]
        input_ids = torch.randint(0, config.vocab_size, (10, 4096), dtype=torch.long, device='cuda')
        attention_mask = attention_mask.type(torch.float)
        model.eval()
        with torch.no_grad():
            for i in range(3):
                out = model(input_ids, attention_mask=attention_mask)

            for i in range(20):
                nvtx.range_push("big_bird")
                out = model(input_ids, attention_mask=attention_mask)
                nvtx.range_pop()

    elif model_name == 'flash_attn':
        model_name = "google/bigbird-roberta-base"
        model = BigBirdModel.from_pretrained("google/bigbird-roberta-base").to(device)
        config = model.config
        config.hidden_act = "gelu_new"
        config.use_flash_attn = True
        config.fused_bias_fc = False
        config.fused_mlp = False
        config.fused_dropout_add_ln = False
        pretrained_state_dict = remap_state_dict(state_dict_from_pretrained(model_name), config)
        model = BertForPreTraining.from_pretrained(model_name,config).cuda().to(dtype=torch.float16)

        model.eval()
        batch_size = 10
        max_seqlen = 4096
        seq_lens = [1024,4096,3072,2048,2048,3072,1024,3072,2048,2048]
        seqlens = torch.tensor(seq_lens,dtype=torch.int,device = device)
        attention_mask = torch.arange(max_seqlen, device='cuda')[None, :] < seqlens[:, None]
        input_ids = torch.randint(0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device='cuda')
        with torch.no_grad():
            for i in range(3):
                out = model.bert(input_ids, attention_mask=attention_mask)

            for i in range(20):
                nvtx.range_push("flash_attn")
                out = model.bert(input_ids, attention_mask=attention_mask)
                nvtx.range_pop()

    print("end")

    # torch.cuda.synchronize()
    # print("end")
    # input_data = generate_input_data(10,1024,4096,block_size)
    # position_ids = [np.arange(len(it)).tolist() for it in input_data]
    # position_ids = reduce(operator.add, position_ids)
    # seq_position_info,seq_position_info_tensor,total_comp_block_num,partition_part_index_tensor,partition_part_tensor = generate_data_metainfo(input_data,64,160)
    # total_seq_len = seq_position_info[-1]*block_size
    # input_data = reduce(operator.add, input_data)
    # hidden_states = torch.IntTensor(input_data).to(device)
    # token_type_ids = torch.zeros_like(hidden_states)
    # position_ids = torch.IntTensor(position_ids).to(device)
    # attention_masks = torch.zeros_like(hidden_states)

    # metadata.update_meta_data(total_seq_len,seq_position_info[-1]+1,total_comp_block_num,seq_position_info)
    # for i in range(10):
    #     metadata.update_meta_data(total_seq_len,seq_position_info[-1]+1,total_comp_block_num,seq_position_info)
    #     output = bertModel(hidden_states,total_seq_len,seq_position_info,seq_position_info_tensor, partition_part_index_tensor, partition_part_tensor,attention_masks,token_type_ids,position_ids)

    # print(queue2.empty())
    # while batch_num != 0:

        # (hidden_states,total_seq_len,seq_position_info,seq_position_info_tensor, partition_part_index_tensor, partition_part_tensor,attention_masks,token_type_ids,position_ids) = queue2.get()
        # # print(batch_num)
        # metadata.update_meta_data(total_seq_len,seq_position_info[-1]+1,total_comp_block_num,seq_position_info)
        # output = bertModel(hidden_states,total_seq_len,seq_position_info,seq_position_info_tensor, partition_part_index_tensor, partition_part_tensor,attention_masks,token_type_ids,position_ids)
        # # print(output.shape)
        # del hidden_states,seq_position_info_tensor,partition_part_index_tensor,partition_part_tensor,attention_masks,token_type_ids,position_ids
        # batch_num -= 1
        # e.set()


    # t1.join()
    
    # t2.join()


