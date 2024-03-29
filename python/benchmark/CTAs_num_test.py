import argparse
import torch
import numpy as np

from functools import reduce
import operator
import time
import pynvml
import datetime
import os

import raptor_t as cxx
from transformers import LongformerModel, AutoTokenizer
from modeling_bert import BertIntermediate,BertOutput,BertAttention,BertLayer,BertEncoder,BertModel,BertModelNoPooler
from transformers import BertModel,BigBirdModel


from transformers import BigBirdConfig,BigBirdModel
from flash_attn.models.bert import BertModel, BertForPreTraining
from flash_attn.models.bert import remap_state_dict
from flash_attn.utils.pretrained import state_dict_from_pretrained

from util import generate_parition_plan,generate_data_metainfo,generate_array_with_avg,device,generate_input_data

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def CTAs_num_test(args):
    model_name = args.model_name
    batch_size = args.batch_size
    seq_len = args.sequence_length
    thread_block_limit = args.thread_block
    kernel_fusion = args.kernel_fusion

    max_seqlen = seq_len+512
    if max_seqlen % 512 != 0:
        max_seqlen += 256

    max_seqlen = min(4096,max_seqlen)
    seq_lens = generate_array_with_avg(seq_len,batch_size,max_seqlen)

    bigbird_dir = args.bigbird_dir

    block_size = 64
    model = BigBirdModel.from_pretrained(bigbird_dir).to(device).half()
    bertModel = BertModelNoPooler.from_torch(model)
    metadata = cxx.MetaData()

    input_data = generate_input_data(batch_size,1024,4096,block_size,seq_lens)
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
        output,test_info = bertModel(hidden_states, total_seq_len,thread_block_limit, seq_position_info, seq_position_info_tensor,  partition_part_index_tensor,  partition_part_tensor, attention_masks, token_type_ids, position_ids,output_layer_temp,intermediate_temp,attention_output)
    
    total_time_info = []

    for i in range(20):
        total_time = datetime.timedelta()
        metadata.update_meta_data(total_seq_len,seq_position_info[-1]+1,total_comp_block_num,seq_position_info)
        begin = datetime.datetime.now()
        output,test_info = bertModel(hidden_states, total_seq_len,thread_block_limit, seq_position_info, seq_position_info_tensor,  partition_part_index_tensor,  partition_part_tensor, attention_masks, token_type_ids, position_ids,output_layer_temp,intermediate_temp,attention_output)
        end = datetime.datetime.now()
        total_time += end - begin
        total_time_info.append(total_time.total_seconds()*1000)
    metadata.terminate_thread()



    median_time = np.median(total_time_info)
    total_time_result = []
    for i in range(len(total_time_info)):
        if total_time_info[i] > median_time*1.2:
            continue
        total_time_result.append(total_time_info[i])

        

    print("{} {} {} {} {} {}".format(model_name,thread_block_limit,batch_size,max_seqlen,seq_len,np.mean(total_time_result)))
