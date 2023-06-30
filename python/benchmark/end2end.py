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


from ft_longformer import build_ft_longformer,parse_from_config

from util import generate_parition_plan,generate_data_metainfo,generate_array_with_avg,device,generate_input_data

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def end2end(args):

    model_name = args.model_name
    batch_size = args.batch_size
    seq_len = args.sequence_length
    thread_block_limit = args.thread_block

    max_seqlen = seq_len+512
    if max_seqlen % 512 != 0:
        max_seqlen += 256

    max_seqlen = min(4096,max_seqlen)
    seq_lens = generate_array_with_avg(seq_len,batch_size,max_seqlen)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_rank)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    before_used_mem = meminfo.used / 1024 /1024

    total_time = datetime.timedelta()

    bigbird_dir = args.bigbird_dir

    if model_name == 'raptor_t':
        block_size = 64
        model = BigBirdModel.from_pretrained(bigbird_dir).to(device).half()
        bertModel = BertModelNoPooler.from_torch(model)
        metadata = cxx.MetaData()
        if thread_block_limit == 0 and batch_size == 1:
            thread_block_limit = 160
        else:
            thread_block_limit = 720
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

        test_info = {}

        for i in range(3):
            metadata.update_meta_data(total_seq_len,seq_position_info[-1]+1,total_comp_block_num,seq_position_info)
            output,test_info = bertModel(hidden_states, total_seq_len,thread_block_limit, seq_position_info, seq_position_info_tensor,  partition_part_index_tensor,  partition_part_tensor, attention_masks, token_type_ids, position_ids,output_layer_temp,intermediate_temp,attention_output,test_info)
        # metadata.update_meta_data(total_seq_len,seq_position_info[-1]+1,total_comp_block_num,seq_position_info)
        # output,test_info = bertModel(hidden_states, total_seq_len,thread_block_limit, seq_position_info, seq_position_info_tensor,  partition_part_index_tensor,  partition_part_tensor, attention_masks, token_type_ids, position_ids,output_layer_temp,intermediate_temp,attention_output,test_info)

        for i in range(20):
            metadata.update_meta_data(total_seq_len,seq_position_info[-1]+1,total_comp_block_num,seq_position_info)
            begin = datetime.datetime.now()
            output,test_info = bertModel(hidden_states, total_seq_len,thread_block_limit, seq_position_info, seq_position_info_tensor,  partition_part_index_tensor,  partition_part_tensor, attention_masks, token_type_ids, position_ids,output_layer_temp,intermediate_temp,attention_output,test_info)
            end = datetime.datetime.now()
            total_time += end - begin
        metadata.terminate_thread()

    elif model_name == "pytorch":
        model = BigBirdModel.from_pretrained(bigbird_dir).to(device).half()
        config = model.config
        seqlens = torch.tensor(seq_lens,dtype=torch.int,device = device)
        attention_mask = torch.arange(max_seqlen, device='cuda')[None, :] < seqlens[:, None]
        input_ids = torch.randint(0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device='cuda')
        attention_mask = attention_mask.type(torch.float)
        model.eval()
        
        with torch.no_grad():
            for i in range(3):
                out = model(input_ids, attention_mask=attention_mask)
            
            test_info = {"pre_process":0.0,"attention":0.0}
            for i in range(20):
                begin = datetime.datetime.now()
                out = model(input_ids, attention_mask=attention_mask,test_info = test_info)
                end = datetime.datetime.now()
                total_time += end - begin

    elif model_name == 'flash_attn':
        model = BigBirdModel.from_pretrained(bigbird_dir).to(device)
        config = model.config
        config.hidden_act = "gelu_new"
        config.use_flash_attn = True
        config.fused_bias_fc = False
        config.fused_mlp = False
        config.fused_dropout_add_ln = False
        model = BertForPreTraining.from_pretrained("google/bigbird-roberta-base",config).cuda().to(dtype=torch.float16)

        model.eval()
        seqlens = torch.tensor(seq_lens,dtype=torch.int,device = device)
        attention_mask = torch.arange(max_seqlen, device='cuda')[None, :] < seqlens[:, None]
        input_ids = torch.randint(0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device='cuda')
        with torch.no_grad():
            for i in range(3):
                out = model.bert(input_ids, attention_mask=attention_mask)
        # out = model.bert(input_ids, attention_mask=attention_mask)

            for i in range(20):
                begin = datetime.datetime.now()
                out = model.bert(input_ids, attention_mask=attention_mask)
                end = datetime.datetime.now()
                total_time += end - begin

    elif model_name == 'bert_like':
        model = BigBirdModel.from_pretrained(bigbird_dir).to(device)
        config = model.config
        model = BertForPreTraining.from_pretrained("google/bigbird-roberta-base",config).cuda().to(dtype=torch.float16)

        model.eval()
        seqlens = torch.tensor(seq_lens,dtype=torch.int,device = device)
        attention_mask = torch.arange(max_seqlen, device='cuda')[None, :] < seqlens[:, None]
        input_ids = torch.randint(0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device='cuda')
        with torch.no_grad():
            for i in range(3):
                out = model.bert(input_ids, attention_mask=attention_mask)

            for i in range(20):
                begin = datetime.datetime.now()
                out = model.bert(input_ids, attention_mask=attention_mask)
                end = datetime.datetime.now()
                total_time += end - begin

    elif model_name == 'fasttransformer':
        model_dir = args.longformer
        ft_longformer_lib = args.ft_longformer_lib
        max_global_token_num = 128
        data_type = 'fp16'

        (layer_num, hidden_size, head_num, size_per_head, intermediate_size, local_attn_window_size, attn_scaler) = parse_from_config(model_dir)


        ft_longformer = build_ft_longformer(model_dir, layer_num, head_num, size_per_head,
                                        intermediate_size, local_attn_window_size,
                                        max_global_token_num, batch_size, max_seqlen,
                                        attn_scaler, ft_longformer_lib, data_type)
        seqlens = torch.tensor(seq_lens,dtype=torch.int,device = device)
        attention_mask = torch.arange(max_seqlen, device='cuda')[None, :] < seqlens[:, None]
        input_ids = torch.randint(0, 30000, (batch_size, max_seqlen), dtype=torch.long, device='cuda')
        global_attention_mask = torch.zeros(
                    input_ids.shape, dtype=torch.long, device=input_ids.device)
        global_attention_mask[0][0:128] = 1
        with torch.no_grad():
            for i in range(3):
                out = ft_longformer(input_ids, attention_mask=attention_mask,global_attention_mask = global_attention_mask)

            for i in range(20):
                begin = datetime.datetime.now()
                out = ft_longformer(input_ids, attention_mask=attention_mask,global_attention_mask = global_attention_mask)
                end = datetime.datetime.now()
                total_time += end - begin


    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_mem = meminfo.used / 1024 /1024


    print("{} {} {} {} {} {}".format(model_name,batch_size,max_seqlen,seq_len,total_time.total_seconds()/20*1000,used_mem-before_used_mem))
