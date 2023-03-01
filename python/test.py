# import torch
# import lltm_cpp as cxx
# from modeling_bert import BertIntermediate,BertOutput,BertAttention,BertLayer,BertEncoder,BertModel,BertModelNoPooler
# from transformers import BertModel,BigBirdModel
# import os
# import torch.cuda.nvtx as nvtx
# # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# device = torch.device('cuda')
# model = BigBirdModel.from_pretrained("google/bigbird-roberta-base").to(device)

# def test_BertIntermediate():
#     bertIntermediate = BertIntermediate.from_torch(model.encoder.layer[0].intermediate)
#     weight = torch.t(model.encoder.layer[0].intermediate.dense.weight)
#     bias = model.encoder.layer[0].intermediate.dense.bias

#     input_tensor = torch.rand(2,768).to(device)
#     out = torch.mm(input_tensor,weight)
#     out = out + bias
#     m = torch.nn.GELU()
#     out = m(out)
#     out_tensor = bertIntermediate(input_tensor)

# def test_BertOutput():
#     bertOutput = BertOutput.from_torch(model.encoder.layer[0].attention.output)
#     weight = torch.t(model.encoder.layer[0].attention.output.dense.weight)
#     bias = model.encoder.layer[0].attention.output.dense.bias
#     layer_weight = model.encoder.layer[0].attention.output.LayerNorm.weight
#     layer_bias = model.encoder.layer[0].attention.output.LayerNorm.bias
#     hidden_states = torch.rand(4096,768).to(device)
#     input_data = torch.rand(4096,768).to(device)
#     out_tensor = bertOutput(hidden_states,input_data,4096)

#     out = torch.mm(hidden_states,weight)
#     out = out + bias + input_data
#     layer_norm = torch.nn.LayerNorm(768,1e-5,False)
#     out = layer_norm(out)
#     out = out*layer_weight + layer_bias
# # bertIntermediate(input_tensor,out_tensor)
# # test_BertIntermediate()

# def test_bertAttention():
#     bertAttention = BertAttention.from_torch(model.encoder.layer[0].attention)
#     input_data = torch.rand(4096,768).to(device)
#     # out_tensor = torch.rand(4096,768).to(device)

# def test_BertLayer():
#     bertLayer = BertLayer.from_torch(model.encoder.layer[0])
#     hidden_states = torch.rand(4096,768).to(device)
#     output = bertLayer(hidden_states,768)

# # def test_BertEncoder():
# # bertEncoder = BertEncoder.from_torch(model.encoder)
# # hidden_states = torch.rand(4096,768).to(device)

# bertModel = BertModelNoPooler.from_torch(model)
# hidden_states = torch.randint(0,2000,(4096,)).to(device)
# token_type_ids = torch.zeros_like(hidden_states)
# position_ids = torch.arange(0,4096,1).to(device)
# attention_masks = torch.zeros_like(hidden_states)

# metadata = cxx.MetaData()
# # metadata.update_meta_data(4096,65,622,[0,64])

# # tensor_set = cxx.TensorSet.get_instance()
# # tensor_set.update_tensor_set(4096,622,65)
# for i in range(0,10):
#     metadata.update_meta_data(4096,65,622,[0,64])
#     nvtx.range_push("BERT")
#     output = bertModel(hidden_states,4096,attention_masks,token_type_ids,position_ids)
#     nvtx.range_pop()

# metadata.terminate_thread()

# a = 1
# # output = model.encoder.layer[0].attention.output
# # input_tensor = torch.rand(2,768).to(device)

# # block_num_array = [64,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,64,10,7,8,8,8,8,8,8,7,10]

# # thread_block_num = 160
# # seq_start_index = [(0,64),(64,10)]

# # import heapq

# # def equal_division(block_num_array:list,seq_start_index:list,thread_block_num:int,head_num:int):
# #     sorted_nums = sorted(enumerate(block_num_array), key=lambda x: x[1],reverse=True)
# #     index = [i[0] for i in sorted_nums]
# #     sorted_value = [i[1] for i in sorted_nums]
# #     partition_part = [[] for i in range(0,thread_block_num)]
# #     min_heap = [(0,i) for i in range(0,thread_block_num)]
# #     for i in range(0,len(sorted_value)):
# #         for j in range(0,head_num):
# #             pop_element = heapq.heappop(min_heap)
# #             start_index = (0,0)
# #             for k in seq_start_index:
# #                 if(index[i] >= k[0]):
# #                     start_index = k
# #                 else:
# #                     break
# #             partition_part[pop_element[1]].append(index[i] - start_index[0] +j*start_index[1]+start_index[0]*head_num)
# #             pop_element = (pop_element[0]+sorted_value[i],pop_element[1])
# #             heapq.heappush(min_heap,pop_element)
# #     return partition_part

# # partition_part = equal_division(block_num_array,seq_start_index,160,12)
# # len_ = [len(it) for it in partition_part]
# # len_ = [sum(len_[0:i]) for i in range(0,len(len_)+1)]

# # partition_part_ = reduce(operator.add, partition_part)


import torch


def get_device():
    return torch.device('cuda')
def new_stream():
    return torch.cuda.Stream(get_device())

def generate_data(m=2048, n=4096, k=1024):
    # x = torch.rand(m, n).to(get_device())
    # w = torch.rand(n, k).to(get_device())
    x = torch.rand(m, n, device=get_device())
    w = torch.rand(n, k, device=get_device())
    c = x[0, :] + w[:, 0]
    return x, w

def cal(x, w):
    c = x @ w

n_iters = 10

mem_stream = new_stream()
comp_stream = new_stream()
n_buffer = 2
mem_end_events = [torch.cuda.Event() for i in range(n_buffer)]
comp_end_event = [torch.cuda.Event() for i in range(n_buffer)]

xs = [None for i in range(n_buffer)]
ws = [None for i in range(n_buffer)]
mem_process = 0
cal_process = 0

def run():
    for i in range(n_iters):
        # p_id = i & (n_buffer-1)
        p_id = i % n_buffer
        with torch.cuda.stream(mem_stream):
            if i >= n_buffer:
                mem_stream.wait_event(comp_end_event[p_id])
            torch.cuda.nvtx.range_push("generate_data")
            x, w = generate_data()
            xs[p_id] = x
            ws[p_id] = w
            torch.cuda.nvtx.range_pop
            mem_end_events[p_id].record()

        with torch.cuda.stream(comp_stream):
            comp_stream.wait_event(mem_end_events[p_id])
            torch.cuda.nvtx.range_push("cal")
            x, w = xs[p_id], ws[p_id]
            cal(x, w)
            torch.cuda.nvtx.range_pop
            comp_end_event[p_id].record()

run()

# import torch.distributed as dist
# name = "default"
# device = get_device()
# wait, warmup, active = 1, 3, 3
with torch.profiler.profile(
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f"profiles/{name}"),
    schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],  # torch.profiler.ProfilerActivity.CPU, 
    profile_memory=True, record_shapes=True, with_flops=True, with_stack=True) as p:

    for i in range(wait + warmup + active):
        run()
        p.step()


