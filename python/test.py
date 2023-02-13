import torch
from modeling_bert import BertIntermediate,BertOutput,BertAttention,BertLayer,BertEncoder
from transformers import BertModel

device = torch.device('cuda')
model = BertModel.from_pretrained("bert-base-uncased").to(device)

def test_BertIntermediate():
    bertIntermediate = BertIntermediate.from_torch(model.encoder.layer[0].intermediate)
    weight = torch.t(model.encoder.layer[0].intermediate.dense.weight)
    bias = model.encoder.layer[0].intermediate.dense.bias

    input_tensor = torch.rand(2,768).to(device)
    out = torch.mm(input_tensor,weight)
    out = out + bias
    m = torch.nn.GELU()
    out = m(out)
    out_tensor = bertIntermediate(input_tensor)

def test_BertOutput():
    bertOutput = BertOutput.from_torch(model.encoder.layer[0].attention.output)
    weight = torch.t(model.encoder.layer[0].attention.output.dense.weight)
    bias = model.encoder.layer[0].attention.output.dense.bias
    layer_weight = model.encoder.layer[0].attention.output.LayerNorm.weight
    layer_bias = model.encoder.layer[0].attention.output.LayerNorm.bias
    hidden_states = torch.rand(4096,768).to(device)
    input_data = torch.rand(4096,768).to(device)
    out_tensor = bertOutput(hidden_states,input_data,4096)

    out = torch.mm(hidden_states,weight)
    out = out + bias + input_data
    layer_norm = torch.nn.LayerNorm(768,1e-5,False)
    out = layer_norm(out)
    out = out*layer_weight + layer_bias
# bertIntermediate(input_tensor,out_tensor)
# test_BertIntermediate()

def test_bertAttention():
    bertAttention = BertAttention.from_torch(model.encoder.layer[0].attention)
    input_data = torch.rand(4096,768).to(device)
    # out_tensor = torch.rand(4096,768).to(device)

def test_BertLayer():
    bertLayer = BertLayer.from_torch(model.encoder.layer[0])
    hidden_states = torch.rand(4096,768).to(device)
    output = bertLayer(hidden_states,768)

# def test_BertEncoder():
bertEncoder = BertEncoder.from_torch(model.encoder)
hidden_states = torch.rand(4096,768).to(device)
    # for i in range(0,10):
output = bertEncoder(hidden_states,4096)

# output = model.encoder.layer[0].attention.output
# input_tensor = torch.rand(2,768).to(device)

# block_num_array = [64,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,64,10,7,8,8,8,8,8,8,7,10]

# thread_block_num = 160
# seq_start_index = [(0,64),(64,10)]

# import heapq

# def equal_division(block_num_array:list,seq_start_index:list,thread_block_num:int,head_num:int):
#     sorted_nums = sorted(enumerate(block_num_array), key=lambda x: x[1],reverse=True)
#     index = [i[0] for i in sorted_nums]
#     sorted_value = [i[1] for i in sorted_nums]
#     partition_part = [[] for i in range(0,thread_block_num)]
#     min_heap = [(0,i) for i in range(0,thread_block_num)]
#     for i in range(0,len(sorted_value)):
#         for j in range(0,head_num):
#             pop_element = heapq.heappop(min_heap)
#             start_index = (0,0)
#             for k in seq_start_index:
#                 if(index[i] >= k[0]):
#                     start_index = k
#                 else:
#                     break
#             partition_part[pop_element[1]].append(index[i] - start_index[0] +j*start_index[1]+start_index[0]*head_num)
#             pop_element = (pop_element[0]+sorted_value[i],pop_element[1])
#             heapq.heappush(min_heap,pop_element)
#     return partition_part

# partition_part = equal_division(block_num_array,seq_start_index,160,12)
# len_ = [len(it) for it in partition_part]
# len_ = [sum(len_[0:i]) for i in range(0,len(len_)+1)]

# partition_part_ = reduce(operator.add, partition_part)