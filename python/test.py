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

def test_BertEncoder():
    bertEncoder = BertEncoder.from_torch(model.encoder)
    hidden_states = torch.rand(4096,768).to(device)
    output = bertEncoder(hidden_states,4096)

output = model.encoder.layer[0].attention.output
input_tensor = torch.rand(2,768).to(device)
