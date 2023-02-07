import torch
import lltm_cpp as cxx
import torch.cuda.nvtx as nvtx

from typing import Union, Optional, Sequence
from transformers.models.bert.modeling_bert import BertEmbeddings as TorchBertEmbeddings
from transformers.models.bert.modeling_bert import BertIntermediate as TorchBertIntermediate
from transformers.models.bert.modeling_bert import BertOutput as TorchBertOutput
from transformers.models.bert.modeling_bert import BertAttention as TorchBertAttention
from transformers.models.bert.modeling_bert import BertLayer as TorchBertLayer
from transformers.models.bert.modeling_bert import BertEncoder as TorchBertEncoder
from transformers.models.bert.modeling_bert import BertModel as TorchBertModel
from transformers.models.bert.modeling_bert import BertPooler as TorchBertPooler

def to_param_dict(torch_module: torch.nn.Module):
    return {k: v for k, v in torch_module.named_parameters()}

def create_empty_if_none(tensor:torch.Tensor,shape,device:torch.device):
    if tensor is None:
        tensor = torch.zeros(shape).to(device)
    return tensor

class BertIntermediate(cxx.BertIntermediate):
    def __call__(self,
                 input_tensor: torch.Tensor,
                 total_seq_len: int,
                 output: Optional[torch.Tensor] = None) -> torch.Tensor:
        nvtx.range_push("intermediate")
        output = create_empty_if_none(output,(total_seq_len,3072),torch.device('cuda'))
        super(BertIntermediate,self).__call__(input_tensor, output)
        nvtx.range_pop()
        return output

    @staticmethod
    def from_torch(intermediate: TorchBertIntermediate):
        intermediate_params = to_param_dict(intermediate)
        weight = torch.clone(
            torch.t(intermediate_params["dense.weight"]).contiguous())
        return BertIntermediate(
            weight,
            intermediate_params['dense.bias'])

class BertOutput(cxx.BertOutput):
    def __call__(self,
                 intermediate_output: torch.Tensor,
                 attention_output: torch.Tensor,
                 total_seq_len: int,
                 output: Optional[torch.Tensor] = None) -> torch.Tensor:
        nvtx.range_push("output")
        output = create_empty_if_none(output,(total_seq_len,768),torch.device('cuda'))
        super(BertOutput, self).__call__(intermediate_output,attention_output,output)
        nvtx.range_pop()
        return output

    @staticmethod
    def from_torch(output: TorchBertOutput):
        params = to_param_dict(output)
        weight = torch.clone(torch.t(params["dense.weight"]).contiguous())
        return BertOutput(weight, params["dense.bias"],
                          params["LayerNorm.weight"],
                          params["LayerNorm.bias"])

class BertAttention(cxx.BertAttention):
    
    def __call__(self,
                 input_tensor: torch.Tensor,
                 total_seq_len: int,
                 attention_mask: Optional[torch.Tensor] = torch.empty(0),
                 head_mask: Optional[torch.Tensor] = torch.empty(0),
                 output_attentions: Optional[bool] = False,
                 is_trans_weight: Optional[bool] = False) -> torch.Tensor:
        nvtx.range_push("attention")
        context_layer = torch.zeros_like(input_tensor)
        super(BertAttention,self).__call__(input_tensor,attention_mask,context_layer,total_seq_len)
        nvtx.range_pop()
        return context_layer

    @staticmethod
    def from_torch(attention: TorchBertAttention):
        params = {k: v for k, v in attention.named_parameters()}
        with torch.no_grad():
            # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
            qkv_weight = torch.clone(
                torch.t(
                    torch.cat((params['self.query.weight'],
                               params['self.key.weight'],
                               params['self.value.weight']),
                              0).contiguous()).contiguous())
            qkv_bias = torch.cat(
                (params['self.query.bias'], params['self.key.bias'],
                 params['self.value.bias']), 0).contiguous()

            output_weight = torch.clone(
                torch.t(params['output.dense.weight']).contiguous())
            att = BertAttention(
                qkv_weight, qkv_bias,
                output_weight,
                params['output.dense.bias'],
                params['output.LayerNorm.weight'],
                params['output.LayerNorm.bias'],
                attention.self.num_attention_heads)

            return att

class BertLayer:
    def __init__(self, attention: BertAttention,
                 intermediate: BertIntermediate, output: BertOutput):
        self.attention = attention
        self.intermediate = intermediate
        self.output = output

    def __call__(self,
                 hidden_states: torch.Tensor,
                 total_seq_len: int ,
                 attention_mask: Optional[torch.Tensor] = torch.empty(0),
                 head_mask: Optional[torch.Tensor] = torch.empty(0),
                 output_attentions=False):
        nvtx.range_push("layer")
        attention_output = self.attention(
            hidden_states,
            total_seq_len,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            )
        intermediate_output = self.intermediate(attention_output,total_seq_len)
        layer_out = self.output(intermediate_output,
                                attention_output,
                                total_seq_len)
        nvtx.range_pop()
        return layer_out

    @staticmethod
    def from_torch(layer: TorchBertLayer):
        return BertLayer(BertAttention.from_torch(layer.attention),
                         BertIntermediate.from_torch(layer.intermediate),
                         BertOutput.from_torch(layer.output))

class BertEncoder:
    def __init__(self, layer: Sequence[BertLayer]):
        self.layer = layer

    def __call__(self,
                 hidden_states: torch.Tensor,
                 total_seq_len: int,
                 attention_mask: Optional[torch.Tensor] = torch.empty(0),
                 head_mask: Optional[torch.Tensor] = torch.empty(0),
                 output_attentions: Optional[bool] = False,
                 output_hidden_states: Optional[bool] = False):
        for l in self.layer:
            layer_outputs = l(hidden_states=hidden_states,
                              total_seq_len=total_seq_len,
                              attention_mask=attention_mask,
                              output_attentions=output_attentions)
            hidden_states = layer_outputs
        
        return hidden_states
        # return outputs

    @staticmethod
    def from_torch(encoder: TorchBertEncoder):
        layer = [
            BertLayer.from_torch(bert_layer) for bert_layer in encoder.layer
        ]
        return BertEncoder(layer)