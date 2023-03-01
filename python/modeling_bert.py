import torch
import torch.nn as nn
import lltm_cpp as cxx
import torch.cuda.nvtx as nvtx
import operator
import enum
import heapq
from functools import reduce
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

class BertEmbeddings():

    def __init__(self,word_embeddings_weight,position_embeddings_weight,token_type_embeddings_weight,layerNorm_weight,layerNorm_bias,vocab_size:int = 50358, hidden_size:int = 768, pad_token_id:int = 0,max_position_embeddings:int  = 4096,type_vocab_size:int = 2) -> None:

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id).to("cuda")
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size).to("cuda")
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size).to("cuda")

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12).to("cuda")

        self.word_embeddings.weight.data.copy_(word_embeddings_weight)
        self.position_embeddings.weight.data.copy_(position_embeddings_weight)
        self.token_type_embeddings.weight.data.copy_(token_type_embeddings_weight)

        self.LayerNorm.weight.data.copy_(layerNorm_weight)
        self.LayerNorm.bias.data.copy_(layerNorm_bias)


    def __call__(self,
                 input_ids: torch.Tensor = None,
                 position_ids: torch.Tensor = None,
                 token_type_ids: torch.Tensor = None,
                 output: torch.Tensor = None):
        if position_ids is None or position_ids is None or token_type_ids is None:
            print("Missing parameters\n")
            exit(0)

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        return embeddings

    @staticmethod
    def from_torch(bert_embedding: TorchBertEmbeddings) -> 'BertEmbeddings':
        params = to_param_dict(bert_embedding)
        return BertEmbeddings(params['word_embeddings.weight'],
                              params['position_embeddings.weight'],
                              params['token_type_embeddings.weight'],
                              params['LayerNorm.weight'],
                              params['LayerNorm.bias'])

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
                 seq_position_info: list ,
                 seq_position_info_tensor:torch.Tensor,
                 partition_part_index_tensor:torch.Tensor,
                 partition_part_tensor:torch.Tensor,
                 context_layer:torch.Tensor,
                 block_limit:int,
                 attention_mask: Optional[torch.Tensor] = torch.empty(0),
                 head_mask: Optional[torch.Tensor] = torch.empty(0),
                 output_attentions: Optional[bool] = False,
                 is_trans_weight: Optional[bool] = False) -> torch.Tensor:
        nvtx.range_push("attention")
        super(BertAttention,self).__call__(input_tensor,attention_mask,context_layer,seq_position_info,seq_position_info_tensor,partition_part_index_tensor,partition_part_tensor,block_limit)
        nvtx.range_pop()
        return context_layer

    @staticmethod
    def from_torch(attention: TorchBertAttention,layer_idx:int):
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
                attention.self.num_attention_heads,
                layer_idx)

            return att

class BertLayer:
    def __init__(self, attention: BertAttention,
                 intermediate: BertIntermediate, output: BertOutput):
        self.attention = attention
        self.intermediate = intermediate
        self.output = output

    def __call__(self,
                 hidden_states: torch.Tensor,
                 seq_position_info: list ,
                 seq_position_info_tensor:torch.Tensor,
                 partition_part_index_tensor:torch.Tensor,
                 partition_part_tensor:torch.Tensor,
                 intermediate_temp:torch.Tensor,
                 output_layer_temp:torch.Tensor,
                 attention_output:torch.Tensor,
                 block_limit:int,
                 attention_mask: Optional[torch.Tensor] = torch.empty(0),
                 head_mask: Optional[torch.Tensor] = torch.empty(0),
                 output_attentions=False):
        nvtx.range_push("layer")
        self.attention(
            hidden_states,
            seq_position_info,
            seq_position_info_tensor,
            partition_part_index_tensor,
            partition_part_tensor,
            attention_output,
            block_limit,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            )
        self.intermediate(attention_output,seq_position_info[-1]*64,intermediate_temp)
        self.output(intermediate_temp,
                                attention_output,
                                seq_position_info[-1]*64,hidden_states)
        nvtx.range_pop()
        return hidden_states

    @staticmethod
    def from_torch(layer: TorchBertLayer,layer_idx:int):
        return BertLayer(BertAttention.from_torch(layer.attention, layer_idx),
                         BertIntermediate.from_torch(layer.intermediate),
                         BertOutput.from_torch(layer.output))

class BertEncoder:
    def __init__(self, layer: Sequence[BertLayer]):
        self.layer = layer


    def __call__(self,
                 hidden_states: torch.Tensor,
                 total_seq_len: int,
                 thread_block_limit:int,
                 seq_position_info,
                 seq_position_info_tensor,
                 partition_part_index_tensor,
                 partition_part_tensor,
                 output_layer_temp,
                 intermediate_temp,
                 attention_output,
                 attention_mask: Optional[torch.Tensor] = torch.empty(0),
                 head_mask: Optional[torch.Tensor] = torch.empty(0),
                 output_attentions: Optional[bool] = False,
                 output_hidden_states: Optional[bool] = False):
        # if(True):
        #     tensor_set = cxx.TensorSet.get_instance()
        #     tensor_set.update_tensor_set(4096,622,65)
        for l in self.layer:
            layer_outputs = l(hidden_states=hidden_states,
                              block_limit=thread_block_limit,
                              seq_position_info=seq_position_info,
                              seq_position_info_tensor=seq_position_info_tensor,
                              partition_part_index_tensor=partition_part_index_tensor,
                              partition_part_tensor=partition_part_tensor,
                              output_layer_temp = output_layer_temp,
                              intermediate_temp = intermediate_temp,
                              attention_output = attention_output,
                              attention_mask=attention_mask,
                              output_attentions=output_attentions)
            hidden_states = layer_outputs
        
        return hidden_states
        # return outputs

    @staticmethod
    def from_torch(encoder: TorchBertEncoder):
        layer = [
            BertLayer.from_torch(bert_layer,layer_idx) for layer_idx,bert_layer in enumerate(encoder.layer)
        ]
        return BertEncoder(layer)
    

class SequencePool():
    def __call__(self,
                 input_tensor: torch.Tensor,
                 output_tensor: Optional[torch.Tensor] = None):
        # input_tensor = input_tensor
        # output_tensor = create_empty_if_none(output_tensor)
        # super(SequencePool, self).__call__(input_tensor, output_tensor)
        return output_tensor

class BertPooler(cxx.BertPooler):
    def __call__(self,
                 input_tensor: torch.Tensor,
                 output: Optional[torch.Tensor] = None):
        input_tensor = input_tensor
        output = create_empty_if_none(output)
        super(BertPooler, self).__call__(input_tensor, output)
        return output

    @staticmethod
    def from_torch(pooler: TorchBertPooler,bigbird:bool = True):
        pooler_params = to_param_dict(pooler)
        if bigbird:
            weight = torch.clone(
            torch.t(pooler_params['weight']).contiguous())
            return BertPooler(weight, pooler_params['bias'])
        
        weight = torch.clone(
            torch.t(pooler_params['dense.weight']).contiguous())
        return BertPooler(weight, pooler_params['dense.bias'])

class PoolingType(enum.Enum):
    FIRST = "First"
    LAST = "Last"
    MEAN = "Mean"
    MAX = "Max"


PoolingMap = {
    PoolingType.FIRST: "First",
    PoolingType.LAST: "Last",
    PoolingType.MEAN: "Mean",
    PoolingType.MAX: "Max"
}



class BertModelNoPooler:
    def __init__(self, embeddings: BertEmbeddings, encoder: BertEncoder):
        self.embeddings = embeddings
        self.encoder = encoder
        # self.prepare = cxx.PrepareBertMasks()
    
    def __call__(
            self,
            inputs: torch.Tensor,
            total_seq_len: int,
            seq_position_info,
            seq_position_info_tensor,
            partition_part_index_tensor,
            partition_part_tensor,
            attention_masks: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_layer_temp:Optional[torch.Tensor] = None,
            intermediate_temp:Optional[torch.Tensor] = None,
            attention_output:Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            pooling_type: PoolingType = PoolingType.FIRST):
            
        attention_masks = create_empty_if_none(attention_masks,(total_seq_len,768),torch.device('cuda'))
        token_type_ids = create_empty_if_none(token_type_ids,(total_seq_len,768),torch.device('cuda'))
        position_ids = create_empty_if_none(position_ids,(total_seq_len,768),torch.device('cuda'))
        inputs = (inputs)
        
        # extended_attention_masks = cxx.Tensor.create_empty()

        # self.prepare(inputs, attention_masks, token_type_ids, position_ids,
        #              extended_attention_masks)

        thread_block_limit = 80
        
        nvtx.range_push("embeddings")
        hidden_cache = self.embeddings(
            inputs,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        nvtx.range_pop()
        
        
        nvtx.range_push("encoder")
        encoder_outputs = self.encoder(
            hidden_states=hidden_cache,
            total_seq_len=total_seq_len,
            thread_block_limit=thread_block_limit,
            seq_position_info=seq_position_info,
            seq_position_info_tensor = seq_position_info_tensor,
            partition_part_index_tensor=partition_part_index_tensor,
            partition_part_tensor=partition_part_tensor,
            output_layer_temp = output_layer_temp,
            intermediate_temp = intermediate_temp,
            attention_output = attention_output,
            )
        nvtx.range_pop()
        return encoder_outputs

    @staticmethod
    def from_torch(model: TorchBertModel,
                   device: Optional[torch.device] = None):
        if device is not None and 'cuda' in device.type and torch.cuda.is_available(
        ):
            model.to(device)
        embeddings = BertEmbeddings.from_torch(model.embeddings)
        encoder = BertEncoder.from_torch(model.encoder)
        return BertModelNoPooler(embeddings, encoder)

class BertModel:
    # @params:
    # pooler is used for turbo backend only
    # config is used for memory optizations
    def __init__(self, model, backend="onnxrt", config=None, pooler=None):
        # TODO type of bertmodel_nopooler is (onnx and torch)
        self.backend = backend
        if backend == "onnxrt":
            self.onnxmodel = model
        elif backend == "turbo":
            self.config = config
            self.bertmodel_nopooler = model
            self.pooler = pooler
            self.backend = "turbo"

    def __call__(self,
                 inputs: torch.Tensor,
                 attention_masks: Optional[torch.Tensor] = None,
                 token_type_ids: Optional[torch.Tensor] = None,
                 position_ids: Optional[torch.Tensor] = None,
                 head_mask: Optional[torch.Tensor] = None,
                 inputs_embeds: Optional[torch.Tensor] = None,
                 output_attentions: Optional[bool] = None,
                 output_hidden_states: Optional[bool] = None,
                 pooling_type: PoolingType = PoolingType.FIRST,
                 pooler_output: Optional[torch.Tensor] = None,):
        
        encoder_outputs = self.bertmodel_nopooler(
            inputs,
            attention_masks,
            token_type_ids,
            position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pooling_type=pooling_type)
        
        sequence_output = encoder_outputs[0]
        self.seq_pool = SequencePool(PoolingMap[pooling_type])
        sequence_pool_output = self.seq_pool(
            input_tensor=sequence_output)
        
        pooler_output = self.pooler(sequence_pool_output, pooler_output)
        return (sequence_output,pooler_output,) + encoder_outputs[1:]


    @staticmethod
    def from_torch(model: TorchBertModel,
                   device: Optional[torch.device] = None,
                   bigbird = False):
        """
        Args:
            model : a PyTorch Bert Model
            device : cpu or GPU
            backend : a string to indicates kernel provides
            Four options. [onnxrt-cpu, onnxrt-gpu, turbo-cpu, turbo-gpu]
            use_memory_opt [bool] whether or not use memory opt for variable length inputs.
        """
        if device is None:
            device = model.device
        
        embeddings = BertEmbeddings.from_torch(model.embeddings)
        encoder = BertEncoder.from_torch(model.encoder)
        bertmodel_nopooler = BertModelNoPooler(embeddings, encoder)
        # pooler = BertPooler.from_torch(model.pooler,bigbird)
        return BertModel(bertmodel_nopooler, "turbo", model.config)