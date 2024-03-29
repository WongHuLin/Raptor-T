from pdb import set_trace
import torch
import os
from transformers import LongformerModel
import json
from transformers.models.longformer.modeling_longformer import LongformerBaseModelOutput


def from_hf_longformer_weight_to_ft(weights_file, layer_num, data_type):
    weights = torch.load(weights_file)
    all_weights = []
    for i in range(0, layer_num):
        # Need to transpose the kernel for torch.nn.Linear
        # q k v kg vg weights and bias should be continuous, required by the ft longformer encoder.
        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.query.weight".format(i)].transpose(0, 1))
        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.key.weight".format(i)].transpose(0, 1))
        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.value.weight".format(i)].transpose(0, 1))
        all_weights.append(
            weights["longformer.encoder.layer.{}.attention.self.key_global.weight".format(i)].transpose(0, 1))
        all_weights.append(
            weights["longformer.encoder.layer.{}.attention.self.value_global.weight".format(i)].transpose(0, 1))

        all_weights.append(
            weights["longformer.encoder.layer.{}.attention.self.query_global.weight".format(i)].transpose(0, 1))

        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.query.bias".format(i)])
        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.key.bias".format(i)])
        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.value.bias".format(i)])
        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.key_global.bias".format(i)])
        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.value_global.bias".format(i)])

        all_weights.append(weights["longformer.encoder.layer.{}.attention.self.query_global.bias".format(i)])

        all_weights.append(
            weights["longformer.encoder.layer.{}.attention.output.dense.weight".format(i)].transpose(0, 1))
        all_weights.append(weights["longformer.encoder.layer.{}.attention.output.dense.bias".format(i)])

        all_weights.append(weights["longformer.encoder.layer.{}.attention.output.LayerNorm.weight".format(i)])
        all_weights.append(weights["longformer.encoder.layer.{}.attention.output.LayerNorm.bias".format(i)])

        all_weights.append(weights["longformer.encoder.layer.{}.intermediate.dense.weight".format(i)].transpose(0, 1))
        all_weights.append(weights["longformer.encoder.layer.{}.intermediate.dense.bias".format(i)])

        all_weights.append(weights["longformer.encoder.layer.{}.output.dense.weight".format(i)].transpose(0, 1))
        all_weights.append(weights["longformer.encoder.layer.{}.output.dense.bias".format(i)])

        all_weights.append(weights["longformer.encoder.layer.{}.output.LayerNorm.weight".format(i)])
        all_weights.append(weights["longformer.encoder.layer.{}.output.LayerNorm.bias".format(i)])

    for i in range(0, len(all_weights)):
        all_weights[i] = all_weights[i].flatten()

    if data_type == "fp16":
        all_weights = torch.cat(all_weights).type(torch.float16)
    elif data_type == "bf16":
        all_weights = torch.cat(all_weights).type(torch.bfloat16)
    elif data_type == "fp32":
        all_weights = torch.cat(all_weights).type(torch.float32)
    return all_weights.contiguous()


class FTLongformerEncoder(torch.nn.Module):
    def __init__(self, weights_file, layer_num, head_num, size_per_head,
                 intermediate_size, local_attn_window_size,
                 max_global_token_num, batch_size, seq_len,
                 attn_scaler, ft_longformer_lib, data_type='fp32', hf_plugin_mode=False):
        super().__init__()
        self.data_type = data_type
        assert seq_len % local_attn_window_size == 0 and seq_len / \
            local_attn_window_size >= 2, "seq_len need to be multiple of local_attn_window_size and at least 2 times big."

        self.hf_plugin_mode = hf_plugin_mode
        all_weight = from_hf_longformer_weight_to_ft(weights_file, layer_num, data_type)
        self.all_weight = all_weight.cuda()
        torch.classes.load_library(ft_longformer_lib)
        self.ft_encoder = torch.classes.FasterTransformer.LongformerEncoder(layer_num, head_num * size_per_head,head_num, size_per_head,intermediate_size, local_attn_window_size,max_global_token_num, batch_size, seq_len,attn_scaler)

    def set_hf_plugin_mode(self, is_plugin):
        self.hf_plugin_mode = is_plugin

    def forward(self, *args, **kwargs):
        encoder_in = args[0]

        if self.hf_plugin_mode:
            # In this mode, assume that HuggingFace's LongformerModel.encoder has been
            # substituted to this class's instance
            extended_attention_mask = kwargs['attention_mask']
            local_attn_mask = torch.zeros_like(extended_attention_mask)
            local_attn_mask[extended_attention_mask > -10000.] = 1.0
            global_attn_mask = torch.zeros_like(extended_attention_mask)
            global_attn_mask[extended_attention_mask > 0.] = 1.0
            output = self.ft_encoder.forward(encoder_in, local_attn_mask, global_attn_mask, self.all_weight, 0)
            return LongformerBaseModelOutput(
                last_hidden_state=output,
                hidden_states=None,
                attentions=None,
                global_attentions=None,
            )
        else:
            local_attn_mask = args[1]
            global_attn_mask = args[2]
            return self.ft_encoder.forward(encoder_in, local_attn_mask, global_attn_mask, self.all_weight, 0)

def build_hf_longformer(model_dir):
    hf_longformer = LongformerModel.from_pretrained(model_dir)
    hf_longformer.cuda()
    hf_longformer.eval()
    return hf_longformer

def build_ft_longformer(hf_model_dir, layer_num, head_num, size_per_head,
                        intermediate_size, local_attn_window_size,
                        max_global_token_num, batch_size, seq_len,
                        attn_scaler, ft_longformer_lib, data_type):
    weights_file = os.path.join(hf_model_dir, 'pytorch_model.bin')
    ft_encoder = FTLongformerEncoder(weights_file, layer_num, head_num, size_per_head,
                                     intermediate_size, local_attn_window_size,
                                     max_global_token_num, batch_size, seq_len,
                                     attn_scaler, ft_longformer_lib, data_type)
    ft_longformer = build_hf_longformer(hf_model_dir)
    if data_type == 'fp16':
        ft_longformer = ft_longformer.half()
    elif data_type == 'bf16':
        ft_longformer = ft_longformer.bfloat16()
    ft_longformer.cuda()
    ft_longformer.eval()
    ft_encoder.set_hf_plugin_mode(True)
    ft_longformer.encoder = ft_encoder
    return ft_longformer


def parse_from_config(model_dir):
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    layer_num = config['num_hidden_layers']
    hidden_size = config['hidden_size']
    head_num = config['num_attention_heads']
    size_per_head = hidden_size // head_num
    intermediate_size = config['intermediate_size']
    # assume all local attn window are same size. TODO: Improve later
    local_attn_window_size = config['attention_window'][0]
    attn_scaler = 1.0 / (size_per_head ** 0.5)
    return (layer_num, hidden_size, head_num, size_per_head,
            intermediate_size, local_attn_window_size, attn_scaler)

