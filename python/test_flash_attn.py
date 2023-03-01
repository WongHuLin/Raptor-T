import torch
import transformers
# from transformers import BigBirdModel
from flash_attn.models.bert import BertModel, BertForPreTraining
from flash_attn.models.bert import remap_state_dict
from flash_attn.utils.pretrained import state_dict_from_pretrained
from einops import rearrange, repeat
device = torch.device("cuda")
model_name = "google/bigbird-roberta-base"
model = transformers.BigBirdModel.from_pretrained("google/bigbird-roberta-base").to(device)
config = model.config
config.hidden_act = "gelu_new"
config.use_flash_attn = True
config.fused_bias_fc = False
config.fused_mlp = False
config.fused_dropout_add_ln = False
pretrained_state_dict = remap_state_dict(state_dict_from_pretrained(model_name), config)
dtype = torch.float16
model = BertForPreTraining.from_pretrained(model_name,config).cuda().to(dtype=dtype)


def generate_random_padding_mask(max_seqlen, batch_size, device, mode='random'):
    assert mode in ['full', 'random', 'third', 'split']
    if mode == 'full':
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == 'random':
        lengths = torch.randint(max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device)
    elif mode == 'third':
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    elif mode == 'split':
        lengths0 = torch.randint(min(128, max_seqlen), max_seqlen + 1,
                                 (batch_size // 4 * 3, 1), device=device)
        lengths1 = torch.randint(min(max(1, max_seqlen - 20), 128), min(max_seqlen, 128) + 1,
                                 (batch_size - batch_size // 4 * 3, 1), device=device)
        lengths = torch.cat([lengths0, lengths1], dim=0)
    padding_mask = repeat(torch.arange(max_seqlen, device=device), 's -> b s', b=batch_size) < lengths
    return padding_mask

model.eval()
torch.manual_seed(0)
batch_size = 4
max_seqlen = 4096
seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,), device='cuda')
attention_mask = torch.arange(max_seqlen, device='cuda')[None, :] < seqlens[:, None]
input_ids = torch.randint(0, config.vocab_size, (batch_size, max_seqlen), dtype=torch.long, device='cuda')
# key_padding_mask = generate_random_padding_mask(seqlens, batch_size, device, mode='random')
with torch.no_grad():
    out = model.bert(input_ids, attention_mask=attention_mask)

print("end")