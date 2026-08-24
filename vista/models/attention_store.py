"""Read-only cross- and self-attention capture, for analysis and figures.

Distinct from `vista.attention`: those hooks rebuild attention differentiably so
gradients can flow back to the features being steered, whereas these simply
record detached maps per scale for inspection. Enabled via
`InfinityWrapper.generate(extract_attention=True)`.
"""

import math
import torch
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image

class InfinityAttentionStore:
    def __init__(self):
        self.cross_store = {} # {res: [attn_map]}
        self.self_store = {}
        
    def reset(self):
        self.cross_store = {}
        self.self_store = {}

    def add_cross(self, res, attn):
        if res not in self.cross_store:
            self.cross_store[res]= []
        self.cross_store[res].append(attn.detach())
    
    # def add_self(self, res, attn, Lk):
    #     if res not in self.self_store:
    #         self.self_store[res] = []
    #     self.self_store[res].append((attn, Lk))

    def add_self(self, res, attn, Lk_actual, current_tokens, history_tokens):
        if res not in self.self_store:
            self.self_store[res] = []
        self.self_store[res].append({
            'attn': attn,
            'Lk_actual': Lk_actual,
            'current_tokens': current_tokens,
            'history_tokens': history_tokens,
        })

    def get_average_cross(self):
        return {
            res: torch.stack(attns).mean(dim=0)
            for res, attns in self.cross_store.items()
        }

    # def get_average_self(self):
    #     return {
    #         res: torch.stack(attns).mean(dim=0)
    #         for res, attns in self.self_store.items()
    #     }
    def get_self(self):
        return self.self_store

def cross_attn_hook(module, inputs, output, store):
    q, ca_kv= inputs
    # Skip SOS/text pooling
    if q is None:
        return

    kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
    B, Lq, C= q.shape
    H= module.num_heads
    d= module.head_dim

    # only keep square spatial resolutions
    side= int(math.sqrt(Lq))
    if side*side!=Lq:# or side<2:
        return

    q_compact = module.mat_q(q).view(B, Lq, H, d).permute(0, 2, 1, 3)
    # shape: (B, H, Lq, d)

    KV = F.linear(
        kv_compact,
        weight=module.mat_kv.weight,
        bias=torch.cat((module.zero_k_bias, module.v_bias)),
    ).view(-1, 2, H, d)

    K= KV[:, 0]
    K= K.view(B, max_seqlen_k, H, d).permute(0, 2, 1, 3)
    # shape: (B, H, Lk, d)

    attn= torch.matmul(q_compact, K.transpose(-2, -1))
    attn= attn*module.scale
    attn= attn.softmax(dim= -1)

    attn= attn.mean(dim=1)  # (B, Lq, Lk)
    store.add_cross(side, attn)
    
    
def self_attn_hook(module, inputs, output, store):
    x = inputs[0]
    B, Lq, C = x.shape

    side = int(math.sqrt(Lq))
    if side * side != Lq:
        return

    if not (hasattr(module, 'cached_k') and module.cached_k is not None):
        return
    if not (hasattr(module, 'scale_key_boundaries') and module.scale_key_boundaries):
        return

    boundary = module.scale_key_boundaries[-1]
    current_tokens  = boundary['current_tokens']   # exact tokens this scale added
    history_tokens  = boundary['history_tokens']   # compressed previous scales
    Lk_before       = boundary['Lk_before']        # total before compression

    H = module.num_heads
    d = module.head_dim

    with torch.no_grad():
        qkv = F.linear(
            x,
            module.mat_qkv.weight,
            torch.cat((module.q_bias, module.zero_k_bias, module.v_bias))
        ).view(B, Lq, 3, H, d)

        q = qkv[:, :, 0].float()
        q_t = q.permute(0, 2, 1, 3)               # (B, H, Lq, d)

        if module.cos_attn:
            scale_mul = module.scale_mul_1H11.clamp_max(module.max_scale_mul).exp()
            scale_mul = scale_mul.view(1, H, 1, 1)
            q_t = F.normalize(q_t, dim=-1, eps=1e-12).mul(scale_mul)

        k_t = module.cached_k.float()              # (B, H, Lk_after_compression, d)
        Lk_actual = k_t.shape[2]

        attn = torch.matmul(q_t, k_t.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        attn = attn.mean(dim=1)                    # (B, Lq, Lk_actual)

    store.add_self(side, attn.cpu(), Lk_actual, current_tokens, history_tokens)
    

def aggregate_image_attention(attn, res):
    """
    attn: (Lq, Lk)
    returns: (res, res)
    """
    pixel_map = attn.mean(dim=-1) 
    return pixel_map.view(res, res)


def aggregate_token_attention(attn, token_idx, res):
    """
    attn: (Lq, Lk)
    """
    token_map= attn[:, token_idx]
    return token_map.view(res, res)


def register_infinity_cross_attention(model, store):
    hooks = []
    for module in model.modules():
        if module.__class__.__name__== "CrossAttention":
            h= module.register_forward_hook(
                lambda m, i, o: cross_attn_hook(m, i, o, store)
            )
            hooks.append(h)
    return hooks

def register_infinity_self_attention(model, store):
    hooks = []
    for module in model.modules():
        if module.__class__.__name__ == "SelfAttention":
            h = module.register_forward_hook(
                lambda m, i, o: self_attn_hook(m, i, o, store)
            )
            hooks.append(h)
    return hooks

def save_heatmap(heatmap: torch.Tensor, save_path: str,size: int= 512):
    """
    heatmap: (H, W) torch tensor
    save_path: full path to .png
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    hm= heatmap.detach().float()
    hm= hm-hm.min()
    hm= hm/(hm.max()+1e-8)
    hm= (hm*255).cpu().numpy().astype(np.uint8)

    img= Image.fromarray(hm, mode="L")
    img= img.resize((size, size), Image.BILINEAR)
    img.save(save_path)


def token_attention_map(attn, token_idx, res):
    return attn[:, token_idx].view(res, res)

def image_attention_map(attn, res):
    """
    attn: (B, Lq, Lk)
    returns: (res, res)
    """
    attn= attn.mean(dim=0)        # (Lq, Lk)
    pixel_map= attn.mean(dim=-1)  # (Lq,)
    if pixel_map.numel()!=res*res:
        return None
    return pixel_map.view(res, res)

def get_scale_token_info(current_res, scale_resolutions):
    scales = []
    start= 0
    for s in scale_resolutions:
        num_tokens= s*s
        end= start+num_tokens
        scales.append((s, start, end))
        start= end
        if s==current_res:
            break
    return scales