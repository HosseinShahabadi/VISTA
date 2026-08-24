"""Differentiable cross-attention capture.

`autoregressive_infer_cfg` runs under @torch.no_grad(), so the capture hook has
to rebuild the attention probabilities itself: any tensor op performed outside
an `enable_grad` region strips grad_fn even when its inputs have
requires_grad=True. `get_averaged()` must therefore be called inside
`with torch.enable_grad()`.

Only the query projection is differentiated. Keys are recomputed under
`torch.no_grad()` (stop-gradient on K): the text stream is fixed at test time,
so gradients through it would only add cost and noise.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


class VISTAAttentionCapture:
    def __init__(self):
        self.maps: dict = {}

    def reset(self):
        self.maps.clear()

    def add(self, res: int, attn: torch.Tensor):
        self.maps.setdefault(res, []).append(attn)

    def get_averaged(self, res: int) -> Optional[torch.Tensor]:
        raw = self.maps.get(res)
        if not raw:
            return None
        return torch.stack(raw).mean(dim=0)   # keeps grad_fn


def _cross_attention_hook(module, inputs, output, capture: VISTAAttentionCapture):
    q, ca_kv = inputs
    if q is None:
        return

    kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
    B, Lq, C = q.shape
    H = module.num_heads
    d = module.head_dim

    side = int(math.sqrt(Lq))
    if side * side != Lq:
        return

    q_proj = module.mat_q(q).view(B, Lq, H, d).permute(0, 2, 1, 3)

    with torch.no_grad():
        KV = F.linear(
            kv_compact,
            weight=module.mat_kv.weight,
            bias=torch.cat((module.zero_k_bias, module.v_bias)),
        ).view(-1, 2, H, d)
        K = KV[:, 0].view(B, max_seqlen_k, H, d).permute(0, 2, 1, 3)

    scale = module.scale if hasattr(module, 'scale') else d ** -0.5
    attn = torch.matmul(q_proj, K.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1).mean(dim=1)   # [B, Lq, Lk] — no .detach()

    capture.add(side, attn)


def register_attention_hooks(infinity, capture: VISTAAttentionCapture) -> list:
    handles = []
    for module in infinity.modules():
        if module.__class__.__name__ == 'CrossAttention':
            h = module.register_forward_hook(
                lambda m, i, o, cap=capture: _cross_attention_hook(m, i, o, cap)
            )
            handles.append(h)
    return handles
