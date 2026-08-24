"""Block-chunk forward pass used to obtain gradients at one scale.

Mirrors the per-scale body of `Infinity.autoregressive_infer_cfg`, but applies
the level embedding out-of-place (`h = param + lvl_emb`, not `+=`) so that
`param` stays a leaf tensor and can receive gradients.

The CrossAttnBlock.forward() signature this relies on is:
    x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn,
    scale_schedule, rope2d_freqs_grid, scale_ind
It does NOT accept `start_layer` or `src`.
"""

import torch


def forward_scale_for_grad(
    infinity,
    param: torch.Tensor,
    si: int,
    scale_schedule: list,
    cond_BD_or_gss: torch.Tensor,
    ca_kv: tuple,
    attn_fn,
    rope2d_freqs_grid: torch.Tensor,
) -> torch.Tensor:
    bs = param.shape[0]
    pt, ph, pw = scale_schedule[si]
    L_si = pt * ph * pw

    lvl_ids = si * torch.ones((bs, L_si), dtype=torch.int, device=param.device)
    lvl_emb = infinity.lvl_embed(lvl_ids).to(param.dtype)
    h = param + lvl_emb   # out-of-place: param stays leaf

    for block_idx, chunk in enumerate(infinity.block_chunks):
        if block_idx > 0 and not infinity.add_lvl_embeding_only_first_block:
            lvl_ids2 = si * torch.ones((bs, h.shape[1]), dtype=torch.int, device=h.device)
            h = h + infinity.lvl_embed(lvl_ids2).to(h.dtype)

        for m in chunk.module:
            h = m(
                x=h,
                cond_BD=cond_BD_or_gss,
                ca_kv=ca_kv,
                attn_bias_or_two_vector=None,
                attn_fn=attn_fn,
                scale_schedule=scale_schedule,
                rope2d_freqs_grid=rope2d_freqs_grid,
                scale_ind=si,
            )
    return h
