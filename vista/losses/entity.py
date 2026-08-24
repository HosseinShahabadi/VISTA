"""Entity-level objectives (L_entity).

Where the binding terms operate on (modifier, noun) pairs, these operate on the
set of *concepts* in the prompt: every noun is pushed to occupy its own region
of the attention map, irrespective of its attributes.

`build_concept_ids_list` collects that concept set out of a VISTA config, so the
same term applies whether a prompt was parsed into attribute pairs, spatial
pairs, or numerical pairs.

Reference
---------
A-STAR  Agarwal et al., "A-STAR: Test-time Attention Segregation and Retention
        for Text-to-image Synthesis" (ICCV 2023, arXiv:2306.14544).
        `astar_segregation_loss` is L_seg (Eq. 2); `astar_retention_loss` is the
        per-concept IoU term of L_ret (Eq. 3), which needs the previous step's
        binarised map from `vista.losses.utils.bbox_binarize`. The paper does
        not specify the high-activation threshold for that box, so it is exposed
        as the `quantile` argument (default: top 20%).
"""

import torch
import torch.nn.functional as F


def astar_segregation_loss(attn: torch.Tensor, concept_ids_list: list) -> torch.Tensor:
    a  = attn[0]            # [Lq, Lk]
    Lk = a.shape[-1]

    maps = []
    for ids in concept_ids_list:
        ids = [t for t in ids if t < Lk]
        if not ids:
            continue
        maps.append(a[:, ids].mean(dim=-1))   # [Lq], non-negative (a is post-softmax)

    if len(maps) < 2:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)

    total, n_pairs = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            inter = torch.minimum(maps[i], maps[j]).sum()
            union = (maps[i] + maps[j]).sum() + 1e-8
            total = total + inter / union
            n_pairs += 1

    return total / n_pairs if n_pairs > 0 else torch.tensor(0.0, device=attn.device, dtype=attn.dtype)


def build_concept_ids_list(vista_cfg: dict) -> list:
    """
    Builds the `concept_ids_list` arg for astar_segregation_loss out of
    your existing vista_cfg structures. A-STAR pairs every CONCEPT
    (noun) against every other concept -- not attribute-noun pairs --
    so this collects every distinct noun across `pairs`, `spatial_pairs`,
    and `numerical_pairs`, deduped by surface text (case-insensitive) so
    a noun appearing in more than one structure contributes once.
    """
    seen = {}   # noun text (lower) -> token ids

    for pair in vista_cfg.get('pairs', []):
        noun, ids = pair.get('noun'), pair.get('noun_t5_ids', [])
        if noun and ids and noun.lower() not in seen:
            seen[noun.lower()] = ids

    for pair in vista_cfg.get('spatial_pairs', []):
        for noun_key, ids_key in (('noun_a', 'noun_a_t5_ids'), ('noun_b', 'noun_b_t5_ids')):
            noun, ids = pair.get(noun_key), pair.get(ids_key, [])
            if noun and ids and noun.lower() not in seen:
                seen[noun.lower()] = ids

    for pair in vista_cfg.get('numerical_pairs', []):
        noun, ids = pair.get('noun'), pair.get('noun_t5_ids', [])
        if noun and ids and noun.lower() not in seen:
            seen[noun.lower()] = ids

    return list(seen.values())


def astar_retention_loss(current_map_flat: torch.Tensor, prev_mask_flat: torch.Tensor) -> torch.Tensor:
    """IoU-consistency term for ONE concept. current_map_flat: [Lq] raw
    (non-negative) attention map at the current step. prev_mask_flat:
    [Lq] binary mask (same Lq) from `bbox_binarize` at the previous step."""
    inter = torch.minimum(current_map_flat, prev_mask_flat).sum()
    union = (current_map_flat + prev_mask_flat).sum() + 1e-8
    return 1.0 - inter / union


def attention_presence_balance_loss(attn, pairs, lambda_p=1.0, lambda_b=0.5):
    a = attn[0]          # [Lq, Lk]
    Lk = a.shape[1]
    maps = []
    for pair in pairs:
        for key in ('noun_a_t5_ids', 'noun_b_t5_ids'):
            ids = [t for t in pair.get(key, []) if t < Lk]
            if ids:
                maps.append(a[:, ids].mean(dim=-1))  # [Lq]
    if not maps:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
    
    total = torch.zeros(1, device=attn.device, dtype=attn.dtype)
    # Presence: minimise variance of each noun's attention
    variances = []
    for m in maps:
        p = F.softmax(m, dim=0)
        cx = (p * torch.arange(len(p), device=attn.device, dtype=attn.dtype)).sum()
        var = (p * (torch.arange(len(p), device=attn.device, dtype=attn.dtype) - cx)**2).sum()
        variances.append(var)
        total = total + lambda_p * var
    # Balance: minimise |σ²_A - σ²_B| between paired nouns
    if len(variances) >= 2:
        for i in range(0, len(variances)-1, 2):
            total = total + lambda_b * (variances[i] - variances[i+1]).abs()
    return total / len(maps)
