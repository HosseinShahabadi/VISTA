"""Attribute-binding objectives (L_bind).

These pull the cross-attention map of a modifier token towards the map of the
noun it modifies, and push it away from every other noun in the prompt.

`conform_contrastive_loss` is the default binding term used in the paper; the
others are ablation / baseline variants.

References
----------
SynGen   Rassin et al., "Linguistic Binding in Diffusion Models" (NeurIPS 2023).
CONFORM  Meral et al., "CONFORM: Contrast is All You Need for High-Fidelity
         Text-to-Image Diffusion Models" (CVPR 2024, arXiv:2312.06059).
D&B      Li et al., "Divide & Bind Your Attention for Improved Generative
         Semantic Nursing" (BMVC 2023, arXiv:2307.10864).
SupCon   Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020).

Deviations from the official CONFORM implementation, flagged explicitly:
  * no previous-timestep attention maps are injected as extra positives (there
    is a single forward pass per scale here, not a diffusion trajectory);
  * the multi-positive NT-Xent objective is reimplemented directly instead of
    calling pytorch_metric_learning, so exact numerics may differ marginally;
  * `attn` is already softmax-normalised over the text-token axis by the VISTA
    capture hook, which plays the role of their `softmax_normalize` step, so it
    is not reapplied here.
"""

import torch
import torch.nn.functional as F

from vista.losses.utils import smooth_map


def syngen_loss(
    attn: torch.Tensor,   # [B, Lq, Lk] with grad_fn
    pairs: list,
    contrastive: bool = True,
    lambda_c: float = 0.3,
) -> torch.Tensor:
    a  = attn[0]
    Lk = a.shape[-1]

    total   = torch.zeros(1, device=attn.device, dtype=attn.dtype)
    n_valid = 0

    for i, pair in enumerate(pairs):
        adj_ids  = [t for t in pair.get('adj_t5_ids',  []) if t < Lk]
        noun_ids = [t for t in pair.get('noun_t5_ids', []) if t < Lk]
        if not adj_ids or not noun_ids:
            continue

        adj_map  = a[:, adj_ids ].mean(dim=-1)
        noun_map = a[:, noun_ids].mean(dim=-1)

        binding = F.cosine_similarity(adj_map.unsqueeze(0), noun_map.unsqueeze(0))
        total = total - binding

        if contrastive:
            for j, other in enumerate(pairs):
                if j == i:
                    continue
                other_ids = [t for t in other.get('noun_t5_ids', []) if t < Lk]
                if not other_ids:
                    continue
                other_map = a[:, other_ids].mean(dim=-1)
                excl = F.cosine_similarity(adj_map.unsqueeze(0), other_map.unsqueeze(0))
                total = total + lambda_c * excl

        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device)

    return total / n_valid


def kl_binding_loss(attn, pairs, lambda_c=0.3, temp=0.02):
    a  = attn[0]   # [Lq, Lk]
    Lk = a.shape[-1]
    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0

    for i, pair in enumerate(pairs):
        adj_ids  = [t for t in pair.get('adj_t5_ids',  []) if t < Lk]
        noun_ids = [t for t in pair.get('noun_t5_ids', []) if t < Lk]
        if not adj_ids or not noun_ids:
            continue

        adj_map  = a[:, adj_ids ].mean(dim=-1)   # [Lq]
        noun_map = a[:, noun_ids].mean(dim=-1)

        # Normalise to distributions
        p = F.softmax(adj_map  / temp, dim=0)    # [Lq]
        q = F.softmax(noun_map / temp, dim=0)    # [Lq]

        # KL(p || q) — minimise to align adj toward noun
        kl = (p * (p.log() - (q + 1e-8).log())).sum()
        total = total + kl

        # Contrastive: maximise KL from other nouns
        if lambda_c > 0:
            for j, other in enumerate(pairs):
                if j == i:
                    continue
                other_ids = [t for t in other.get('noun_t5_ids', []) if t < Lk]
                if not other_ids:
                    continue
                other_map = a[:, other_ids].mean(dim=-1)
                q_other = F.softmax(other_map / temp, dim=0)
                kl_other = (p * (p.log() - (q_other + 1e-8).log())).sum()
                total = total - lambda_c * kl_other   # push apart

        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device)
    return total / n_valid


def jsd_binding_loss(attn, pairs, lambda_c=0.3, temp=1.0):
    """
    Jensen-Shannon divergence binding loss.

    Same goal as kl_binding_loss (align adj attention distribution with
    noun attention distribution) but JSD is symmetric and bounded in
    [0, ln 2], unlike KL which is asymmetric and can spike when q has
    near-zero mass where p doesn't. Generally a smoother, more stable
    training signal than raw KL.

    JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m),  m = 0.5 * (p + q)
    """
    a  = attn[0]   # [Lq, Lk]
    Lk = a.shape[-1]
    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0

    for i, pair in enumerate(pairs):
        adj_ids  = [t for t in pair.get('adj_t5_ids',  []) if t < Lk]
        noun_ids = [t for t in pair.get('noun_t5_ids', []) if t < Lk]
        if not adj_ids or not noun_ids:
            continue

        adj_map  = a[:, adj_ids ].mean(dim=-1)   # [Lq]
        noun_map = a[:, noun_ids].mean(dim=-1)

        p = F.softmax(adj_map  / temp, dim=0)
        q = F.softmax(noun_map / temp, dim=0)
        m = 0.5 * (p + q)

        jsd = 0.5 * (p * ((p + 1e-8).log() - (m + 1e-8).log())).sum() \
            + 0.5 * (q * ((q + 1e-8).log() - (m + 1e-8).log())).sum()
        total = total + jsd

        if lambda_c > 0:
            for j, other in enumerate(pairs):
                if j == i:
                    continue
                other_ids = [t for t in other.get('noun_t5_ids', []) if t < Lk]
                if not other_ids:
                    continue
                other_map = a[:, other_ids].mean(dim=-1)
                q_other = F.softmax(other_map / temp, dim=0)
                m_other = 0.5 * (p + q_other)
                jsd_other = 0.5 * (p * ((p + 1e-8).log() - (m_other + 1e-8).log())).sum() \
                          + 0.5 * (q_other * ((q_other + 1e-8).log() - (m_other + 1e-8).log())).sum()
                total = total - lambda_c * jsd_other   # push apart from wrong nouns

        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device)
    return total / n_valid


def composite_binding_loss(attn, pairs, alpha=0.5, lambda_c=0.3, temp=1.0):
    l_cos = syngen_loss(attn, pairs, contrastive=True,  lambda_c=lambda_c)
    l_kl  = kl_binding_loss(attn, pairs, lambda_c=lambda_c, temp=temp)
    return alpha * l_cos + (1 - alpha) * l_kl


def peak_alignment_loss(attn, pairs, lambda_c=0.3):
    a  = attn[0]
    Lq, Lk = a.shape
    side = int(Lq ** 0.5)
    if side * side != Lq:
        return torch.tensor(0.0, device=attn.device)

    # 2D grid coordinates for each spatial token
    ys = torch.arange(side, device=attn.device, dtype=attn.dtype)
    xs = torch.arange(side, device=attn.device, dtype=attn.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid_y = grid_y.reshape(-1)   # [Lq]
    grid_x = grid_x.reshape(-1)   # [Lq]

    def soft_centroid(map_vec):
        # map_vec: [Lq], unnormalised
        weights = F.softmax(map_vec * 10, dim=0)   # sharpen
        cy = (weights * grid_y).sum()
        cx = (weights * grid_x).sum()
        return torch.stack([cy, cx])   # [2]

    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0

    for i, pair in enumerate(pairs):
        adj_ids  = [t for t in pair.get('adj_t5_ids',  []) if t < Lk]
        noun_ids = [t for t in pair.get('noun_t5_ids', []) if t < Lk]
        if not adj_ids or not noun_ids:
            continue

        adj_map  = a[:, adj_ids ].mean(dim=-1)
        noun_map = a[:, noun_ids].mean(dim=-1)

        adj_center  = soft_centroid(adj_map)
        noun_center = soft_centroid(noun_map)

        # L2 distance between centroids — minimise to align peaks
        dist = ((adj_center - noun_center) ** 2).sum()
        total = total + dist

        if lambda_c > 0:
            for j, other in enumerate(pairs):
                if j == i:
                    continue
                other_ids = [t for t in other.get('noun_t5_ids', []) if t < Lk]
                if not other_ids:
                    continue
                other_map    = a[:, other_ids].mean(dim=-1)
                other_center = soft_centroid(other_map)
                dist_other   = ((adj_center - other_center) ** 2).sum()
                total = total - lambda_c * dist_other   # push apart

        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device)
    return total / n_valid


def exclusive_binding_loss(attn, pairs, temp=0.1):
    a  = attn[0]
    Lk = a.shape[-1]
    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0

    # Collect all valid noun maps once
    noun_maps = []
    for pair in pairs:
        noun_ids = [t for t in pair.get('noun_t5_ids', []) if t < Lk]
        if noun_ids:
            noun_maps.append(a[:, noun_ids].mean(dim=-1))   # [Lq]

    if len(noun_maps) < 2:
        # Fall back to cosine for single-pair case
        return syngen_loss(attn, pairs)

    noun_stack = torch.stack(noun_maps, dim=0)   # [P, Lq]

    for i, pair in enumerate(pairs):
        adj_ids = [t for t in pair.get('adj_t5_ids', []) if t < Lk]
        if not adj_ids:
            continue
        adj_map = a[:, adj_ids].mean(dim=-1)   # [Lq]

        # Cosine similarity of adj to every noun
        adj_norm  = F.normalize(adj_map.unsqueeze(0), dim=-1)       # [1, Lq]
        noun_norm = F.normalize(noun_stack, dim=-1)                  # [P, Lq]
        sims = (adj_norm * noun_norm).sum(dim=-1) / temp             # [P]

        # Cross-entropy: maximise prob of correct noun
        log_probs = F.log_softmax(sims, dim=0)
        total = total - log_probs[i]
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device)
    return total / n_valid


def contrastive_binding_loss(attn, pairs, tau=0.2):
    """
    Supervised contrastive loss (Khosla et al., NeurIPS 2020)
    applied to cross-attention maps.
    
    = -cos(adj_i, noun_i) / tau 
      + log( sum_j exp(cos(adj_i, noun_j) / tau) )
    
    Both terms use F.normalize so they're on the same scale.
    tau controls competition sharpness (0.1 = sharp, from exclusive_binding_loss).
    """
    a  = attn[0]
    Lk = a.shape[-1]
    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0

    noun_maps = []
    valid_pairs = []
    for pair in pairs:
        adj_ids  = [t for t in pair.get('adj_t5_ids',  []) if t < Lk]
        noun_ids = [t for t in pair.get('noun_t5_ids', []) if t < Lk]
        if not adj_ids or not noun_ids:
            continue
        noun_maps.append(
            F.normalize(a[:, noun_ids].mean(dim=-1), dim=0)  # [Lq]
        )
        valid_pairs.append(adj_ids)

    if not valid_pairs:
        return torch.tensor(0.0, device=attn.device)

    noun_stack = torch.stack(noun_maps, dim=0)   # [P, Lq], all normalized

    if len(valid_pairs) == 1:
        # no competition — pure alignment only
        adj_norm  = F.normalize(a[:, valid_pairs[0]].mean(dim=-1), dim=0)
        return -(adj_norm * noun_stack[0]).sum()   # cosine, already normalized

    for i, adj_ids in enumerate(valid_pairs):
        adj_norm = F.normalize(a[:, adj_ids].mean(dim=-1), dim=0)   # [Lq]

        # cosine similarities to ALL nouns, same normalization for both terms
        sims = (noun_stack * adj_norm.unsqueeze(0)).sum(dim=-1)      # [P], in [-1,1]

        # supervised contrastive: -sim(adj, noun_i)/tau + log sum_j exp(sim/tau)
        loss_i = -sims[i] / tau + torch.logsumexp(sims / tau, dim=0)

        total  = total + loss_i
        n_valid += 1

    return total / n_valid


def conform_contrastive_loss(
    attn: torch.Tensor,
    pairs: list,
    temperature: float = 0.07,
    do_smoothing: bool = True,
    smoothing_kernel_size: int = 3,
    smoothing_sigma: float = 0.5,
) -> torch.Tensor:
    a  = attn[0]            # [Lq, Lk]
    Lk = a.shape[-1]

    side = int(round(a.shape[0] ** 0.5))
    can_reshape = (side * side == a.shape[0])

    embeddings, labels = [], []

    for i, pair in enumerate(pairs):
        adj_ids  = [t for t in pair.get('adj_t5_ids',  []) if t < Lk]
        noun_ids = [t for t in pair.get('noun_t5_ids', []) if t < Lk]
        if not adj_ids or not noun_ids:
            continue

        for ids in (adj_ids, noun_ids):
            emb = a[:, ids].mean(dim=-1)     # [Lq] -- this "object"'s spatial map

            if do_smoothing and can_reshape:
                emb = smooth_map(
                    emb.view(side, side),
                    kernel_size=smoothing_kernel_size,
                    sigma=smoothing_sigma,
                ).reshape(-1)

            emb = F.normalize(emb, dim=0, eps=1e-8)    # cosine-ready
            embeddings.append(emb)
            labels.append(i)   # pair index = object/group id -- attr+noun share a label

    if len(embeddings) < 2:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)

    E = torch.stack(embeddings, dim=0)                 # [N, Lq]
    y = torch.tensor(labels, device=attn.device)

    sim = (E @ E.T) / temperature                      # [N, N] cosine sims (E already L2-normed)
    N = sim.shape[0]
    self_mask = torch.eye(N, dtype=torch.bool, device=attn.device)
    sim = sim.masked_fill(self_mask, float('-inf'))

    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & ~self_mask
    has_pos  = pos_mask.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    # NOTE: do NOT do `(log_prob * pos_mask).sum(...)` -- pos_mask is False
    # on the diagonal where log_prob is -inf (self-similarity, masked out
    # above), and -inf * 0.0 == nan in IEEE float arithmetic, which then
    # poisons the whole row via nan + x == nan. torch.where sidesteps this
    # by selecting values directly instead of multiplying by 0/1.
    zero = torch.zeros_like(log_prob)
    pos_log_prob = torch.where(pos_mask, log_prob, zero).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)

    loss = -pos_log_prob[has_pos].mean()
    return loss


def divide_and_bind_loss(attn, pairs, temp=0.02):
    """
    Li, Keuper, Zhang, Khoreva — "Divide & Bind Your Attention for
    Improved Generative Semantic Nursing" (arXiv:2307.10864, BMVC 2023).

    L_bind = JSD( softmax(rescale(A_r)) || softmax(A_s) )      (Eq. 5)

    A_s : raw, pre-softmax attention map of the OBJECT/noun token.
    A_r : raw, pre-softmax attention map of the ATTRIBUTE/adj token.

    Per Sec 4.2, A_r is min-max rescaled into A_s's value range
    BEFORE the spatial softmax — raw adj-token attention runs
    systematically lower than noun-token attention, so without this
    step the softmax distributions aren't comparable.

    No contrastive/negative term: the paper relies on a separate
    TV-based attendance loss (L_attend) for cross-object separation,
    not on this loss. This function intentionally mirrors that —
    purely a positive alignment term, one pair at a time.
    """
    a  = attn[0]            # [Lq, Lk]
    Lk = a.shape[-1]
    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0

    for pair in pairs:
        adj_ids  = [t for t in pair.get('adj_t5_ids',  []) if t < Lk]
        noun_ids = [t for t in pair.get('noun_t5_ids', []) if t < Lk]
        if not adj_ids or not noun_ids:
            continue

        adj_map  = a[:, adj_ids ].mean(dim=-1)   # [Lq] raw, pre-softmax
        noun_map = a[:, noun_ids].mean(dim=-1)   # [Lq] raw, pre-softmax

        # ── rescale adj map into noun map's value range (Sec 4.2) ──
        s_min, s_max = noun_map.min(), noun_map.max()
        r_min, r_max = adj_map.min(),  adj_map.max()
        adj_rescaled = (adj_map - r_min) / (r_max - r_min + 1e-8)
        adj_rescaled = adj_rescaled * (s_max - s_min) + s_min

        # ── softmax-normalise both into spatial probability dists ──
        p_r = F.softmax(adj_rescaled / temp, dim=0)   # attribute, "Ã_r"
        p_s = F.softmax(noun_map     / temp, dim=0)   # object,    "Ã_s"

        # ── Jensen-Shannon divergence (symmetric) ───────────────────
        m = 0.5 * (p_r + p_s)
        jsd = 0.5 * (p_r * ((p_r + 1e-8).log() - (m + 1e-8).log())).sum() \
            + 0.5 * (p_s * ((p_s + 1e-8).log() - (m + 1e-8).log())).sum()

        total   = total + jsd
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device)
    return total / n_valid


def overlap_coefficient_loss(attn, pairs, lambda_c=0.3, sharpen=1.0):
    """
    Distributional overlap (Bhattacharyya-coefficient-style) binding loss.

    Cosine similarity (syngen_loss) compares the *shape*/direction of the
    adj and noun attention vectors and can score highly even when their
    probability mass sits in non-overlapping spatial regions (e.g. two
    similarly-peaked but spatially offset maps). This loss instead
    measures actual probability mass overlap directly:

        overlap(p, q) = sum( sqrt(p_i * q_i) )   in [0, 1]

    which is 1 only when p and q place mass in the same locations.
    Minimise -overlap to pull adj and noun attention into the same
    spatial support.
    """
    a  = attn[0]   # [Lq, Lk]
    Lk = a.shape[-1]
    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0

    for i, pair in enumerate(pairs):
        adj_ids  = [t for t in pair.get('adj_t5_ids',  []) if t < Lk]
        noun_ids = [t for t in pair.get('noun_t5_ids', []) if t < Lk]
        if not adj_ids or not noun_ids:
            continue

        adj_map  = a[:, adj_ids ].mean(dim=-1)
        noun_map = a[:, noun_ids].mean(dim=-1)

        p = F.softmax(adj_map  * sharpen, dim=0)
        q = F.softmax(noun_map * sharpen, dim=0)

        overlap = (p * q).clamp(min=0).sqrt().sum()   # Bhattacharyya coefficient
        total = total - overlap   # maximise overlap -> minimise negative overlap

        if lambda_c > 0:
            for j, other in enumerate(pairs):
                if j == i:
                    continue
                other_ids = [t for t in other.get('noun_t5_ids', []) if t < Lk]
                if not other_ids:
                    continue
                other_map = a[:, other_ids].mean(dim=-1)
                q_other = F.softmax(other_map * sharpen, dim=0)
                overlap_other = (p * q_other).clamp(min=0).sqrt().sum()
                total = total + lambda_c * overlap_other   # minimise overlap with wrong noun

        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device)
    return total / n_valid


def focal_binding_loss(attn, pairs, lambda_c=0.3, gamma=2.0):
    """
    Focal-weighted cosine binding loss.

    Identical cosine-similarity core to syngen_loss, but each pair's
    contribution is reweighted by (1 - binding)^gamma, focal-loss style.
    Well-bound pairs (binding close to 1) contribute almost nothing;
    badly-bound pairs (binding close to 0 or negative) dominate the
    gradient. Useful for multi-pair prompts where a couple of
    adj-noun pairs are already well aligned and you don't want their
    near-zero gradients diluting the signal for the pairs that still
    need correcting.

    gamma=0 reduces to plain syngen_loss (no reweighting).
    Higher gamma -> sharper focus on the worst-bound pairs.
    """
    a  = attn[0]
    Lk = a.shape[-1]
    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0

    for i, pair in enumerate(pairs):
        adj_ids  = [t for t in pair.get('adj_t5_ids',  []) if t < Lk]
        noun_ids = [t for t in pair.get('noun_t5_ids', []) if t < Lk]
        if not adj_ids or not noun_ids:
            continue

        adj_map  = a[:, adj_ids ].mean(dim=-1)
        noun_map = a[:, noun_ids].mean(dim=-1)

        binding = F.cosine_similarity(adj_map.unsqueeze(0), noun_map.unsqueeze(0))

        # focal weight: detached so it modulates gradient magnitude only,
        # not the gradient direction of `binding` itself
        focal_weight = (1.0 - binding).detach().clamp(min=0.0) ** gamma
        total = total - focal_weight * binding

        if lambda_c > 0:
            for j, other in enumerate(pairs):
                if j == i:
                    continue
                other_ids = [t for t in other.get('noun_t5_ids', []) if t < Lk]
                if not other_ids:
                    continue
                other_map = a[:, other_ids].mean(dim=-1)
                excl = F.cosine_similarity(adj_map.unsqueeze(0), other_map.unsqueeze(0))
                total = total + lambda_c * excl

        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device)
    return total / n_valid
