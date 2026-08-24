"""Counting objectives (L_num).

NOTE: the numeracy term is implemented but DISABLED in every experiment reported
in the paper - it is not part of the default active loss set in
`vista.registry`. It is released for completeness and for follow-up work.

Four formulations are provided:
  cluster    soft non-maximum-suppression peak extraction, scored on peak
             strength, uniformity and overflow past the target count;
  entropy    normalised attention entropy as a monotonic proxy for count;
  softmask   quantile thresholding plus soft assignment to greedily seeded
             anchors, a differentiable stand-in for connected components;
  template   KL against a fixed mixture-of-N-Gaussians target on a separable
             lattice, phase-aligned to the map's current centre of mass.
"""

import math

import torch
import torch.nn.functional as F

from vista.losses.utils import canonical_centers, grid_coords, mean_token_map, soft_peak_count


def numerical_cluster_count_loss(
    attn: torch.Tensor,      # [B, Lq, Lk] with grad_fn
    numerical_pairs: list,
    lambda_uniform: float = 0.3,
    sharpen: float = 12.0,
    suppress_radius_frac: float = 0.6,
    max_count: int = 12,
) -> torch.Tensor:
    """
    Encourages the noun's attention map to decompose into exactly
    `number` separated peaks of comparable mass (e.g. "7 cats" -> 7
    roughly-equal blobs), instead of one blob or a smeared-out map.

    Loss = -mean(top-N peak masses) + lambda_uniform * var(top-N peak masses)
           + overflow_penalty(mass found in extra slots beyond N)

    - The first term rewards strong, well-separated peaks for the first
      N extractions (negative because we minimise).
    - The variance term discourages "1 huge blob + N-1 tiny crumbs",
      pushing toward N *comparably sized* clusters.
    - The overflow term penalises leftover mass still concentrated in
      slot N+1, N+2 (i.e. evidence of a stray Nth+1 cluster, meaning
      the count is likely too high / map is overcrowded).
    """
    a  = attn[0]
    Lq, Lk = a.shape
    gy, gx, side = grid_coords(Lq, a.device, a.dtype)
    if side is None:
        return torch.tensor(0.0, device=attn.device)

    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0

    for pair in numerical_pairs:
        if pair.get('relation') != 'count':
            continue
        try:
            n_target = int(pair['number'])
        except (KeyError, ValueError, TypeError):
            continue
        if n_target < 2 or n_target > max_count:
            # 1 cat -> nothing to separate; absurdly large counts -> skip,
            # attention map can't meaningfully resolve that many blobs
            continue

        noun_map = mean_token_map(a, pair.get('noun_t5_ids', []), Lk)
        if noun_map is None:
            continue

        # normalise to a comparable scale across noun maps
        noun_map = (noun_map - noun_map.mean()) / (noun_map.std() + 1e-6)

        peak_masses = soft_peak_count(
            noun_map, side, gy, gx, n_target,
            sharpen=sharpen, suppress_radius_frac=suppress_radius_frac,
        )   # [n_target + 2]

        target_masses = peak_masses[:n_target]
        overflow_masses = peak_masses[n_target:]

        strength_term = -target_masses.mean()
        uniformity_term = target_masses.var(unbiased=False) if n_target > 1 else torch.zeros_like(strength_term)
        overflow_term = F.relu(overflow_masses).mean() if overflow_masses.numel() > 0 else torch.zeros_like(strength_term)

        total = total + strength_term + lambda_uniform * uniformity_term + lambda_uniform * overflow_term
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device)
    return total / n_valid


def numerical_entropy_count_loss(
    attn: torch.Tensor,
    numerical_pairs: list,
    temp: float = 0.5, #1.0,
) -> torch.Tensor:
    """
    Cheaper alternative to cluster counting: uses the entropy of the
    noun's spatial attention distribution as a monotonic proxy for
    "how many distinct regions are lit up". A single cat -> low-entropy,
    concentrated map. Many cats -> higher-entropy, more spread-out map.

    We map n_target to a *target entropy* via log(n_target) (entropy of
    n_target equally-likely uniform clusters covering the same total
    area), and minimise squared distance between actual and target
    entropy (normalised by the max possible entropy log(Lq) so it's
    scale-free across resolutions).
    """
    a  = attn[0]
    Lq, Lk = a.shape
    max_entropy = torch.log(torch.tensor(float(Lq), device=attn.device))

    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0

    for pair in numerical_pairs:
        if pair.get('relation') != 'count':
            continue
        try:
            n_target = int(pair['number'])
        except (KeyError, ValueError, TypeError):
            continue
        if n_target < 1:
            continue

        noun_map = mean_token_map(a, pair.get('noun_t5_ids', []), Lk)
        if noun_map is None:
            continue

        p = F.softmax(noun_map / temp, dim=0)
        entropy = -(p * (p + 1e-8).log()).sum()
        norm_entropy = entropy / (max_entropy + 1e-8)

        # target: entropy of n_target uniform clusters relative to whole grid
        target_entropy = torch.log(torch.clamp(torch.tensor(float(n_target), device=attn.device), min=1.0))
        norm_target = torch.clamp(target_entropy / (max_entropy + 1e-8), max=1.0)

        total = total + (norm_entropy - norm_target) ** 2
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device)
    return total / n_valid


def numerical_softmask_component_loss(
    attn: torch.Tensor,
    numerical_pairs: list,
    threshold_q: float = 0.7,
    lambda_separation: float = 0.5,
) -> torch.Tensor:
    """
    Thresholds the noun map at its q-th quantile to get a soft binary
    "active region" mask, then measures pairwise spatial compactness
    among active cells to discourage one giant connected blob when
    n_target > 1 (a single large blob has many close-together active
    cells; n_target separate small blobs has active cells clustered
    into far-apart groups). This is a soft, differentiable proxy for
    "number of connected components" without needing actual labeling
    (which isn't differentiable).

    We approximate "spread into n_target groups" by checking that the
    active mass, when split into n_target nearest-neighbour groups via
    k-means-like soft assignment to n_target learnable-free anchor
    points (initialized at top-n_target local maxima), has low
    within-group variance and high between-group separation.
    """
    a  = attn[0]
    Lq, Lk = a.shape
    gy, gx, side = grid_coords(Lq, a.device, a.dtype)
    if side is None:
        return torch.tensor(0.0, device=attn.device)

    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0

    for pair in numerical_pairs:
        if pair.get('relation') != 'count':
            continue
        try:
            n_target = int(pair['number'])
        except (KeyError, ValueError, TypeError):
            continue
        if n_target < 2:
            continue

        noun_map = mean_token_map(a, pair.get('noun_t5_ids', []), Lk)
        if noun_map is None:
            continue

        probs = F.softmax(noun_map, dim=0)   # [Lq], soft "activation mass"
        q = torch.quantile(probs.detach(), threshold_q)
        active_weight = torch.sigmoid((probs - q) * 50.0)   # soft mask, differentiable
        active_weight = active_weight * probs               # weight by actual mass too

        # seed n_target anchors at greedily-suppressed peaks (detached, no grad needed
        # for anchor *positions* — only the soft assignment weights carry gradient)
        with torch.no_grad():
            work = active_weight.clone()
            anchors_y, anchors_x = [], []
            radius = max(1.0, side * 0.5)
            for _ in range(n_target):
                idx = torch.argmax(work)
                anchors_y.append(gy[idx])
                anchors_x.append(gx[idx])
                dist2 = (gy - gy[idx]) ** 2 + (gx - gx[idx]) ** 2
                work = work * (1.0 - torch.exp(-dist2 / (2 * radius ** 2)))
            anchors_y = torch.stack(anchors_y)
            anchors_x = torch.stack(anchors_x)

        # soft assignment of every cell to nearest anchor
        d2 = (gy.unsqueeze(1) - anchors_y.unsqueeze(0)) ** 2 + \
             (gx.unsqueeze(1) - anchors_x.unsqueeze(0)) ** 2   # [Lq, n_target]
        assign = F.softmax(-d2 / (2 * (side * 0.25) ** 2), dim=1)   # [Lq, n_target]

        group_mass = (active_weight.unsqueeze(1) * assign).sum(dim=0)   # [n_target]
        group_mass_norm = group_mass / (group_mass.sum() + 1e-8)

        # encourage mass spread roughly evenly across the n_target groups
        uniform_target = torch.full_like(group_mass_norm, 1.0 / n_target)
        balance_term = ((group_mass_norm - uniform_target) ** 2).mean()

        # encourage groups to actually be spatially separated (anchors not collapsed)
        with torch.no_grad():
            pairwise = torch.cdist(
                torch.stack([anchors_y, anchors_x], dim=1),
                torch.stack([anchors_y, anchors_x], dim=1),
            )
            min_sep = pairwise[~torch.eye(n_target, dtype=torch.bool, device=a.device)].min() if n_target > 1 else torch.tensor(float(side), device=a.device)
        separation_term = F.relu(1.0 - min_sep / (side * 0.15))   # penalise collapsed anchors

        total = total + balance_term + lambda_separation * separation_term
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device)
    return total / n_valid


def numerical_template_loss(
    attn: torch.Tensor,          # [B, Lq, Lk], grad_fn
    numerical_pairs: list,
    alpha_res: float = 2.0,      # scale gate: require side >= alpha_res * sqrt(N)
    tau: float = 0.3,            # blob floor at half-spacing (CountCluster eq. 2)
    max_count: int = 12,
) -> torch.Tensor:
    """
    N-modal template matching for count. Instead of reading the count out of
    the map (entropy / soft-NMS peaks), we build a fixed mixture-of-N-Gaussians
    target on an always-separable lattice, phase-align it to the map's current
    center of mass (translation only), and minimize KL(target || q). KL in this
    direction is mode-covering => strongly penalizes dropping any instance.
    Skips scales too coarse to host N separated blobs (returns 0 -> composes
    cleanly with spatial/binding losses on the same scale).
    """
    a = attn[0]
    Lq, Lk = a.shape
    gy, gx, side = grid_coords(Lq, a.device, a.dtype)
    if side is None:
        return torch.tensor(0.0, device=attn.device)

    total = torch.zeros(1, device=attn.device, dtype=attn.dtype)
    n_valid = 0

    for pair in numerical_pairs:
        if pair.get('relation') != 'count':
            continue
        try:
            N = int(pair['number'])
        except (KeyError, ValueError, TypeError):
            continue
        if N < 2 or N > max_count:
            continue
        # ── resolution gate ──────────────────────────────────────────
        if side < math.ceil(alpha_res * math.sqrt(N)):
            continue

        noun_map = mean_token_map(a, pair.get('noun_t5_ids', []), Lk)
        if noun_map is None:
            continue
        q = F.softmax(noun_map / 0.5, dim=0)                       # [Lq], concentrated

        centers = canonical_centers(N, side, a.device, a.dtype)  # [N,2]
        # phase-align lattice to current mass (respect where objects want to be)
        with torch.no_grad():
            cy = (q * gy).sum(); cx = (q * gx).sum()
            shift = torch.stack([cy - centers[:, 0].mean(),
                                 cx - centers[:, 1].mean()])
            centers = centers + shift
            centers[:, 0].clamp_(0, side - 1); centers[:, 1].clamp_(0, side - 1)

        spacing = side / math.sqrt(N)
        sigma2 = (0.5 * spacing) ** 2 / (-2.0 * math.log(tau))
        d2 = (gy.unsqueeze(1) - centers[:, 0]) ** 2 + \
             (gx.unsqueeze(1) - centers[:, 1]) ** 2          # [Lq, N]
        target = torch.exp(-d2 / (2 * sigma2)).max(dim=1).values  # [Lq], N-modal
        target = (target / (target.sum() + 1e-8)).detach()

        # KL(target || q): mode-covering, concentrates onto blobs, empties valleys
        kl = (target * ((target + 1e-8).log() - (q + 1e-8).log())).sum()
        total = total + kl / math.sqrt(N)                   # √N scaling
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device)
    return (total / n_valid) * Lq


def composite_numerical_loss(
    attn: torch.Tensor,
    numerical_pairs: list,
    w_cluster: float = 0.5,
    w_entropy: float = 0.3,
    w_component: float = 0.2,
    **kwargs,
) -> torch.Tensor:
    l1 = numerical_cluster_count_loss(attn, numerical_pairs)
    l2 = numerical_entropy_count_loss(attn, numerical_pairs)
    l3 = numerical_softmask_component_loss(attn, numerical_pairs)
    return w_cluster * l1 + w_entropy * l2 + w_component * l3
