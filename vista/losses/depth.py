"""Depth-ordering objective: Boundary Containment (BC).

Applied to spatial pairs with axis=None, i.e. relations such as "behind",
"in front of" and "hidden by", where the constraint is an occlusion ordering
rather than an in-plane direction.

`depth_occlusion_loss` is the BC objective reported in the paper. It combines
two terms:

  separation  A and B must not occupy the same region. This builds on the
              segregation principle of A-STAR (see `vista.losses.entity`);
              here it is expressed as an overlap between soft bounding boxes
              rather than a map-level IoU.

  adjacency   the part of B's perimeter that faces A must be *touching* A:
              random convex combinations of the facing perimeter tokens have to
              land inside A's high-attention region. This encodes "B peeks out
              from behind A" rather than merely "B is somewhere else", which
              plain separation cannot distinguish.

`depth_proxy_loss` is the cheaper entropy-based alternative used in ablations.
"""

import torch
import torch.nn.functional as F

from vista.losses.utils import build_grid, soft_perimeter_vectorised


def depth_occlusion_loss(
    attn: torch.Tensor,         # [B, Lq, Lk]  with grad_fn
    pairs: list,                # spatial pairs with axis=None (depth)
    sharpen:        float = 8.0,
    n_perimeter:    int   = 64,     # K: how many facing-perimeter tokens to use
    n_samples:      int   = 256,    # N: convex combination samples
    tau:            float = 0.3,    # temperature for "facing" selection
    kernel_bw:      float = 0.15,   # Gaussian kernel bandwidth for a_A(p)
    inside_thresh:  float = 0.3,    # fraction of A's peak to count as "inside"
    w_sep:          float = 1.0,    # weight: separation sub-loss
    w_adj:          float = 1.0,    # weight: adjacency sub-loss
) -> torch.Tensor:
    """
    Depth occlusion loss for "B behind A" spatial pairs.

    Enforces two properties in attention space:
      1. A and B attention maps do not overlap (separation).
      2. The perimeter of B, on the side facing A, is adjacent to A —
         i.e. random convex combinations of those perimeter tokens
         land inside A's high-attention region.

    Args
    ----
    attn          [B, Lq, Lk] cross-attention map with grad_fn.
    pairs         List of spatial pair dicts. Only processes pairs
                  where axis is None (depth relations). For axis != None
                  pairs, contributes zero loss.
    sharpen       Softmax temperature: higher = more peaked distributions.
    n_perimeter   Number of top facing-perimeter tokens to sample from (K).
    n_samples     Number of convex combination virtual points (N).
    tau           Facing-selection temperature. Lower = only tokens very
                  close to A's centroid are selected as "facing".
    kernel_bw     Gaussian kernel bandwidth for evaluating whether a
                  virtual point is inside A's attention. Units: normalised
                  [0,1] image coords.
    inside_thresh Fraction of A's peak attention that counts as "inside A".
                  0.3 means: a point must receive ≥ 30% of A's max attention.
    w_sep         Weight of the separation sub-loss.
    w_adj         Weight of the adjacency sub-loss.

    Returns
    -------
    Scalar tensor with grad_fn.
    """
    
    a    = attn[0]           # [Lq, Lk]
    Lq   = a.shape[0]
    Lk   = a.shape[1]
    side = int(Lq ** 0.5)
    if side * side != Lq:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)

    grid_x, grid_y = _build_grid(side, attn.device, attn.dtype)
    coords = torch.stack([grid_x, grid_y], dim=-1)   # [Lq, 2]

    total_sep = torch.zeros(1, device=attn.device, dtype=attn.dtype)
    total_adj = torch.zeros(1, device=attn.device, dtype=attn.dtype)
    n_valid   = 0

    for pair in pairs:
        # Only process depth relations (axis=None)
        if pair.get('axis') is not None:
            continue

        # In "X behind Y": noun_a=X (occluded/behind), noun_b=Y (occluder/front)
        # We call them B (behind=noun_a) and A (front=noun_b)
        _FRONT_RELATIONS = {"in front of", "front"}
        _BEHIND_RELATIONS = {"behind", "hidden by", "hidden"}

        relation = pair.get('relation', '')
        if relation in _BEHIND_RELATIONS:
            print("########## it is behind ############3")
            b_ids = [t for t in pair.get('noun_a_t5_ids', []) if t < Lk]   # behind
            a_ids = [t for t in pair.get('noun_b_t5_ids', []) if t < Lk]   # in front
        elif relation in _FRONT_RELATIONS:
            print("########## it is front ############3")
            a_ids = [t for t in pair.get('noun_a_t5_ids', []) if t < Lk]   # behind
            b_ids = [t for t in pair.get('noun_b_t5_ids', []) if t < Lk]   # in front
        else:
            print("########## it is NONE ############3")
            continue

        # b_ids = [t for t in pair.get('noun_a_t5_ids', []) if t < Lk]   # behind
        # a_ids = [t for t in pair.get('noun_b_t5_ids', []) if t < Lk]   # in front

        if not a_ids or not b_ids:
            continue

        # ── Step 1: Soft attention distributions ──────────────────
        map_a = a[:, a_ids].mean(dim=-1)   # [Lq]  "in front" object
        map_b = a[:, b_ids].mean(dim=-1)   # [Lq]  "behind"   object

        prob_a = F.softmax(map_a * sharpen, dim=0)   # [Lq]
        prob_b = F.softmax(map_b * sharpen, dim=0)   # [Lq]

        # ── Step 2: Separation loss ────────────────────────────────
        # Penalise spatial overlap of A and B attention bounding boxes.
        cx_a = (prob_a * grid_x).sum()
        cy_a = (prob_a * grid_y).sum()
        sx_a = ((prob_a * (grid_x - cx_a) ** 2).sum() + 1e-6).sqrt()
        sy_a = ((prob_a * (grid_y - cy_a) ** 2).sum() + 1e-6).sqrt()

        cx_b = (prob_b * grid_x).sum()
        cy_b = (prob_b * grid_y).sum()
        sx_b = ((prob_b * (grid_x - cx_b) ** 2).sum() + 1e-6).sqrt()
        sy_b = ((prob_b * (grid_y - cy_b) ** 2).sum() + 1e-6).sqrt()

        def _overlap_1d(c1, s1, c2, s2):
            return F.relu(torch.min(c1 + s1, c2 + s2) - torch.max(c1 - s1, c2 - s2))

        overlap_x = _overlap_1d(cx_a, sx_a, cx_b, sx_b)
        overlap_y = _overlap_1d(cy_a, sy_a, cy_b, sy_b)
        sep_loss  = overlap_x * overlap_y
        total_sep = total_sep + sep_loss

        # ── Step 3: Soft perimeter of B ────────────────────────────
        # edge_b[i] is large where B's attention map changes sharply.
        edge_b = _soft_perimeter_vectorised(prob_b, side)   # [Lq], grad_fn

        # ── Step 4: Facing-perimeter selection ────────────────────
        # Score each token by: how much it is on B's edge AND
        # how close it is to A's centroid (= the side facing A).
        #
        # dist_to_A_centroid[i] = || coord[i] - (cx_a, cy_a) ||
        ca = torch.stack([cx_a, cy_a])                       # [2]
        dist_to_a = ((coords - ca.detach()) ** 2).sum(-1).sqrt()   # [Lq]
        # Detach A's centroid for this distance: we want perimeter selection
        # to be driven by geometry, not to pull A's centroid toward B.

        facing_score = edge_b * torch.exp(-dist_to_a / (tau + 1e-8))   # [Lq]

        # Soft-select top-K facing tokens via gumbel-style or just top-k weights
        # We use a differentiable soft-topk via normalised facing_score:
        #   treat facing_score as unnormalised weights and sample K tokens
        #   proportionally. In the loss, we weight each perimeter token by
        #   its normalised facing weight — no hard selection needed.
        facing_w = facing_score / (facing_score.sum() + 1e-8)   # [Lq], sums to 1

        # For coordinate computation, use the top-n_perimeter tokens
        # (by detached score) — this defines WHICH coords we sample from,
        # while facing_w (with grad) weights their contribution to the loss.
        with torch.no_grad():
            topk_vals, topk_idx = facing_score.topk(
                min(n_perimeter, Lq), sorted=False
            )
        # topk_idx: [K]  — indices of the K most "facing" perimeter tokens
        K = topk_idx.shape[0]

        # Weights for those K tokens (with grad)
        topk_weights = facing_w[topk_idx]                    # [K]
        topk_weights = topk_weights / (topk_weights.sum() + 1e-8)   # normalise

        # Coordinates of those K tokens
        topk_coords = coords[topk_idx]                       # [K, 2]

        # ── Step 5: Convex combination sampling ───────────────────
        # Draw N weight vectors w_n ~ Dirichlet(1,...,1), shape [N, K].
        # Each virtual point: p_n = Σ_k w_n[k] × topk_coords[k]
        # We use topk_weights as a BIAS in the convex combination:
        #   effective weight = w_n × topk_weights (then renormalise)
        # This makes samples concentrate near the most "facing" tokens
        # while still exploring the hull.
        #
        # Dirichlet(1,...,1) = uniform over simplex.
        # Using Dirichlet via Gamma: w ~ Gamma(1,1), then normalise.
        # torch.distributions.Dirichlet requires float32; cast if needed.

        raw_alpha = torch.ones(K, device=attn.device, dtype=torch.float32)
        dirichlet = torch.distributions.Dirichlet(raw_alpha)
        raw_samples = dirichlet.sample((n_samples,))          # [N, K], float32
        raw_samples = raw_samples.to(attn.dtype)              # match model dtype

        # Bias by topk_weights (with grad) and renormalise
        biased = raw_samples * topk_weights.unsqueeze(0)      # [N, K]
        biased = biased / (biased.sum(dim=-1, keepdim=True) + 1e-8)  # [N, K]

        # Virtual points in [0,1]² : [N, 2]
        virtual_pts = biased @ topk_coords                    # [N, 2]

        # ── Step 6: "Inside A" check via Gaussian kernel ──────────
        # For each virtual point p_n, compute how much A attends to it:
        #   a_A(p_n) = Σ_i prob_a[i] × exp(-||coord[i] - p_n||² / (2h²))
        #
        # Shape: [N, Lq] distances, then kernel-weighted sum → [N]

        # coords: [Lq, 2], virtual_pts: [N, 2]
        diff = coords.unsqueeze(0) - virtual_pts.unsqueeze(1)   # [N, Lq, 2]
        sq_dist = (diff ** 2).sum(-1)                            # [N, Lq]
        kernel  = torch.exp(-sq_dist / (2 * kernel_bw ** 2))    # [N, Lq]

        # a_A at each virtual point
        a_A_at_pts = (kernel * prob_a.unsqueeze(0)).sum(-1)     # [N]

        # ── Step 7: Adjacency loss ────────────────────────────────
        # We want ≥ target_frac fraction of virtual points to be "inside A".
        # "Inside A" = a_A(p) ≥ inside_thresh × max(prob_a).
        #
        # Loss: mean( relu(threshold - a_A(p_n)) )
        # This is zero for points already inside, positive for outside.
        # Minimising this pulls B's perimeter toward A's interior.

        threshold = inside_thresh * prob_a.detach().max()
        per_point_loss = F.relu(threshold - a_A_at_pts)        # [N]
        adj_loss  = per_point_loss.mean()
        total_adj = total_adj + adj_loss

        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)

    loss = (w_sep * total_sep + w_adj * total_adj) / n_valid
    return loss


def depth_proxy_loss(attn, spatial_pairs, temp=10.0):
    """
    For 'behind'/'in front of'/'hidden by' pairs, the foreground
    object's attention mass should be more concentrated and over
    a LARGER footprint than the background object's.
    Uses normalized attention entropy as inverse depth proxy.
    """
    a  = attn[0]
    Lk = a.shape[-1]
    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0

    DEPTH_REL = {"behind": -1, "in front of": +1, "hidden by": -1,
                  "hidden": -1, "front": +1, "atop": +1, "over": +1, "under": -1}

    for pair in spatial_pairs:
        rel = pair.get("relation")
        sign = DEPTH_REL.get(rel)
        if sign is None:
            continue
        ids_a = [t for t in pair.get("noun_a_t5_ids", []) if t < Lk]
        ids_b = [t for t in pair.get("noun_b_t5_ids", []) if t < Lk]
        if not ids_a or not ids_b:
            continue

        map_a = F.softmax(a[:, ids_a].mean(dim=-1) * temp, dim=0)
        map_b = F.softmax(a[:, ids_b].mean(dim=-1) * temp, dim=0)

        # support size proxy = exp(entropy); larger = bigger footprint = foreground
        size_a = torch.exp(-(map_a * (map_a + 1e-8).log()).sum())
        size_b = torch.exp(-(map_b * (map_b + 1e-8).log()).sum())

        # sign=+1 means noun_a should be foreground (larger) → want size_a > size_b
        diff = sign * (size_b - size_a)   # minimize: want this negative
        total = total + F.relu(diff + 0.0)  # hinge-style
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
    return total / n_valid


def combined_depth_loss(
    attn: torch.Tensor,
    spatial_pairs: list,
    depth_cfg: dict = None,
) -> torch.Tensor:
    """
    Convenience wrapper. Splits spatial_pairs into depth (axis=None)
    and directional (axis != None), applies depth_occlusion_loss to
    depth pairs only.

    Usage inside run_scale_optimization:
        from depth_occlusion_loss import combined_depth_loss
        loss = combined_depth_loss(attn_avg, vista_cfg['spatial_pairs'])

    depth_cfg keys (all optional, fall back to depth_occlusion_loss defaults):
        sharpen, n_perimeter, n_samples, tau, kernel_bw,
        inside_thresh, w_sep, w_adj
    """
    depth_pairs = [p for p in spatial_pairs if p.get('axis') is None]
    if not depth_pairs:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)

    cfg = depth_cfg or {}
    return depth_occlusion_loss(
        attn,
        depth_pairs,
        sharpen       = cfg.get('sharpen',       8.0),
        n_perimeter   = cfg.get('n_perimeter',   64),
        n_samples     = cfg.get('n_samples',     256),
        tau           = cfg.get('tau',           0.3),
        kernel_bw     = cfg.get('kernel_bw',     0.15),
        inside_thresh = cfg.get('inside_thresh', 0.3),
        w_sep         = cfg.get('w_sep',         1.0),
        w_adj         = cfg.get('w_adj',         1.0),
    )
