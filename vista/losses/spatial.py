"""In-plane relational objectives (L_rel).

These act on spatial pairs (noun_a, relation, noun_b) that carry an explicit
image axis: axis='x' for left/right, axis='y' for top/bottom. Pairs with
axis=None are depth relations and are handled in `vista.losses.depth`.

`pos_spatial_loss` is the default relational term used in the paper; the rest
are ablation variants and the building blocks of `composite_spatial_loss`.

Reference
---------
PSG  Rezaei et al., "Precise Spatial Grounding" (2025) - the PoS(A, B)
     formulation used by `pos_spatial_loss`.
"""

import torch
import torch.nn.functional as F

from vista.losses.utils import build_grid, half_mask, soft_centroid_xy


def pos_spatial_loss(
    attn: torch.Tensor,   # [B, Lq, Lk]
    pairs: list,
    sharpen: float = 8.0,
) -> torch.Tensor:
    """
    PoS-based spatial loss (PSG, Rezaei et al. 2025).
    
    Loss = -PoS(A, B)^2  where PoS(A,B) = P(x_A > x_B) along the
    relation axis, computed on softmax-normalised attention projections.
    Minimising drives PoS -> 1, i.e. noun_a is consistently on the
    correct side of noun_b.
    
    Supports axis='x' (left/right) and axis='y' (top/bottom).
    Skips depth pairs (axis=None) — use depth_proxy_loss for those.
    """
    a    = attn[0]          # [Lq, Lk]
    Lq   = a.shape[0]
    Lk   = a.shape[1]
    side = int(Lq ** 0.5)
    if side * side != Lq:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)

    # normalised grid [0, side-1]  (integer coords, not [0,1])
    ys = torch.arange(side, device=attn.device, dtype=attn.dtype)
    xs = torch.arange(side, device=attn.device, dtype=attn.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    # [side, side] -> [Lq]
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)

    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0

    for pair in pairs:
        noun_a_ids = [t for t in pair.get('noun_a_t5_ids', []) if t < Lk]
        noun_b_ids = [t for t in pair.get('noun_b_t5_ids', []) if t < Lk]
        axis       = pair.get('axis')         # 'x', 'y', or None
        noun_a_side = pair.get('noun_a_side') # 'right','left','top','bottom'

        if not noun_a_ids or not noun_b_ids or axis is None:
            continue  # skip depth pairs

        map_a = a[:, noun_a_ids].mean(dim=-1)  # [Lq]
        map_b = a[:, noun_b_ids].mean(dim=-1)

        p_a = F.softmax(map_a * sharpen, dim=0)  # [Lq] — distribution over spatial tokens
        p_b = F.softmax(map_b * sharpen, dim=0)

        # Project onto the relevant axis
        # For 'right': want P(x_A > x_B) ~ 1 -> PoS(A,B) on x-axis
        # For 'left':  want P(x_B > x_A) ~ 1 -> PoS(B,A) on x-axis = PoS(A,B) ~ 0
        # Equivalently: always compute PoS(A,B) and flip the sign for 'left'/'top'
        coord = grid_x if axis == 'x' else grid_y

        # PoS(A, B) = sum_j p_B[j] * sum_{i >= j} p_A[i]
        # = sum_j p_B[j] * CDF_A_complement(j)
        # where CDF_A_complement(j) = P(X_A >= j) = 1 - CDF_A(j-1)
        # Efficient O(Lq log Lq) via sorting or O(Lq^2) direct
        #
        # Direct O(Lq^2) — fine for side <= 16:
        # pos = sum_{i,j: coord[i] >= coord[j]} p_a[i] * p_b[j]
        #
        # Vectorised:  pos = p_a @ (p_b @ (coord_a >= coord_b).float())
        # coord_a[i] >= coord_b[j]  ->  outer comparison matrix [Lq, Lq]
        mask = (coord.unsqueeze(1) >= coord.unsqueeze(0)).to(attn.dtype)  # [Lq, Lq]
        # pos_val = sum_i sum_j p_a[i] * p_b[j] * (coord_a[i] >= coord_b[j])
        pos_val = (p_a.unsqueeze(1) * p_b.unsqueeze(0) * mask).sum()  # scalar

        # For 'left' or 'top': noun_a should be on the smaller-coord side
        # -> want PoS(B, A) ~ 1, equivalently PoS(A, B) ~ 0
        # Loss = -(PoS)^2 drives PoS -> 1; for inverted relations flip
        if noun_a_side in ('left', 'top'):
            pos_val = 1.0 - pos_val   # now "PoS(B,A)" effectively

        loss_pair = -(pos_val ** 2)  # minimise -> PoS -> 1
        total    = total + loss_pair
        n_valid  += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
    return total / n_valid


def spatial_relative_centroid_loss(
    attn: torch.Tensor,     # [B, Lq, Lk]
    pairs: list,
    margin: float = 0.3,
    sharpen: float = 30.0,
) -> torch.Tensor:
    """
    Enforces centroid_axis(noun_a) > centroid_axis(noun_b) + margin.
 
    For "a to the right of b":  cx(a) > cx(b) + margin
    For "a above b":            cy(a) < cy(b) - margin  (y increases downward)
    """
    a    = attn[0]
    Lq   = a.shape[0]
    Lk   = a.shape[1]
    side = int(Lq ** 0.5)
    if side * side != Lq:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
 
    grid_x, grid_y = build_grid(side, attn.device, attn.dtype)
    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0
 
    for pair in pairs:
        noun_a_ids= [t for t in pair.get('noun_a_t5_ids', []) if t < Lk]
        noun_b_ids= [t for t in pair.get('noun_b_t5_ids', []) if t < Lk]
        axis= pair.get('axis')
        noun_a_side= pair.get('noun_a_side')
 
        if not noun_a_ids or not noun_b_ids or axis is None or noun_a_side is None:
            continue   # depth/proximity relations — no 2D ordering
 
        noun_a_map= a[:, noun_a_ids].mean(dim=-1)
        noun_b_map= a[:, noun_b_ids].mean(dim=-1)
 
        cx_a, cy_a= soft_centroid_xy(noun_a_map, grid_x, grid_y, sharpen)
        cx_b, cy_b= soft_centroid_xy(noun_b_map, grid_x, grid_y, sharpen)
 
        if axis == 'x':
            if noun_a_side == 'right':
                # want cx_a > cx_b + margin  →  hinge on (margin - (cx_a - cx_b))
                violation = margin - (cx_a - cx_b)
            else:   # noun_a_side == 'left'
                # want cx_a < cx_b - margin  →  hinge on (margin - (cx_b - cx_a))
                violation = margin - (cx_b - cx_a)
 
        elif axis == 'y':
            if noun_a_side == 'bottom':
                # y increases downward; "bottom" → cy_a > cy_b
                violation = margin - (cy_a - cy_b)
            else:   # noun_a_side == 'top'
                violation = margin - (cy_b - cy_a)
        else:
            continue
 
        # Hinge: only penalise violations, not satisfied constraints
        # loss_pair = F.relu(violation)
        loss_pair = F.gelu(violation)   # InfSplign's f_spatial
        total     = total + loss_pair
        n_valid  += 1
 
    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
    return total / n_valid


def spatial_half_field_loss(
    attn: torch.Tensor,     # [B, Lq, Lk]
    pairs: list,
    lambda_c: float = 0.3,
    sharpen: float = 10.0, #5
) -> torch.Tensor:
    """
    Penalises noun attention mass in the wrong image half.
 
    For "cat on the left":  minimise attention of 'cat' tokens in RIGHT half.
    Contrastive:            also push other nouns OUT of cat's LEFT half.
    """
    
    a= attn[0]           # [Lq, Lk]
    Lq= a.shape[0]
    Lk= a.shape[1]
    side= int(Lq**0.5)
    if side*side != Lq:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
 
    grid_x, grid_y = build_grid(side, attn.device, attn.dtype)
    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0
 
    for i, pair in enumerate(pairs):
        noun_a_ids= [t for t in pair.get('noun_a_t5_ids', []) if t < Lk]
        noun_a_side= pair.get('noun_a_side')
        if not noun_a_ids or noun_a_side is None:
            continue
 
        noun_a_map = a[:, noun_a_ids].mean(dim=-1)   # [Lq]
        probs= F.softmax(noun_a_map * sharpen, dim=0)
 
        # correct half mask for noun_a
        correct_mask= half_mask(side, noun_a_side, attn.device, attn.dtype)
        wrong_mask= 1.0 - correct_mask
 
        wrong_mass= (probs * wrong_mask).sum()
        total= total + wrong_mass
 
        # Contrastive: other nouns should NOT crowd noun_a's correct half
        if lambda_c > 0:
            for j, other in enumerate(pairs):
                if j==i:
                    continue
                other_ids = [t for t in other.get('noun_a_t5_ids', []) if t < Lk]
                if not other_ids:
                    continue
                other_map= a[:, other_ids].mean(dim=-1)
                other_probs= F.softmax(other_map * sharpen, dim=0)
                # penalise other noun for being in noun_a's correct half
                intrusion= (other_probs * correct_mask).sum()
                total= total + lambda_c*intrusion
        n_valid+= 1
 
    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
    return total / n_valid


def spatial_separation_loss(
    attn: torch.Tensor,     # [B, Lq, Lk]
    pairs: list,
    lambda_c: float = 0.3,
) -> torch.Tensor:
    """
    Maximises spatial separation between noun_a and noun_b maps.
    Applies to ALL spatial pairs including depth (axis=None).
    """
    a= attn[0]
    Lk= a.shape[1]
    total, n_valid= torch.zeros(1, device=attn.device, dtype=attn.dtype), 0
 
    for pair in pairs:
        noun_a_ids= [t for t in pair.get('noun_a_t5_ids', []) if t < Lk]
        noun_b_ids= [t for t in pair.get('noun_b_t5_ids', []) if t < Lk]
        if not noun_a_ids or not noun_b_ids:
            continue
 
        map_a= a[:, noun_a_ids].mean(dim=-1)   # [Lq]
        map_b= a[:, noun_b_ids].mean(dim=-1)
 
        # Cosine similarity ∈ [-1, 1]; maximise dissimilarity → minimise sim
        sim= F.cosine_similarity(map_a.unsqueeze(0), map_b.unsqueeze(0))
        total= total + sim   # minimise → maps become orthogonal
 
        n_valid += 1
 
    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
    return total / n_valid


def spatial_gaussian_target_loss(
    attn: torch.Tensor,     # [B, Lq, Lk]
    pairs: list,
    sigma: float = 0.25,
    sharpen: float = 10.0,
) -> torch.Tensor:
    """
    KL(noun_attention || gaussian_half_target).
    Smoother than hard half-field loss; better gradient near boundary.
    """
    a= attn[0]
    Lq= a.shape[0]
    Lk= a.shape[1]
    side= int(Lq ** 0.5)
    if side * side != Lq:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
 
    grid_x, grid_y = build_grid(side, attn.device, attn.dtype)
    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0
 
    def gaussian_target(noun_a_side: str) -> torch.Tensor:
        """1D Gaussian centred at the correct edge, broadcast over 2D grid."""
        if noun_a_side == "left":
            raw = torch.exp(-grid_x ** 2 / (2 * sigma ** 2))
        elif noun_a_side == "right":
            raw = torch.exp(-(1 - grid_x) ** 2 / (2 * sigma ** 2))
        elif noun_a_side == "top":
            raw = torch.exp(-grid_y ** 2 / (2 * sigma ** 2))
        elif noun_a_side == "bottom":
            raw = torch.exp(-(1 - grid_y) ** 2 / (2 * sigma ** 2))
        else:
            return None
        return raw / (raw.sum() + 1e-8)   # normalise to distribution
 
    for pair in pairs:
        noun_a_ids= [t for t in pair.get('noun_a_t5_ids', []) if t < Lk]
        noun_a_side= pair.get('noun_a_side')
        if not noun_a_ids or noun_a_side is None:
            continue
 
        target = gaussian_target(noun_a_side)
        if target is None:
            continue
 
        noun_a_map = a[:, noun_a_ids].mean(dim=-1)
        p = F.softmax(noun_a_map * sharpen, dim=0)   # [Lq], predicted distribution
 
        kl = (target * (target.log() - (p + 1e-8).log())).sum()
        total   = total + kl
        n_valid += 1
 
    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
    return total / n_valid


def spatial_iou_loss(
    attn: torch.Tensor,     # [B, Lq, Lk]
    pairs: list,
    sharpen: float = 10.0,
) -> torch.Tensor:
    """
    Penalises spatial overlap between noun_a and noun_b attention maps.
    Uses soft bounding boxes: mean ± std of attention-weighted coordinates.
    Works for ALL spatial pairs including depth.
    """
    a= attn[0]
    Lq= a.shape[0]
    Lk= a.shape[1]
    side = int(Lq ** 0.5)
    if side * side != Lq:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
 
    grid_x, grid_y = build_grid(side, attn.device, attn.dtype)
    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0
 
    def soft_bbox(map_vec: torch.Tensor):
        """Returns (cx, cy, sx, sy): mean and std of attention-weighted coords."""
        w  = F.softmax(map_vec * sharpen, dim=0)
        cx = (w * grid_x).sum()
        cy = (w * grid_y).sum()
        # Variance ≈ E[x²] - E[x]²
        sx = ((w * (grid_x - cx) ** 2).sum() + 1e-6).sqrt()
        sy = ((w * (grid_y - cy) ** 2).sum() + 1e-6).sqrt()
        return cx, cy, sx, sy
 
    def interval_overlap(c1, s1, c2, s2):
        """1D overlap between [c1-s1, c1+s1] and [c2-s2, c2+s2]. Soft, differentiable."""
        lo = torch.max(c1 - s1, c2 - s2)
        hi = torch.min(c1 + s1, c2 + s2)
        return F.relu(hi - lo)
 
    for pair in pairs:
        noun_a_ids = [t for t in pair.get('noun_a_t5_ids', []) if t < Lk]
        noun_b_ids = [t for t in pair.get('noun_b_t5_ids', []) if t < Lk]
        if not noun_a_ids or not noun_b_ids:
            continue
 
        map_a = a[:, noun_a_ids].mean(dim=-1)
        map_b = a[:, noun_b_ids].mean(dim=-1)
 
        cx_a, cy_a, sx_a, sy_a = soft_bbox(map_a)
        cx_b, cy_b, sx_b, sy_b = soft_bbox(map_b)
 
        overlap_x = interval_overlap(cx_a, sx_a, cx_b, sx_b)
        overlap_y = interval_overlap(cy_a, sy_a, cy_b, sy_b)
        overlap_area = overlap_x * overlap_y
 
        total   = total + overlap_area
        n_valid += 1
 
    if n_valid == 0:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
    return total / n_valid


def composite_spatial_loss(
    attn: torch.Tensor,
    pairs: list,
    w_sep:    float = 1.0,
    w_dir:    float = 0.5,
    lambda_c: float = 0.3,
    margin:   float = 0.1,
    sharpen:  float = 8.0,
    sigma:    float = 0.25,
) -> torch.Tensor:
    """
    Combined spatial loss:
      w_sep × (separation + iou_overlap)
    + w_dir × (half_field + relative_centroid + gaussian_target)
 
    Use in run_scale_optimization by replacing / alongside syngen_loss:
      loss = composite_spatial_loss(attn_avg, vista_cfg['spatial_pairs'], ...)
    """
    total = torch.zeros(1, device=attn.device, dtype=attn.dtype)
 
    if w_sep > 0:
        total = total + w_sep * spatial_separation_loss(attn, pairs, lambda_c=lambda_c)
        total = total + w_sep * spatial_iou_loss(attn, pairs, sharpen=sharpen)
 
    if w_dir > 0:
        total = total + w_dir * spatial_half_field_loss(attn, pairs, lambda_c=lambda_c, sharpen=sharpen)
        total = total + w_dir * spatial_relative_centroid_loss(attn, pairs, margin=margin, sharpen=sharpen)
        total = total + w_dir * spatial_gaussian_target_loss(attn, pairs, sigma=sigma, sharpen=sharpen)
 
    return total


def cross_scale_consistency_loss(attn, spatial_pairs, prev_centroids: dict, temp=10.0):
    """
    prev_centroids: dict mapping (noun_a_text) -> (cy, cx) from the
    previously optimized scale, normalized to [0,1]. Caller maintains
    this dict across calls to run_scale_optimization.
    Returns (loss, updated_centroids) — caller stores updated_centroids.
    """
    a = attn[0]
    Lq, Lk = a.shape
    side = int(Lq ** 0.5)
    if side * side != Lq:
        return torch.zeros(1, device=attn.device, dtype=attn.dtype), prev_centroids

    ys = torch.arange(side, device=attn.device, dtype=attn.dtype) / (side - 1 + 1e-6)
    xs = torch.arange(side, device=attn.device, dtype=attn.dtype) / (side - 1 + 1e-6)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gy, gx = gy.reshape(-1), gx.reshape(-1)

    total, n_valid = torch.zeros(1, device=attn.device, dtype=attn.dtype), 0
    new_centroids = {}

    for pair in spatial_pairs:
        ids_a = [t for t in pair.get("noun_a_t5_ids", []) if t < Lk]
        if not ids_a:
            continue
        key = pair["noun_a"]
        weights = F.softmax(a[:, ids_a].mean(dim=-1) * temp, dim=0)
        cy = (weights * gy).sum()
        cx = (weights * gx).sum()
        new_centroids[key] = (cy.detach(), cx.detach())

        if key in prev_centroids:
            py, px = prev_centroids[key]
            loss = (cy - py).pow(2) + (cx - px).pow(2)
            total = total + loss
            n_valid += 1

    if n_valid == 0:
        return torch.zeros(1, device=attn.device, dtype=attn.dtype), new_centroids
    return total / n_valid, new_centroids
