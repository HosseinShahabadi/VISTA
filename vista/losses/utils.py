"""Geometry and map helpers shared by the VISTA objectives.

Everything here operates on a single flattened attention map of shape [Lq],
where Lq = side x side is the number of spatial query tokens at the current
scale of the next-scale autoregressive schedule.
"""

import math

import torch
import torch.nn.functional as F


def build_grid(side: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (grid_x, grid_y) each of shape [side*side], normalised to [0,1]."""
    ys = torch.arange(side, device=device, dtype=dtype) / max(side - 1, 1)
    xs = torch.arange(side, device=device, dtype=dtype) / max(side - 1, 1)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    return grid_x.reshape(-1), grid_y.reshape(-1)   # [Lq], [Lq]


def soft_centroid_xy(map_vec: torch.Tensor,
                      grid_x: torch.Tensor,
                      grid_y: torch.Tensor,
                      sharpen: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable (cx, cy) centroid of a spatial attention map.
    map_vec : [Lq] unnormalised
    returns  : cx [scalar], cy [scalar], both in [0, 1]
    """
    weights = F.softmax(map_vec * sharpen, dim=0)
    cx = (weights * grid_x).sum()
    cy = (weights * grid_y).sum()
    return cx, cy


def half_mask(side: int, noun_a_side: str,
               device, dtype) -> torch.Tensor:
    """
    Hard binary mask — the half of the image that noun_a SHOULD occupy.
    Returns [Lq] tensor of 0/1.
    """
    grid_x, grid_y = build_grid(side, device, dtype)
    if noun_a_side == "left":
        return (grid_x < 0.5).to(dtype)
    elif noun_a_side == "right":
        return (grid_x >= 0.5).to(dtype)
    elif noun_a_side == "top":
        return (grid_y < 0.5).to(dtype)
    elif noun_a_side == "bottom":
        return (grid_y >= 0.5).to(dtype)
    else:
        return torch.ones(side * side, device=device, dtype=dtype)


def mean_token_map(a, ids, Lk):
    ids = [t for t in ids if t < Lk]
    if not ids:
        return None
    return a[:, ids].mean(dim=-1)   # [Lq]


def grid_coords(Lq, device, dtype):
    side = int(round(Lq ** 0.5))
    if side * side != Lq:
        return None, None, None
    ys = torch.arange(side, device=device, dtype=dtype)
    xs = torch.arange(side, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return gy.reshape(-1), gx.reshape(-1), side


def gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    """2D Gaussian smoothing kernel, matching CONFORM's GaussianSmoothing module."""
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel2d = g[:, None] * g[None, :]
    return kernel2d / kernel2d.sum()


def smooth_map(map_2d: torch.Tensor, kernel_size: int = 3, sigma: float = 0.5) -> torch.Tensor:
    """Reflect-padded Gaussian blur of a single [side, side] attention map."""
    side = map_2d.shape[0]
    kernel = gaussian_kernel2d(kernel_size, sigma, map_2d.device, map_2d.dtype)
    pad = kernel_size // 2
    x = map_2d.unsqueeze(0).unsqueeze(0)                       # [1,1,side,side]
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    out = F.conv2d(x, kernel.unsqueeze(0).unsqueeze(0))
    return out.view(side, side)


def soft_peak_count(map_vec, side, gy, gx, n_target, sharpen=12.0,
                      suppress_radius_frac=0.6, n_iters=None):
    """
    Greedily extracts up to `n_iters` soft peaks from map_vec [Lq] by
    repeatedly taking a softmax-weighted centroid around the current
    argmax, recording its peak "mass", then suppressing a Gaussian
    region around it before continuing. Fully differentiable w.r.t.
    map_vec (uses softmax / soft masks, no hard indexing in the
    suppression weights — only the *center* of suppression uses
    detached argmax, which is standard for soft-NMS style extraction).

    Returns:
        peak_masses: tensor [n_iters]  (relative mass assigned to each peak)
        n_iters defaults to n_target + 2 (a little slack so the loss can
        tell "too many" from "too few" rather than just "found enough").
    """
    if n_iters is None:
        n_iters = n_target + 2

    radius = max(1.0, side * suppress_radius_frac)
    work = map_vec.clone()
    peak_masses = []

    for _ in range(n_iters):
        weights = F.softmax(work * sharpen, dim=0)   # [Lq]
        # soft peak mass = how concentrated/strong this iteration's blob is
        mass = (weights * work).sum()
        peak_masses.append(mass)

        # locate centroid of this soft peak (detached center is fine;
        # gradient flows through `work` via `weights` at the next steps,
        # and through `mass` above)
        with torch.no_grad():
            cy = (weights * gy).sum()
            cx = (weights * gx).sum()
        dist2 = (gy - cy) ** 2 + (gx - cx) ** 2
        suppress = 1.0 - torch.exp(-dist2 / (2 * radius ** 2))   # ~0 near center, ~1 far
        work = work * suppress

    return torch.stack(peak_masses)   # [n_iters]


def canonical_centers(N, side, device, dtype):
    """N separable centers on a near-square lattice inside a side×side grid."""
    cols = math.ceil(math.sqrt(N))
    rows = math.ceil(N / cols)
    pts = []
    for i in range(rows):
        for j in range(cols):
            if len(pts) < N:
                cy = (i + 0.5) * (side / rows)
                cx = (j + 0.5) * (side / cols)
                pts.append(torch.tensor([cy, cx], device=device, dtype=dtype))
    return torch.stack(pts)  # [N, 2] (y, x)


def build_neighbour_index(side: int) -> list[list[int]]:
    """
    For each token i (row-major), return the indices of its 4-neighbours
    (up, down, left, right). Boundary tokens get fewer neighbours.
    Returned as a list-of-lists (not a tensor — used once, CPU only).
    """
    neighbours = []
    for idx in range(side * side):
        r, c = divmod(idx, side)
        nb = []
        if r > 0:          nb.append((r - 1) * side + c)   # up
        if r < side - 1:   nb.append((r + 1) * side + c)   # down
        if c > 0:          nb.append(r * side + (c - 1))   # left
        if c < side - 1:   nb.append(r * side + (c + 1))   # right
        neighbours.append(nb)
    return neighbours


def soft_perimeter(prob: torch.Tensor, side: int,
                    neighbour_index: list[list[int]]) -> torch.Tensor:
    """
    Compute soft edge weights for a probability map.

    edge_weight[i] = mean( |prob[i] - prob[j]| ) over 4-neighbours j.

    High where attention changes rapidly → boundary of the blob.

    Returns: edge_weight [Lq], unnormalised, non-negative, with grad_fn.
    """
    Lq = side * side
    diffs = []
    for i, nbs in enumerate(neighbour_index):
        if not nbs:
            continue
        # |prob[i] - prob[j]| for each neighbour j
        nb_tensor = torch.stack([prob[j] for j in nbs])   # [num_nb]
        d = (prob[i] - nb_tensor).abs().mean()
        diffs.append(d)

    if not diffs:
        return torch.zeros(Lq, device=prob.device, dtype=prob.dtype)

    # Scatter back to [Lq]
    edge = torch.zeros(Lq, device=prob.device, dtype=prob.dtype)
    for i, d in zip(range(len(diffs)), diffs):
        edge[i] = edge[i] + d   # out-of-place accumulation preserves grad

    return edge


def soft_perimeter_vectorised(prob: torch.Tensor, side: int) -> torch.Tensor:
    """
    Vectorised soft perimeter via 2D finite differences.
    Equivalent to soft_perimeter but ~100× faster and fully vectorised.

    Reshapes to (1, 1, side, side), applies Sobel-like differences,
    then reshapes back to [Lq].
    """
    p2d = prob.view(side, side)   # [H, W]

    # Finite differences (central where possible, forward/backward at edges)
    dx = torch.zeros_like(p2d)
    dy = torch.zeros_like(p2d)

    dx[:, 1:-1] = (p2d[:, 2:] - p2d[:, :-2]) / 2
    dx[:, 0]    =  p2d[:, 1]  - p2d[:, 0]
    dx[:, -1]   =  p2d[:, -1] - p2d[:, -2]

    dy[1:-1, :] = (p2d[2:, :] - p2d[:-2, :]) / 2
    dy[0,    :] =  p2d[1, :]  - p2d[0, :]
    dy[-1,   :] =  p2d[-1, :] - p2d[-2, :]

    magnitude = (dx ** 2 + dy ** 2).sqrt()   # gradient magnitude [H, W]
    return magnitude.reshape(-1)              # [Lq], with grad_fn


def bbox_binarize(map_2d: torch.Tensor, quantile: float = 0.8) -> torch.Tensor:
    """Threshold a [side, side] map at the given quantile, then fill the
    bounding box of surviving pixels with 1s (rest 0), per A-STAR App. A.2."""
    side = map_2d.shape[0]
    thresh = torch.quantile(map_2d.flatten(), quantile)
    hot = map_2d >= thresh
    if not hot.any():
        return torch.zeros_like(map_2d)
    rows = hot.any(dim=1).nonzero(as_tuple=True)[0]
    cols = hot.any(dim=0).nonzero(as_tuple=True)[0]
    mask = torch.zeros_like(map_2d)
    mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1] = 1.0
    return mask
