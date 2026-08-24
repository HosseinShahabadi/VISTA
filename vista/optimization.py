"""Per-scale test-time optimisation.

This is the core of VISTA. For a selected scale `si`, the token features that
the backbone is about to turn into logits are treated as a free variable and
perturbed by a few steps of *normalised* gradient descent on the compositional
objective, using cross-attention maps captured differentiably from the same
forward pass.

Three design points matter, and are the reason a direct port of diffusion-side
attention steering does not work here:

Per-token gradient normalisation
    A single global L2 norm over a [bs, L, C] tensor dilutes the per-element
    perturbation by roughly 1/sqrt(N). Each objective's gradient is instead
    normalised per token (`g.norm(dim=-1, keepdim=True)`), so every token gets a
    unit-length direction and the step size means the same thing at every scale.

Multi-objective combination
    All active objectives share one forward pass; their gradients are taken
    w.r.t. the same leaf and combined as a sum of unit per-token directions, so
    no term dominates by raw scale alone. An objective whose raw gradient norm
    has fallen below its own tolerance simply contributes nothing this step —
    recomputed fresh each step, so it can come back later and there is no
    retirement state to leak across scales.

Adaptive step budget
    Step 0 is needed anyway, so its gradient norms are free information. The
    ratio ||g_0|| / grad_tol, in log units, sets N_eff <= N_max: badly aligned
    scales get the full budget, nearly aligned scales exit after one step.

Plain normalised gradient descent is used rather than Adam: with 32 layers of
gradient amplification, Adam's per-coordinate rescaling made steps wildly
inconsistent across scales, whereas a fixed per-step displacement budget is
stable.
"""

import logging
import math

import torch

from vista.attention import VISTAAttentionCapture, register_attention_hooks
from vista.forward import forward_scale_for_grad
from vista.kv_cache import caching_is_active, restore_kv_cache, save_kv_cache
from vista.registry import DEFAULT_ACTIVE_LOSSES, LOSS_GRAD_TOLS, LOSS_REGISTRY

logger = logging.getLogger('vista')


def combine_loss_gradients(loss_terms, param, grad_tols, eps=1e-8):
    """One unit per-token direction per objective, gated on its own tolerance.

    Returns (directions, raw_norms):
      directions  {loss_name: g / ||g||_token}, only for objectives whose raw
                  per-token gradient norm exceeds their own `grad_tols` entry.
      raw_norms   {loss_name: mean raw per-token gradient norm}, logged for
                  every objective including the ones gated out this step.

    Gradient-norm balancing rather than loss-value balancing: loss VALUES do not
    predict gradient MAGNITUDE (a bounded cosine loss can have a huge or tiny
    gradient; an unbounded KL can be nearly flat), so only normalising the
    gradients actually controls which term drives the update.
    """
    valid = [(name, l) for name, l in loss_terms.items() if l is not None and l.requires_grad]
    if not valid:
        return {}, {}

    dirs, grad_norms = {}, {}
    for i, (name, loss) in enumerate(valid):
        retain = i < len(valid) - 1     # keep the graph alive until the last term
        g = torch.autograd.grad(loss, param, retain_graph=retain, allow_unused=True)[0]
        if g is None:
            continue
        per_token_norm = g.norm(dim=-1, keepdim=True)
        raw_norm = per_token_norm.mean().item()
        grad_norms[name] = raw_norm
        if raw_norm < grad_tols.get(name, 1e-3):
            continue
        dirs[name] = g / (per_token_norm + eps)     # unit per-token direction, per objective

    return dirs, grad_norms


def adaptive_lr(last_stage: torch.Tensor, base_lr: float) -> float:
    """Turn the dimensionless `base_lr` into an absolute step size.

    Normalised GD moves `param` by exactly `lr` along the gradient direction,
    but what counts as a meaningful displacement depends on the feature scale,
    which varies both by scale index (coarse scales carry lower-energy features
    than fine ones) and by prompt (confident outputs have larger magnitudes).

    Setting lr = base_lr x RMS(last_stage) x sqrt(C) makes each TOKEN's update
    have L2 norm ~= base_lr x that token's own L2 norm, i.e. a fixed fraction of
    the local feature magnitude — prompt-invariant and scale-invariant. So
    base_lr = 0.005 means "perturb each token by 0.5% of its own magnitude per
    step", and lr stops being a per-prompt hyperparameter.

    The sqrt(C) factor is what keeps the budget per TOKEN; without it the same
    displacement is divided across every (token x channel) coordinate.
    """
    C = last_stage.shape[-1]
    rms = last_stage.float().pow(2).mean().sqrt().item()
    return base_lr * rms * math.sqrt(C)


def run_scale_optimization(
    infinity,
    last_stage: torch.Tensor,
    si: int,
    pn: tuple,
    scale_schedule: list,
    cond_BD_or_gss: torch.Tensor,
    ca_kv: tuple,
    attn_fn,
    rope2d_freqs_grid: torch.Tensor,
    vista_cfg: dict,
) -> torch.Tensor:
    """Steer the features of one scale in place of the plain forward pass.

    Returns the (possibly perturbed) `last_stage`, cast back to the model dtype.
    Any early return hands back the input tensor untouched.
    """

    pt, ph, pw = pn
    if ph != pw:
        return last_stage
    if ph * pw < 4:
        # Degenerate grids carry no usable spatial signal.
        # NOTE (Infinity-8B): with apply_spatial_patchify=1 the token grid is
        # halved, so a given scale index maps to a much smaller (ph, pw) than on
        # 2B and this guard can silently no-op scales that do steer on 2B.
        return last_stage

    pairs           = vista_cfg.get('pairs', [])
    spatial_pairs   = vista_cfg.get('spatial_pairs', [])
    numerical_pairs = vista_cfg.get('numerical_pairs', [])
    n_steps         = vista_cfg.get('n_steps', 5)
    base_lr         = vista_cfg.get('base_lr', 0.005)
    grad_tol        = vista_cfg.get('grad_tol', 1e-3)   # global fallback

    if (not pairs) and (not spatial_pairs) and (not numerical_pairs):
        return last_stage

    model_dtype = last_stage.dtype
    lr = adaptive_lr(last_stage, base_lr)

    use_cache   = caching_is_active(infinity)
    kv_snapshot = save_kv_cache(infinity) if use_cache else {}

    capture = VISTAAttentionCapture()
    hooks   = register_attention_hooks(infinity, capture)

    param_orig = last_stage.detach().float()
    param      = param_orig.clone().requires_grad_(True)

    ran_steps     = 0
    effective_max = n_steps

    active_losses = vista_cfg.get('active_losses', DEFAULT_ACTIVE_LOSSES)
    grad_tols = {
        name: LOSS_GRAD_TOLS.get(name, grad_tol)
        for name in active_losses
    }

    try:
        for step in range(n_steps):

            if step >= effective_max:
                logger.debug('[VISTA]   si=%d budget reached (%d/%d) - stopping',
                             si, effective_max, n_steps)
                break

            capture.reset()

            if use_cache:
                restore_kv_cache(infinity, kv_snapshot)

            with torch.enable_grad():
                forward_scale_for_grad(
                    infinity, param.to(model_dtype), si, scale_schedule,
                    cond_BD_or_gss, ca_kv, attn_fn, rope2d_freqs_grid,
                )
                attn_avg = capture.get_averaged(ph)
                if attn_avg is None:
                    logger.warning('[VISTA]   si=%d step %d: no attn map - aborting scale',
                                   si, step)
                    break

                loss_terms = {
                    name: LOSS_REGISTRY[name]['fn'](attn_avg, vista_cfg.get(LOSS_REGISTRY[name]['pairs_key'], []))
                    for name in active_losses
                    if vista_cfg.get(LOSS_REGISTRY[name]['pairs_key'], [])
                }

                dirs, grad_norms = combine_loss_gradients(loss_terms, param, grad_tols)
                if not dirs:
                    logger.debug('[VISTA]   si=%d step %d: all active losses below grad_tol '
                                 '- stopping  raw_norms=%s', si, step, grad_norms)
                    break

            loss_val = sum(l.item() for l in loss_terms.values())   # logging only
            logger.debug('[VISTA]   si=%d step %d  raw grad norms (pre-balance): %s',
                         si, step, grad_norms)

            if step == 0:
                severities = []
                for name, raw_norm in grad_norms.items():
                    tol = grad_tols.get(name, grad_tol)
                    if raw_norm > tol:
                        severities.append(math.log(raw_norm / tol) / math.log(n_steps + 1))
                severity_frac = min(max(severities, default=1.0), 1.0)
                effective_max = max(1, math.ceil(n_steps * severity_frac))

                logger.info('[VISTA]   si=%d res=%dx%d  loss_0=%.8f  grad_norms=%s  '
                            'N_eff=%d/%d  lr_eff=%.8f',
                            si, ph, pw, loss_val, grad_norms,
                            effective_max, n_steps, lr)

            with torch.no_grad():
                lr_map = vista_cfg.get('loss_lrs', {})
                update = torch.zeros_like(param)
                for name, d in dirs.items():
                    lr_name = lr * lr_map.get(name, 1.0)      # per-objective multiple of lr_eff
                    update += lr_name * d
                param -= update

            ran_steps += 1
            delta = (param.detach() - param_orig).abs()
            logger.debug('[VISTA]   step %d/%d  loss=%.8f  delta_max=%.8f',
                         step + 1, effective_max, loss_val, delta.max())

    finally:
        for h in hooks:
            h.remove()
        if use_cache:
            restore_kv_cache(infinity, kv_snapshot)

    if ran_steps > 0:
        delta = (param.detach() - param_orig).abs()
        logger.info('[VISTA]   si=%d done  ran_steps=%d/%d (N_max=%d)  '
                    'delta_mean=%.5f  delta_max=%.5f',
                    si, ran_steps, effective_max, n_steps,
                    delta.mean(), delta.max())

    return param.detach().to(model_dtype)
