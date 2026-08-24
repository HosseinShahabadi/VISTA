"""Build a per-prompt VISTA config: what to steer, and where.

`build_vista_cfg` runs once per prompt, before generation. It parses the prompt
into the three structures the objectives consume (attribute pairs, spatial
pairs, numerical pairs), resolves every span to T5 token ids, and picks the set
of scale indices to steer.

Scale selection follows the draft-then-refine intuition of next-scale
autoregressive generation: semantic layout is committed at the coarse scales,
so `steer_frac` keeps only the coarsest fraction of unique resolutions, and
degenerate grids (non-square, or 2x2 and smaller) are excluded. With the default
steer_frac=0.5 on the 1M schedule this yields three steered scales, 4x4 through
8x8.

Returns None when the prompt contains nothing steerable — the caller should then
generate normally.
"""

import logging
from typing import Optional

logger = logging.getLogger('vista')


def find_t5_ids(word: str, tokenizer, prompt: str) -> list:
    """Fallback token-id lookup, used only when the parser returns empty ids.

    Exact match only: matching on startswith() let short words contaminate
    longer ones sharing a prefix.
    """
    tokens  = tokenizer(prompt, return_tensors='pt').input_ids[0]
    decoded = tokenizer.convert_ids_to_tokens(tokens)
    word_lower = word.lower()
    return [i for i, tok in enumerate(decoded)
            if tok.lstrip('\u2581').lower() == word_lower]


def build_vista_cfg(
    prompt: str,
    scale_schedule: list,
    t5_path: str,
    spacy_weights_dir: Optional[str] = None,
    steer_frac: float = 0.5,
    n_steps: int = 5,
    lr: float = 0.005,
    contrastive: bool = True,
    lambda_c: float = 0.3,
    binding_threshold: float = 0.75,
    min_improvement: float = 1e-8,
    lambda_spatial_sep: float = 1.0,
    lambda_spatial_dir: float = 0.5,
    grad_tol: float = 1e-3,
    min_rel_improvement: float = 0.01,
    active_losses: Optional[list] = None,
    loss_lrs: Optional[dict] = None,
) -> Optional[dict]:
    """Build the VISTA config for one prompt.

    Parameters
    ----------
    steer_frac          Fraction of the coarsest unique resolutions to steer.
    n_steps             N_max: the step-budget ceiling. Actual steps N_eff <=
                        N_max are decided at runtime from ||g_0|| (see
                        grad_tol), so a generous ceiling is cheap — easy scales
                        exit after one step.
    lr / base_lr        Dimensionless step size: lr_eff = base_lr x RMS(features)
                        x sqrt(C). 0.005 perturbs each token by ~0.5% of its own
                        magnitude per step.
    grad_tol            Per-token gradient-norm floor, with two roles:
                          1. sets the step budget,
                             N_eff = ceil(N_max . clamp(log(||g_0||/grad_tol)
                                                        / log(N_max+1), 0, 1));
                          2. stops early once every objective falls below it.
                        Loss-type-agnostic, so binding, spatial and numeracy
                        terms are treated on the same footing.
                        NOTE: 1e-3 is calibrated for Infinity-2B. Infinity-8B
                        needs a far smaller value (e.g. 1e-12) or optimisation
                        is silently gated off at the finer steered scales.
    active_losses       Objective names to run; defaults to
                        `vista.registry.DEFAULT_ACTIVE_LOSSES`.
    loss_lrs            Optional per-objective multiplier on lr_eff.
    binding_threshold   Legacy gate (skip a scale whose step-0 loss is already
                        below -threshold). Retained in the config for
                        reproducibility of earlier ablations; the adaptive
                        budget above is what actually governs stopping.
    min_improvement,
    min_rel_improvement Legacy loss-delta stopping guards, superseded by the
                        gradient-norm criterion; kept for the same reason.
    lambda_spatial_sep,
    lambda_spatial_dir  Weights for `composite_spatial_loss`, which is an
                        ablation variant rather than the default spatial term.
    """
    base_lr = lr

    # Imported lazily and deliberately: `vista.parsing.extractor` pins the
    # parser (spaCy) to CPU by clearing CUDA_VISIBLE_DEVICES at import time, so
    # it must not be imported before the generator has initialised CUDA.
    from vista.parsing.extractor import (
        extract_numerical_pairs,
        extract_pairs,
        extract_spatial_pairs,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(t5_path)
    pairs = extract_pairs(prompt, t5_path=t5_path,
                          spacy_weights_dir=spacy_weights_dir)

    spatial_pairs = extract_spatial_pairs(prompt, t5_path=t5_path,
                                          spacy_weights_dir=spacy_weights_dir)

    numerical_pairs = extract_numerical_pairs(prompt, t5_path=t5_path,
                                              spacy_weights_dir=spacy_weights_dir)

    has_adj       = len(pairs) > 0
    has_spatial   = len(spatial_pairs) > 0
    has_numerical = len(numerical_pairs) > 0

    if not has_adj and not has_spatial and not has_numerical:
        logger.warning('[VISTA] No adj-noun or spatial pairs found in: %r', prompt)
        return None

    # Resolve empty token ids (the rule-based fallback leaves them empty)
    for pair in pairs:
        if not pair['adj_t5_ids']:
            pair['adj_t5_ids']  = find_t5_ids(pair['adj'],  tokenizer, prompt)
        if not pair['noun_t5_ids']:
            pair['noun_t5_ids'] = find_t5_ids(pair['noun'], tokenizer, prompt)

    for pair in spatial_pairs:
        if not pair['noun_a_t5_ids']:
            pair['noun_a_t5_ids'] = find_t5_ids(pair['noun_a'], tokenizer, prompt)
        if not pair['noun_b_t5_ids']:
            pair['noun_b_t5_ids'] = find_t5_ids(pair['noun_b'], tokenizer, prompt)

    for pair in numerical_pairs:
        if not pair['noun_t5_ids']:
            pair['noun_t5_ids'] = find_t5_ids(pair['noun'], tokenizer, prompt)
        if not pair['number_t5_ids']:
            pair['number_t5_ids'] = find_t5_ids(pair['number_surface'], tokenizer, prompt)

    # Drop pairs that still have no ids
    spatial_pairs   = [p for p in spatial_pairs   if p['noun_a_t5_ids'] and p['noun_b_t5_ids']]
    numerical_pairs = [p for p in numerical_pairs if p['noun_t5_ids']]

    if not pairs and not spatial_pairs and not numerical_pairs:
        logger.warning('[VISTA] No pairs with valid token ids.')
        return None

    if pairs:
        logger.info('[VISTA] Adj-noun pairs:')
        for p in pairs:
            logger.info("  (%r, %r)  adj_t5=%s  noun_t5=%s",
                        p['adj'], p['noun'], p['adj_t5_ids'], p['noun_t5_ids'])
    else:
        logger.info('[VISTA] No adj-noun pairs.')

    if spatial_pairs:
        logger.info('[VISTA] Spatial pairs:')
        for p in spatial_pairs:
            logger.info("  (%r %r %r)  axis=%s  side=%s  noun_a_t5=%s  noun_b_t5=%s",
                        p['noun_a'], p['relation'], p['noun_b'], p['axis'],
                        p['noun_a_side'], p['noun_a_t5_ids'], p['noun_b_t5_ids'])
    else:
        logger.info('[VISTA] No spatial pairs.')

    if numerical_pairs:
        logger.info('[VISTA] Numerical pairs:')
        for p in numerical_pairs:
            logger.info("  (%r %r)  relation=%s  noun_t5=%s  number_t5=%s",
                        p['number'], p['noun'], p['relation'],
                        p['noun_t5_ids'], p['number_t5_ids'])
    else:
        logger.info('[VISTA] No numerical pairs.')

    # Scale pre-filter: coarsest steer_frac, excluding degenerate sizes
    all_res   = sorted({h for (_, h, w) in scale_schedule})
    n_target  = max(1, int(len(all_res) * steer_frac))
    steer_res = set(all_res[:n_target])

    optimize_scales = [
        si for si, (_, ph, pw) in enumerate(scale_schedule)
        if ph in steer_res and ph == pw and ph * pw > 4
    ]

    if loss_lrs is None:
        loss_lrs = {'binding': 1.0, 'spatial': 1.0, 'depth': 1.0}

    logger.info('[VISTA] Steered scales: %s  res=%s  base_lr=%s  grad_tol=%s  '
                'has_adj=%s  has_spatial=%s  has_numerical=%s',
                optimize_scales, sorted(steer_res), base_lr, grad_tol,
                has_adj, has_spatial, has_numerical)

    cfg = {
        # parsed structures
        'pairs':               pairs,
        'spatial_pairs':       spatial_pairs,
        'numerical_pairs':     numerical_pairs,
        # scale selection
        'optimize_scales':     optimize_scales,
        # optimisation
        'n_steps':             n_steps,
        'base_lr':             base_lr,
        'grad_tol':            grad_tol,
        'loss_lrs':            loss_lrs,
        'contrastive':         contrastive,
        'lambda_c':            lambda_c,
        # legacy stopping guards (see docstring)
        'binding_threshold':   binding_threshold,
        'min_improvement':     min_improvement,
        'min_rel_improvement': min_rel_improvement,
        # spatial loss weights (composite_spatial_loss ablation)
        'lambda_spatial_sep':  lambda_spatial_sep,
        'lambda_spatial_dir':  lambda_spatial_dir,
        'centroid_cache':      {},
        # meta
        # NOTE: evaluated after the id-resolution filter above, so these
        # reflect what will actually be steered, not what the parser found.
        'has_adj_pairs':       len(pairs) > 0,
        'has_spatial_pairs':   len(spatial_pairs) > 0,
        'has_numerical_pairs': len(numerical_pairs) > 0,
    }

    if active_losses is not None:
        cfg['active_losses'] = active_losses

    return cfg
