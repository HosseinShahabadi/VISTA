"""VISTA — Visual Autoregressive Semantic Test-time Alignment.

Gradient-based test-time optimisation for compositional alignment in next-scale
visual autoregressive generation.

Typical use is indirect: build a config for a prompt and hand it to the
generator, which calls `run_scale_optimization` at each selected scale.

    from vista import build_vista_cfg
    cfg = build_vista_cfg(prompt, scale_schedule, t5_path=T5_PATH)
    image = gen_one_img(..., vista_cfg=cfg)

`vista.parsing` is deliberately NOT imported here — see its docstring.
"""

from vista.attention import VISTAAttentionCapture, register_attention_hooks
from vista.config import build_vista_cfg
from vista.forward import forward_scale_for_grad
from vista.kv_cache import caching_is_active, restore_kv_cache, save_kv_cache
from vista.optimization import adaptive_lr, combine_loss_gradients, run_scale_optimization
from vista.registry import DEFAULT_ACTIVE_LOSSES, LOSS_GRAD_TOLS, LOSS_REGISTRY

__version__ = '1.0.0'

__all__ = [
    'build_vista_cfg',
    'run_scale_optimization',
    'combine_loss_gradients',
    'adaptive_lr',
    'VISTAAttentionCapture',
    'register_attention_hooks',
    'forward_scale_for_grad',
    'caching_is_active',
    'save_kv_cache',
    'restore_kv_cache',
    'LOSS_REGISTRY',
    'LOSS_GRAD_TOLS',
    'DEFAULT_ACTIVE_LOSSES',
]
