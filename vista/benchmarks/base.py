"""Shared plumbing for the benchmark runners.

Every runner takes a `GenerationSettings` describing how to generate — the same
settings object drives the VISTA runs and the unmodified baselines, so the only
difference between an ablation and its control is one field.
"""

import dataclasses
import logging
from typing import List, Optional

logger = logging.getLogger('vista')

# `steer_frac` keeps the coarsest fraction of unique resolutions in the schedule.
# On the 1M schedule these are the values that yield 1..4 steered scales; the
# paper default is 3 steered scales (4x4 through 8x8), i.e. steer_frac=0.5.
STEER_FRAC_BY_NUM_SCALES = {
    0: 0.3,
    1: 0.35,
    2: 0.4,
    3: 0.5,
}


def steer_frac_for_num_scales(num_scales: int) -> float:
    if num_scales not in STEER_FRAC_BY_NUM_SCALES:
        raise ValueError(
            f'num_scales={num_scales} has no calibrated steer_frac; '
            f'choose one of {sorted(STEER_FRAC_BY_NUM_SCALES)} or pass steer_frac directly.'
        )
    return STEER_FRAC_BY_NUM_SCALES[num_scales]


@dataclasses.dataclass
class GenerationSettings:
    """How to generate one image. Defaults reproduce the paper configuration."""

    use_vista: bool = True
    n_steps: int = 5                       # N_max, the step-budget ceiling
    lr: float = 0.005                      # dimensionless base_lr
    steer_frac: float = 0.5                # 3 steered scales on the 1M schedule
    grad_tol: float = 1e-3                 # use ~1e-12 for Infinity-8B
    active_losses: Optional[List[str]] = None   # None -> registry default
    spacy_weights_dir: Optional[str] = None
    seed: int = 42
    cfg: float = 4.0
    tau: float = 1.0

    @classmethod
    def from_num_scales(cls, num_scales: int, **kwargs):
        return cls(steer_frac=steer_frac_for_num_scales(num_scales), **kwargs)

    def to_generate_kwargs(self) -> dict:
        return dict(
            cfg=self.cfg,
            tau=self.tau,
            use_vista=self.use_vista,
            vista_n_steps=self.n_steps,
            vista_lr=self.lr,
            vista_steer_frac=self.steer_frac,
            vista_grad_tol=self.grad_tol,
            vista_active_losses=self.active_losses,
            spacy_weights_dir=self.spacy_weights_dir,
        )


class BaseGenerator:
    """Base class for benchmark image generators."""

    def __init__(self, model_name: str, settings: Optional[GenerationSettings] = None):
        self.model_name = model_name
        self.settings = settings or GenerationSettings()

    def _generate(self, model, prompt: str, seed: Optional[int] = None):
        return model.generate(
            prompt=prompt,
            seed=self.settings.seed if seed is None else seed,
            **self.settings.to_generate_kwargs(),
        )
