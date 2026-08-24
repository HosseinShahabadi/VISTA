# VISTA — Visual Autoregressive Semantic Test-time Alignment

Reference implementation of VISTA: gradient-based test-time optimisation for
compositional alignment in next-scale visual autoregressive (VAR) generation,
built on [Infinity](https://github.com/FoundationVision/Infinity) (2B and 8B).

VISTA needs no training, no extra parameters and no reward model. At a small set
of coarse scales it treats the token features the backbone is about to decode as
a free variable, and runs a few steps of normalised gradient descent on a
compositional objective read off the model's own cross-attention.

```
prompt ──▶ parse ──▶ per-scale steering ──▶ sample ──▶ image
                     └─ L_compos = L_entity + L_bind + L_rel
```

---

## Why this is not a straight port of diffusion-side attention steering

Attend-and-Excite / SynGen-style steering assumes a continuous latent that is
denoised over many timesteps. Next-scale VAR gives you neither. Three concrete
barriers, and what this code does about them:

| Barrier | Consequence | Solution here |
|---|---|---|
| Sampling is **discrete** — tokens are committed per scale, with no latent to nudge afterwards | A scale's semantics become irrevocable once sampled | Steer the pre-logit features, before the token distribution is sampled (`vista/optimization.py`) |
| **Gradient dilution** — a global L2 norm over a `[bs, L, C]` tensor scales each element by ~`1/sqrt(N)` | Updates vanish at exactly the scales that matter | Per-token normalisation, `g.norm(dim=-1, keepdim=True)`, with a per-token displacement budget |
| **KV caching** — the extra forward pass writes into the autoregressive cache | Steering silently corrupts generation | Snapshot / restore around every optimisation forward, and again in `finally` (`vista/kv_cache.py`) |

Two further details that took real debugging and are easy to get wrong:

- **Adam is unsuitable.** Across 32 layers of gradient amplification its
  per-coordinate rescaling makes step sizes wildly inconsistent between scales.
  Normalised gradient descent with a fixed per-step displacement is stable.
- **`grad_tol` is model-specific.** The 2B default (`1e-3`) is far larger than
  8B's typical per-token gradient norms and silently zeroes out optimisation at
  the finer steered scales. Use `~1e-12` for 8B — the scripts do this
  automatically when `--model 8b` is passed.

---

## Install

```bash
git clone <repo-url> && cd VISTA
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_trf     # optional, improves parsing
```

Tested on Python 3.10, torch 2.5.1, CUDA 12.4.

Fetch the Infinity weights (from the Infinity release) and a `flan-t5-xl`
checkpoint, then point the scripts at them:

```
weights/
├── infinity_2b_reg.pth
├── infinity_vae_d32_reg.pth
├── infinity_8b_weights/                     # sharded
├── infinity_vae_d56_f8_14_patchify.pth
└── flan-t5-xl/
```

`export VISTA_T5_PATH=weights/flan-t5-xl` saves repeating `--t5-path`.

---

## Quick start

```bash
python scripts/generate_prompts.py \
    --model-path weights/infinity_2b_reg.pth \
    --vae-path   weights/infinity_vae_d32_reg.pth \
    --save-dir   outputs/qualitative \
    --prompt "a red bench and a yellow clock" \
    --prompt "a cat behind a laptop"
```

Drop `--no-vista` in to get the unmodified backbone through the identical code
path, which is what you want for a controlled baseline.

### As a library

```python
from vista import build_vista_cfg
from vista.models import InfinityWrapper2B

model = InfinityWrapper2B(model_path=..., vae_path=..., text_encoder_ckpt=...)
image = model.generate("a red bench and a yellow clock", use_vista=True)
```

Or drive the optimiser directly: `build_vista_cfg(prompt, scale_schedule,
t5_path)` returns the config, and the backbone calls `run_scale_optimization`
at each scale listed in `cfg['optimize_scales']`.

---

## Reproducing the paper

**T2I-CompBench / T2I-CompBench++**

```bash
python scripts/generate_t2i_compbench.py \
    --model-path weights/infinity_2b_reg.pth \
    --vae-path   weights/infinity_vae_d32_reg.pth \
    --prompts-dir path/to/T2I-CompBench/examples/dataset \
    --save-dir   outputs/t2i_compbench
```

Writes `<save_dir>/<run>_T2I/<category>/samples/`, the layout the official
T2I-CompBench evaluators consume. Interrupted runs resume by default.

**GenEval**

```bash
python scripts/generate_geneval.py \
    --model-path weights/infinity_2b_reg.pth \
    --vae-path   weights/infinity_vae_d32_reg.pth \
    --metadata   path/to/geneval/prompts/evaluation_metadata.jsonl \
    --save-dir   outputs/geneval \
    --n-samples 4
```

Writes `<index>/samples/<n>.png` plus `metadata.jsonl` per prompt, ready for
GenEval's `evaluate_images.py`. Sample *n* uses seed *n*, per the GenEval
reference sampler.

**Infinity-8B** — swap in the 8B weights and `--model 8b`; `grad_tol` and the
checkpoint/VAE configuration adjust automatically.

**Baselines** — `--no-vista` disables steering, `--no-scale-kv` disables ScaleKV.
Both flags are independent, so all four combinations are runnable.

### Configurations used in the paper

| Setting | Value | Flag |
|---|---|---|
| Steered scales | 3 (4×4 → 8×8) | `--num-scales 3` (`steer_frac=0.5`) |
| Step budget `N_max` | 5 | `--n-steps 5` |
| Base learning rate | 0.005 | `--lr 0.005` |
| `grad_tol` | 1e-3 (2B) / 1e-12 (8B) | `--grad-tol` |
| Active objectives | binding, spatial, depth | `--active-losses` |
| CFG / tau | 4.0 / 1.0 | `--cfg`, `--tau` |

Numeracy (`L_num`) is implemented but **disabled in every reported result**; it
is absent from `DEFAULT_ACTIVE_LOSSES` in `vista/registry.py`.

---

## How it works

**Scale selection.** Coarse scales fix layout, fine scales fix texture, so
steering only the coarsest `steer_frac` of unique resolutions captures the
semantics while leaving most of the schedule untouched. Non-square and ≤2×2
grids are skipped.

**Differentiable capture** (`vista/attention.py`). `autoregressive_infer_cfg`
runs under `@torch.no_grad()`, so the hook recomputes attention probabilities
inside an `enable_grad` region. Only the query projection carries gradient;
keys are recomputed under `no_grad` (stop-gradient on K), since the text stream
is fixed at test time.

**Multi-objective combination** (`vista/optimization.py`). All active objectives
share one forward/backward pass. Each gradient is normalised to a unit per-token
direction before summing, so no term dominates by raw scale. An objective whose
raw gradient norm falls below its own tolerance contributes nothing that step —
recomputed fresh each step, so it can re-enter later and no retirement state
leaks across scales. Optimisation stops when every objective is below tolerance
or the budget is spent.

**Adaptive step budget.** Step 0 is needed anyway, so its gradient norms are
free information:

```
N_eff = ceil(N_max · clamp( log(‖g₀‖ / grad_tol) / log(N_max + 1), 0, 1 ))
```

Badly aligned scales get the full budget; nearly aligned ones exit after one
step. This makes a generous `N_max` cheap.

**Step size.** `lr_eff = base_lr × RMS(features) × sqrt(C)`, so each token moves
by roughly `base_lr` times its own magnitude — prompt- and scale-invariant,
which removes `lr` as a per-prompt hyperparameter.

### Objectives

| Module | Term | Default | Also implemented |
|---|---|---|---|
| `losses/binding.py` | `L_bind` | CONFORM contrastive | SynGen, Divide & Bind, KL/JSD, focal, SupCon, overlap |
| `losses/entity.py` | `L_entity` | A-STAR segregation | A-STAR retention, presence/balance |
| `losses/spatial.py` | `L_rel` | PoS (PSG) | half-field, relative centroid, IoU, Gaussian target |
| `losses/depth.py` | Boundary Containment | separation + adjacency | entropy depth proxy |
| `losses/numeracy.py` | `L_num` *(disabled)* | cluster count | entropy, softmask, template |

Every objective has the signature `(attn, pairs) -> scalar`, which is what lets
`vista/registry.py` swap terms without touching the optimiser:

```python
from vista.registry import LOSS_REGISTRY, available_losses
from vista.losses.binding import syngen_loss
LOSS_REGISTRY['binding']['fn'] = syngen_loss     # ablation, one line
```

**Boundary Containment** deserves a note, since it is the one objective without
a direct antecedent. For "B behind A" it imposes two constraints: A and B must
not overlap (a *separation* term, which builds on A-STAR's segregation
principle rather than introducing it), and the portion of B's perimeter facing A
must be *adjacent to* A — random convex combinations of the facing perimeter
tokens have to land inside A's high-attention region. The second constraint is
the novel part: it distinguishes "B peeks out from behind A" from "B is merely
somewhere else", which separation alone cannot.

---

## Layout

```
vista/                    the method
├── attention.py          differentiable cross-attention capture (stop-grad on K)
├── kv_cache.py           snapshot / restore around optimisation forwards
├── forward.py            block-chunk forward that keeps `param` a leaf
├── optimization.py       gradient combination, adaptive step size, per-scale loop
├── config.py             per-prompt config: parse, resolve token ids, pick scales
├── registry.py           objective registry + defaults + per-loss tolerances
├── losses/               binding | entity | spatial | depth | numeracy | utils
├── parsing/              prompt → attribute / spatial / numerical structures
├── models/               Infinity wrappers (2B, 8B) + attention instrumentation
└── benchmarks/           T2I-CompBench, GenEval, free-form prompt runners

infinity/                 vendored Infinity backbone (one file patched, see below)
tools/run_infinity.py     vendored Infinity loader / sampler
scale_kv/                 vendored ScaleKV + cache budgets
scripts/                  CLI entry points
```

### Changes to the vendored backbone

`infinity/models/infinity.py` is upstream Infinity with one insertion:
`autoregressive_infer_cfg` gains a `vista_cfg=None` argument, and at each scale
listed in `vista_cfg['optimize_scales']` it calls `run_scale_optimization` on
`last_stage` before the usual block loop. If steering raises, the exception is
logged and generation continues with unmodified features. `tools/run_infinity.py`
threads `vista_cfg` through `gen_one_img`. Everything else is untouched, so
diffing against upstream Infinity shows exactly the integration surface.

`scale_kv/` is ScaleKV with the cache-budget path made package-relative
(`scale_kv/budgets/`, overridable via `VISTA_SCALEKV_BUDGET`).

---

## Notes and gotchas

- **`vista.parsing` must be imported lazily.** It clears
  `CUDA_VISIBLE_DEVICES` at import time to pin spaCy to CPU. `vista.config`
  imports it inside the function for this reason; importing it at process start
  would hide the GPU from torch. `vista/__init__.py` deliberately does not
  import it.
- **8B halves the token grid** (`apply_spatial_patchify=1`), so a scale index
  maps to a smaller `(ph, pw)` than on 2B and can fall below the minimum-grid
  guard in `run_scale_optimization`. Check the logged steered scales when
  moving between models.
- **Logging.** The package logs through the `vista` logger and stays silent
  when imported as a library. Scripts configure it; `-v` adds per-step gradient
  norms and deltas.
- **Parser fallback.** Without spaCy weights the rule-based parser is used
  automatically. Coverage is lower; the log line listing extracted pairs is
  worth checking on new prompt distributions.

---

## Citation

```bibtex
@inproceedings{vista,
  title     = {VISTA: Visual Autoregressive Semantic Test-time Alignment},
  author    = {Anonymous},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

## Acknowledgements

Built on [Infinity](https://github.com/FoundationVision/Infinity) and
[ScaleKV](https://github.com/Peinlight/ScaleKV). The objectives adapt ideas from
SynGen, Attend-and-Excite, CONFORM, A-STAR, Divide & Bind and PSG; each loss
module documents its source and any deviation from the original formulation.
Vendored components retain their original licences.
