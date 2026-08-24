"""Shared CLI plumbing for the generation scripts.

Import-order note: nothing here (and nothing in the scripts) may import
`vista.parsing` at module level — it pins the process to CPU. `vista.config`
imports it lazily, after the model is on the GPU.
"""

import argparse
import logging
import os
import sys

# Make the repo root importable when a script is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vista.benchmarks import GenerationSettings                    # noqa: E402
from vista.models import InfinityWrapper2B, InfinityWrapper8B      # noqa: E402


def add_model_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group('model')
    g.add_argument('--model', choices=['2b', '8b'], default='2b',
                   help='Infinity variant to load.')
    g.add_argument('--model-path', required=True,
                   help='Infinity checkpoint (a .pth for 2B, a shard directory for 8B).')
    g.add_argument('--vae-path', required=True,
                   help='Infinity VAE checkpoint (d32 for 2B, d56/f8 patchify for 8B).')
    g.add_argument('--t5-path', default=os.environ.get('VISTA_T5_PATH', 'weights/flan-t5-xl'),
                   help='flan-t5-xl directory used as the text encoder.')
    g.add_argument('--gpu', type=int, default=0)
    g.add_argument('--pn', default='1M', help='Resolution preset (default: 1M).')
    g.add_argument('--no-scale-kv', action='store_true',
                   help='Disable ScaleKV — use for the unmodified Infinity baseline.')
    g.add_argument('--scale-kv-window', type=int, default=8)
    g.add_argument('--scale-kv-capacity', type=int, default=400)


def add_vista_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group('VISTA')
    g.add_argument('--no-vista', action='store_true',
                   help='Generate without steering (baseline run).')
    g.add_argument('--n-steps', type=int, default=5,
                   help='N_max, the per-scale step-budget ceiling (default: 5).')
    g.add_argument('--lr', type=float, default=0.005,
                   help='Dimensionless base_lr; lr_eff = base_lr x RMS x sqrt(C).')
    g.add_argument('--steer-frac', type=float, default=None,
                   help='Fraction of coarsest resolutions to steer '
                        '(default 0.5 = 3 steered scales). Overrides --num-scales.')
    g.add_argument('--num-scales', type=int, default=3, choices=[0, 1, 2, 3],
                   help='Calibrated steer_frac preset (default: 3).')
    g.add_argument('--grad-tol', type=float, default=None,
                   help='Per-token gradient-norm floor. Default 1e-3 for 2B, '
                        '1e-12 for 8B (the 2B value gates 8B off entirely).')
    g.add_argument('--active-losses', nargs='+', default=None,
                   metavar='NAME',
                   help="Objectives to run, e.g. --active-losses binding spatial. "
                        "Default: binding spatial depth.")
    g.add_argument('--spacy-weights-dir', default=None,
                   help='Directory holding en_core_web_trf; falls back to the '
                        'rule-based parser when unavailable.')
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--cfg', type=float, default=4.0)
    g.add_argument('--tau', type=float, default=1.0)


def add_output_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group('output')
    g.add_argument('--save-dir', required=True)
    g.add_argument('--run-name', default=None,
                   help='Label for the output folder (default: derived from settings).')
    g.add_argument('-v', '--verbose', action='store_true',
                   help='Log every optimisation step, not just the per-scale summary.')


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger('vista').setLevel(logging.DEBUG if verbose else logging.INFO)


def build_model(args):
    """Instantiate the requested Infinity wrapper."""
    cls = InfinityWrapper2B if args.model == '2b' else InfinityWrapper8B
    return cls(
        model_path=args.model_path,
        vae_path=args.vae_path,
        text_encoder_ckpt=args.t5_path,
        gpu_id=args.gpu,
        pn=args.pn,
        use_scale_kv=not args.no_scale_kv,
        window_size=args.scale_kv_window,
        max_capacity=args.scale_kv_capacity,
    )


def build_settings(args) -> GenerationSettings:
    """Turn parsed CLI args into a GenerationSettings."""
    if args.steer_frac is not None:
        settings = GenerationSettings(steer_frac=args.steer_frac)
    else:
        settings = GenerationSettings.from_num_scales(args.num_scales)

    # The 2B default silently disables steering on 8B: its per-token gradient
    # norms are orders of magnitude smaller than 1e-3.
    grad_tol = args.grad_tol
    if grad_tol is None:
        grad_tol = 1e-12 if args.model == '8b' else 1e-3

    settings.use_vista = not args.no_vista
    settings.n_steps = args.n_steps
    settings.lr = args.lr
    settings.grad_tol = grad_tol
    settings.active_losses = args.active_losses
    settings.spacy_weights_dir = args.spacy_weights_dir
    settings.seed = args.seed
    settings.cfg = args.cfg
    settings.tau = args.tau
    return settings


def describe(args, settings: GenerationSettings) -> str:
    """One-line summary of the run, also used as a default folder name."""
    if not settings.use_vista:
        return f'{args.model}_baseline_seed{settings.seed}'
    return (f'{args.model}_vista_steer{settings.steer_frac}_lr{settings.lr}'
            f'_n{settings.n_steps}_seed{settings.seed}')
