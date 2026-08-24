#!/usr/bin/env python3
"""Generate the T2I-CompBench image set.

Example (VISTA on Infinity-2B, paper configuration):

    python scripts/generate_t2i_compbench.py \
        --model-path weights/infinity_2b_reg.pth \
        --vae-path   weights/infinity_vae_d32_reg.pth \
        --t5-path    weights/flan-t5-xl \
        --prompts-dir Benchmarks/T2I_CompBench/examples/dataset \
        --save-dir   outputs/t2i_compbench

Baseline (same code path, steering off):  add --no-vista
"""

import argparse

from _common import (
    add_model_args,
    add_output_args,
    add_vista_args,
    build_model,
    build_settings,
    describe,
    setup_logging,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    add_vista_args(parser)
    add_output_args(parser)
    parser.add_argument('--prompts-dir', required=True,
                        help='Directory with the official <category>_val.txt files.')
    parser.add_argument('--categories', nargs='+', default=None,
                        help='Subset of categories to run (default: all six).')
    parser.add_argument('--no-resume', action='store_true',
                        help='Regenerate images that already exist on disk.')
    args = parser.parse_args()

    setup_logging(args.verbose)

    from vista.benchmarks import T2ICompBenchGenerator
    from vista.benchmarks.t2i_compbench import DEFAULT_CATEGORIES, OPTIONAL_CATEGORIES

    categories = None
    if args.categories:
        known = {**DEFAULT_CATEGORIES, **OPTIONAL_CATEGORIES}
        missing = [c for c in args.categories if c not in known]
        if missing:
            parser.error(f'unknown categories {missing}; choose from {sorted(known)}')
        categories = {c: known[c] for c in args.categories}

    settings = build_settings(args)
    model = build_model(args)

    generator = T2ICompBenchGenerator(
        model_name=args.run_name or describe(args, settings),
        prompts_dir=args.prompts_dir,
        settings=settings,
        categories=categories,
    )
    generator.generate_images(model, args.save_dir, resume=not args.no_resume)


if __name__ == '__main__':
    main()
