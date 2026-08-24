#!/usr/bin/env python3
"""Generate the GenEval image set.

Example (VISTA on Infinity-2B):

    python scripts/generate_geneval.py \
        --model-path weights/infinity_2b_reg.pth \
        --vae-path   weights/infinity_vae_d32_reg.pth \
        --metadata   Benchmarks/geneval/prompts/evaluation_metadata.jsonl \
        --save-dir   outputs/geneval

Then score with the official GenEval evaluator:

    python evaluation/evaluate_images.py outputs/geneval/<run_name>_GenEval ...
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
    parser.add_argument('--metadata', required=True,
                        help='Path to GenEval evaluation_metadata.jsonl.')
    parser.add_argument('--n-samples', type=int, default=4,
                        help='Images per prompt (default: 4, as in GenEval).')
    parser.add_argument('--tags', nargs='+', default=None,
                        help='Restrict to GenEval tags, e.g. colors position color_attr.')
    parser.add_argument('--num-prompts', type=int, default=None,
                        help='Reproducibly subsample the filtered prompt set.')
    parser.add_argument('--save-grid', action='store_true',
                        help='Also write a contact sheet per prompt.')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    setup_logging(args.verbose)

    from vista.benchmarks import GenEvalGenerator

    settings = build_settings(args)
    model = build_model(args)

    generator = GenEvalGenerator(
        model_name=args.run_name or describe(args, settings),
        metadata_path=args.metadata,
        settings=settings,
    )
    generator.generate_images(
        model,
        args.save_dir,
        n_samples=args.n_samples,
        tag_filter=args.tags,
        num_prompts=args.num_prompts,
        sample_seed=args.seed,
        save_grid=args.save_grid,
        resume=not args.no_resume,
    )


if __name__ == '__main__':
    main()
