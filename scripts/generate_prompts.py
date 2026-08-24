#!/usr/bin/env python3
"""Generate images for arbitrary prompts (qualitative figures, quick checks).

    python scripts/generate_prompts.py \
        --model-path weights/infinity_2b_reg.pth \
        --vae-path   weights/infinity_vae_d32_reg.pth \
        --save-dir   outputs/qualitative \
        --prompt "a red bench and a yellow clock" \
        --prompt "a cat behind a laptop"

Prompts can also come from a file, one per line, via --prompt-file.
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
    parser.add_argument('--prompt', action='append', default=[],
                        help='A prompt; repeat the flag for several.')
    parser.add_argument('--prompt-file', default=None,
                        help='Text file with one prompt per line.')
    args = parser.parse_args()

    setup_logging(args.verbose)

    prompts = list(args.prompt)
    if args.prompt_file:
        with open(args.prompt_file, encoding='utf-8') as f:
            prompts += [line.strip() for line in f if line.strip()]
    if not prompts:
        parser.error('give at least one --prompt or a --prompt-file')

    from vista.benchmarks import PromptGenerator

    settings = build_settings(args)
    model = build_model(args)

    generator = PromptGenerator(model_name=args.model, settings=settings)
    generator.generate_images(
        model, args.save_dir, prompts,
        run_name=args.run_name or describe(args, settings),
    )


if __name__ == '__main__':
    main()
