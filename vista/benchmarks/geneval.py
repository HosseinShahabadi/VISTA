"""GenEval generation.

Expects the official `evaluation_metadata.jsonl`. Images are written to

    <save_dir>/<model_name>_GenEval/<index:05d>/samples/<sample:05d>.png
    <save_dir>/<model_name>_GenEval/<index:05d>/metadata.jsonl

which is the layout `evaluation/evaluate_images.py` from GenEval expects. Each
prompt is sampled `n_samples` times; following the GenEval reference sampler,
sample n uses seed n so the sample set is reproducible.
"""

import json
import logging
import os
import random
from typing import List, Optional

import torch
from tqdm import tqdm, trange

from vista.benchmarks.base import BaseGenerator, GenerationSettings

logger = logging.getLogger('vista')


class GenEvalGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str,
        metadata_path: str,
        settings: Optional[GenerationSettings] = None,
    ):
        super().__init__(model_name, settings)
        self.metadata_path = metadata_path
        with open(metadata_path) as fp:
            self.metadatas = [json.loads(line) for line in fp]

    def generate_images(
        self,
        model,
        save_dir: str,
        n_samples: int = 4,
        tag_filter: Optional[List[str]] = None,
        num_prompts: Optional[int] = None,
        sample_seed: int = 42,
        save_grid: bool = False,
        resume: bool = True,
    ) -> None:
        """Generate `n_samples` images for each selected GenEval prompt.

        tag_filter   restrict to GenEval tags, e.g. ['colors', 'position',
                     'color_attr']; None runs the full benchmark.
        num_prompts  reproducibly subsample the filtered set (uses sample_seed).
        save_grid    additionally write a contact sheet per prompt.
        """
        root = os.path.join(save_dir, f'{self.model_name}_GenEval')
        os.makedirs(root, exist_ok=True)

        if tag_filter is not None:
            tag_set = {tag_filter} if isinstance(tag_filter, str) else set(tag_filter)
            selected = [(i, m) for i, m in enumerate(self.metadatas) if m.get('tag') in tag_set]
        else:
            selected = list(enumerate(self.metadatas))

        if num_prompts is not None and num_prompts < len(selected):
            rng = random.Random(sample_seed)
            selected = sorted(rng.sample(selected, num_prompts), key=lambda x: x[0])

        logger.info('GenEval: %d prompts x %d samples -> %s',
                    len(selected), n_samples, root)

        for index, metadata in tqdm(selected, desc='Generating'):
            prompt_dir = os.path.join(root, f'{index:0>5}')
            sample_dir = os.path.join(prompt_dir, 'samples')
            os.makedirs(sample_dir, exist_ok=True)
            with open(os.path.join(prompt_dir, 'metadata.jsonl'), 'w') as fp:
                json.dump(metadata, fp)

            prompt = metadata['prompt']
            all_samples = []

            with torch.no_grad():
                for n in trange(n_samples, desc='Sampling', leave=False):
                    out_path = os.path.join(sample_dir, f'{n:05}.png')
                    if resume and os.path.exists(out_path) and not save_grid:
                        continue
                    sample = self._generate(model, prompt, seed=n)
                    sample.save(out_path)
                    if save_grid:
                        all_samples.append(sample)

            if save_grid and all_samples:
                self._save_grid(all_samples, os.path.join(sample_dir, 'grid.png'))

    @staticmethod
    def _save_grid(samples, path: str) -> None:
        import numpy as np
        from einops import rearrange
        from PIL import Image
        from torchvision.transforms import ToTensor
        from torchvision.utils import make_grid

        grid = torch.stack([ToTensor()(s) for s in samples], 0)
        grid = make_grid(grid, nrow=len(samples))
        grid = 255.0 * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        Image.fromarray(grid.astype(np.uint8)).save(path)
