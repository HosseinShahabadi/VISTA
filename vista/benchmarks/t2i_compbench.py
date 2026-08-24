"""T2I-CompBench / T2I-CompBench++ generation.

Expects the official prompt files, i.e. `<prompts_dir>/<category>_val.txt`, one
prompt per line. Images are written to

    <save_dir>/<model_name>_T2I/<category>/samples/<prompt>_<index>.png

which is the layout the official T2I-CompBench evaluation scripts consume.
"""

import logging
import os
from typing import Dict, List, Optional

from tqdm import tqdm

from vista.benchmarks.base import BaseGenerator, GenerationSettings

logger = logging.getLogger('vista')

# Benchmark category -> prompt file stem. Both the compositional categories used
# in the paper and the remaining T2I-CompBench++ ones are listed; pass
# `categories=` to restrict the run.
DEFAULT_CATEGORIES: Dict[str, str] = {
    'color':      'color',
    'shape':      'shape',
    'texture':    'texture',
    '2d_spatial': 'spatial',
    '3d_spatial': '3d_spatial',
    'complex':    'complex',
}

OPTIONAL_CATEGORIES: Dict[str, str] = {
    'non_spatial': 'non_spatial',
    'numeracy':    'numeracy',
}


class T2ICompBenchGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str,
        prompts_dir: str,
        settings: Optional[GenerationSettings] = None,
        categories: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model_name, settings)
        self.prompts_dir = prompts_dir
        self.category_files = dict(categories or DEFAULT_CATEGORIES)
        self.categories = {
            name: self._load_prompts(stem)
            for name, stem in self.category_files.items()
        }

    def _load_prompts(self, stem: str) -> List[str]:
        path = os.path.join(self.prompts_dir, f'{stem}_val.txt')
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def generate_images(self, model, save_dir: str, resume: bool = True) -> None:
        """Generate every prompt in every configured category.

        `resume=True` skips prompts whose output file already exists, so an
        interrupted sweep can be restarted without regenerating.
        """
        for category, prompts in self.categories.items():
            out_dir = os.path.join(save_dir, f'{self.model_name}_T2I', category, 'samples')
            os.makedirs(out_dir, exist_ok=True)
            logger.info('[%s] %d prompts -> %s', category, len(prompts), out_dir)

            for i, prompt in enumerate(tqdm(prompts, desc=f'{category}')):
                image_path = os.path.join(out_dir, f'{prompt}_{i:06d}.png')
                if resume and os.path.exists(image_path):
                    continue
                image = self._generate(model, prompt)
                image.save(image_path, format='PNG')
