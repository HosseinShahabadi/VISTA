"""Free-form prompt generation (no benchmark harness).

Useful for qualitative figures and for sweeping a single hyperparameter over a
small prompt set. Output layout:

    <save_dir>/<model_name>_prompts/steer<frac>_lr<lr>_seed<seed>/<prompt>.png
"""

import logging
import os
import re
from typing import List, Optional

from tqdm import tqdm

from vista.benchmarks.base import BaseGenerator

logger = logging.getLogger('vista')


def _slugify(prompt: str, max_len: int = 100) -> str:
    """Filesystem-safe file stem for a prompt."""
    slug = re.sub(r'[^\w\s-]', '', prompt).strip()
    slug = re.sub(r'\s+', '_', slug)
    return slug[:max_len] or 'prompt'


class PromptGenerator(BaseGenerator):
    def generate_images(
        self,
        model,
        save_dir: str,
        prompts: List[str],
        run_name: Optional[str] = None,
    ) -> None:
        s = self.settings
        run_name = run_name or f'steer{s.steer_frac}_lr{s.lr}_seed{s.seed}'
        out_dir = os.path.join(save_dir, f'{self.model_name}_prompts', run_name)
        os.makedirs(out_dir, exist_ok=True)
        logger.info('%d prompts -> %s', len(prompts), out_dir)

        for prompt in tqdm(prompts, desc='Generating'):
            image = self._generate(model, prompt)
            image.save(os.path.join(out_dir, f'{_slugify(prompt)}.png'))
