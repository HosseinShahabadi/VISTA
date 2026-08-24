"""Text-to-image wrappers around the Infinity backbone, with VISTA steering.

`InfinityWrapper` loads the tokenizer, VAE and transformer once and exposes a
single `generate()` entry point. Passing `use_vista=True` parses the prompt,
builds a VISTA config and lets the backbone call `run_scale_optimization` at the
selected scales; passing `use_vista=False` gives the unmodified Infinity
baseline through exactly the same code path.

`InfinityWrapper2B` and `InfinityWrapper8B` differ only in checkpoint layout and
model configuration:

                        2B                   8B
    vae_type            32                   14
    apply_spatial_patchify  0                1
    checkpoint_type     torch                torch_shard

The `apply_spatial_patchify` difference matters for steering: on 8B the token
grid at a given scale index is half the size it is on 2B, so a scale that steers
on 2B can fall below the minimum-grid guard on 8B. 8B also needs a much smaller
`vista_grad_tol` (see `vista.registry`).
"""

import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image

from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from tools.run_infinity import (
    gen_one_img,
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
)
from vista.config import build_vista_cfg

logger = logging.getLogger('vista')

DEFAULT_T5_PATH = os.environ.get('VISTA_T5_PATH', 'weights/flan-t5-xl')


class T2IModelWrapper:
    """Base wrapper for text-to-image models."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, seed: int = 42) -> Image.Image:
        raise NotImplementedError


class InfinityWrapper(T2IModelWrapper):
    """Infinity backbone + optional ScaleKV + optional VISTA steering."""

    # Overridden by the 2B / 8B subclasses.
    MODEL_TYPE = 'infinity_2b'
    VAE_TYPE = 32
    APPLY_SPATIAL_PATCHIFY = 0
    CHECKPOINT_TYPE = 'torch'
    CACHE_DIR = '/dev/shm'

    def __init__(
        self,
        model_path: str,
        vae_path: str,
        text_encoder_ckpt: str = DEFAULT_T5_PATH,
        gpu_id: int = 0,
        pn: str = '1M',
        use_scale_kv: bool = True,
        window_size: int = 8,
        max_capacity: int = 400,
        kernel_size: int = 5,
        pooling: str = 'maxpool',
        cache_dir: str = None,
    ):
        super().__init__('Infinity')

        torch.cuda.set_device(gpu_id)
        self.device = torch.device('cuda')

        self.use_scale_kv = use_scale_kv
        self.window_size = window_size
        self.max_capacity = max_capacity
        self.kernel_size = kernel_size
        self.pooling = pooling

        self.args = argparse.Namespace(
            pn=pn,
            model_path=model_path,
            vae_path=vae_path,
            text_encoder_ckpt=text_encoder_ckpt,
            cfg_insertion_layer=0,
            vae_type=self.VAE_TYPE,
            model_type=self.MODEL_TYPE,
            apply_spatial_patchify=self.APPLY_SPATIAL_PATCHIFY,
            checkpoint_type=self.CHECKPOINT_TYPE,
            cache_dir=cache_dir or self.CACHE_DIR,
            add_lvl_embeding_only_first_block=1,
            use_bit_label=1,
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            use_scale_schedule_embedding=0,
            sampling_per_bits=1,
            text_channels=2048,
            h_div_w_template=1.000,
            use_flex_attn=0,
            bf16=1,
            enable_model_cache=1,
        )

        logger.info('Loading %s ...', self.MODEL_TYPE)
        self.text_tokenizer, self.text_encoder = load_tokenizer(
            t5_path=self.args.text_encoder_ckpt
        )
        self.vae = load_visual_tokenizer(self.args)
        self.infinity = load_transformer(self.vae, self.args)

        if use_scale_kv:
            logger.info('Enabling ScaleKV (window_size=%d, max_capacity=%d, '
                        'kernel_size=%d, pooling=%s)',
                        window_size, max_capacity, kernel_size, pooling)
            self.infinity = self._enable_scale_kv()

        logger.info('%s ready (ScaleKV=%s).', self.MODEL_TYPE, use_scale_kv)

    def _enable_scale_kv(self):
        raise NotImplementedError

    # ── generation ──────────────────────────────────────────────────────────
    def build_scale_schedule(self, h_div_w: float = 1.0):
        """Resolution ladder for the requested aspect ratio."""
        h_div_w_template_ = h_div_w_templates[
            np.argmin(np.abs(h_div_w_templates - h_div_w))
        ]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][self.args.pn]['scales']
        return [(1, h, w) for (_, h, w) in scale_schedule]

    def generate(
        self,
        prompt: str,
        seed: int = 42,
        cfg: float = 4.0,
        tau: float = 1.0,
        h_div_w: float = 1.0,
        enable_positive_prompt: int = 0,
        # ── VISTA ───────────────────────────────────────────────────────────
        use_vista: bool = False,
        vista_n_steps: int = 5,
        vista_lr: float = 0.005,
        vista_steer_frac: float = 0.4,
        vista_grad_tol: float = 1e-3,
        vista_active_losses: list = None,
        spacy_weights_dir: str = None,
        # ── analysis / diagnostics ──────────────────────────────────────────
        extract_attention: bool = False,
        t5_attention: bool = False,
        get_final_resolution: bool = False,
        time_per_scale: bool = False,
        custom_text_embedding=None,
        sgvs_res: int = 6,
    ):
        """Generate one image.

        Returns a PIL image, or — when any diagnostic flag is set — a list whose
        first element is the image followed by the requested extras.
        """
        if time_per_scale and not get_final_resolution:
            raise ValueError('time_per_scale requires get_final_resolution=True; '
                             'per-scale timings are produced by that code path.')

        if extract_attention:
            from vista.models.attention_store import (
                InfinityAttentionStore,
                register_infinity_cross_attention,
                register_infinity_self_attention,
            )
            attn_store = InfinityAttentionStore()
            hooks = (register_infinity_cross_attention(self.infinity, attn_store)
                     + register_infinity_self_attention(self.infinity, attn_store))

        if t5_attention:
            from vista.models.t5_attention_store import (
                T5AttentionStore,
                register_t5_attention_hooks,
            )
            t5_store = T5AttentionStore()
            t5_hooks = register_t5_attention_hooks(self.text_encoder, t5_store)

        scale_schedule = self.build_scale_schedule(h_div_w)
        scale_resolutions = [h for (_, h, _) in scale_schedule]

        # ── VISTA config for this prompt ────────────────────────────────────
        vista_cfg = None
        if use_vista:
            vista_cfg = build_vista_cfg(
                prompt=prompt,
                scale_schedule=scale_schedule,
                t5_path=self.args.text_encoder_ckpt,
                spacy_weights_dir=spacy_weights_dir,
                steer_frac=vista_steer_frac,
                n_steps=vista_n_steps,
                lr=vista_lr,
                grad_tol=vista_grad_tol,
                active_losses=vista_active_losses,
            )
            if vista_cfg is None:
                logger.info('[VISTA] nothing steerable in %r - generating normally.', prompt)

        generated_image = gen_one_img(
            self.infinity,
            self.vae,
            self.text_tokenizer,
            self.text_encoder,
            prompt,
            g_seed=seed,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=cfg,
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[self.args.cfg_insertion_layer],
            vae_type=self.args.vae_type,
            sampling_per_bits=self.args.sampling_per_bits,
            enable_positive_prompt=enable_positive_prompt,
            get_final_resolution=get_final_resolution,
            t5_attention=t5_attention,
            custom_text_embedding=custom_text_embedding,
            vista_cfg=vista_cfg,
        )

        outputs = []

        if extract_attention:
            cross_attn_maps = attn_store.get_average_cross()
            self_attn_maps = attn_store.get_self()
            tokens = self.text_tokenizer(prompt, return_tensors='pt').input_ids[0]
            decoded_tokens = self.text_tokenizer.convert_ids_to_tokens(tokens)
            for h in hooks:
                h.remove()
            outputs.append(cross_attn_maps)
            outputs.append((self_attn_maps, scale_resolutions))
            outputs.append(decoded_tokens)

        if t5_attention:
            t5_attn_maps = t5_store.get()
            tokens = self.text_tokenizer(prompt, return_tensors='pt').input_ids[0]
            decoded_tokens = self.text_tokenizer.convert_ids_to_tokens(tokens)
            for h in t5_hooks:
                h.remove()
            outputs.append(t5_attn_maps)
            outputs.append(decoded_tokens)

        if get_final_resolution:
            per_scale_time = generated_image[2]
            per_scale_images = generated_image[1]
            generated_image = generated_image[0]

            pil_per_scale_images = []
            for img_ps, scales_hw in per_scale_images:
                h, w = scales_hw
                pil_per_scale_images.append((h, w, self._to_pil_intermediate(img_ps)))

            sgvs_input = (
                self._to_pil_intermediate(per_scale_images[sgvs_res][0], output_size=(512, 512)),
                sgvs_res,
            )
            outputs.append((pil_per_scale_images, sgvs_input))

        pil_image = self._to_pil(generated_image)

        if outputs:
            outputs.append(pil_image)
            if time_per_scale:
                return outputs[::-1], per_scale_time
            return outputs[::-1]

        return pil_image

    # ── tensor -> PIL ───────────────────────────────────────────────────────
    @staticmethod
    def _to_pil(generated_image):
        img_np = generated_image.cpu().numpy()

        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))    # CHW -> HWC

        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        if img_np.shape[-1] == 3:
            img_np = img_np[..., ::-1]                  # BGR -> RGB

        return Image.fromarray(img_np)

    @staticmethod
    def _to_pil_intermediate(img_tensor, output_size=None):
        """Convert a per-scale [B, C, H, W] or [C, H, W] tensor to a PIL image."""
        img = img_tensor.detach().float()
        if img.dim() == 4:
            img = img[0]
        img = img.squeeze(0) if img.shape[0] == 1 else img.permute(1, 2, 0)
        img = img.cpu().numpy()
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255)
        pil = Image.fromarray(img.astype(np.uint8))
        if output_size is not None:
            pil = pil.resize(output_size, Image.LANCZOS)
        return pil

    def __del__(self):
        for attr in ('infinity', 'vae', 'text_encoder', 'text_tokenizer'):
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()


class InfinityWrapper2B(InfinityWrapper):
    MODEL_TYPE = 'infinity_2b'
    VAE_TYPE = 32
    APPLY_SPATIAL_PATCHIFY = 0
    CHECKPOINT_TYPE = 'torch'
    CACHE_DIR = '/dev/shm2'

    def _enable_scale_kv(self):
        from scale_kv.enable_scale_kv_2b import enable_scale_kv
        return enable_scale_kv(
            self.infinity,
            window_size=self.window_size,
            max_capacity=self.max_capacity,
            kernel_size=self.kernel_size,
            pooling=self.pooling,
        )


class InfinityWrapper8B(InfinityWrapper):
    MODEL_TYPE = 'infinity_8b'
    VAE_TYPE = 14
    APPLY_SPATIAL_PATCHIFY = 1
    CHECKPOINT_TYPE = 'torch_shard'
    CACHE_DIR = '/dev/shm'

    def _enable_scale_kv(self):
        from scale_kv.enable_scale_kv_8b import enable_scale_kv
        return enable_scale_kv(
            self.infinity,
            window_size=self.window_size,
            max_capacity=self.max_capacity,
            kernel_size=self.kernel_size,
            pooling=self.pooling,
        )
