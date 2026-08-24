"""The VISTA objective library, grouped by compositional category.

  binding   L_bind    — attribute/modifier to noun alignment
  entity    L_entity  — per-concept spatial separation and retention
  spatial   L_rel     — in-plane relations (left/right, above/below)
  depth               — Boundary Containment for occlusion ordering
  numeracy  L_num     — counting (implemented; disabled in all reported runs)
  utils               — shared grid / centroid / perimeter helpers

Every objective takes (attn, pairs) where `attn` is a [B, Lq, Lk] cross-attention
map carrying grad_fn and `pairs` is the matching parsed structure, and returns a
scalar tensor. That uniform signature is what lets `vista.registry` swap terms
in and out without touching the optimiser.
"""

from vista.losses import binding, depth, entity, numeracy, spatial, utils

__all__ = ['binding', 'depth', 'entity', 'numeracy', 'spatial', 'utils']
