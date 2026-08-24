"""Registry of the objectives VISTA can steer with.

`LOSS_REGISTRY` maps an objective name to the callable that computes it and to
the key of the parsed structure it consumes from the VISTA config. Swapping an
objective is a one-line `fn=` change; the optimiser code never names a loss
directly.

The unified compositional objective reported in the paper is

    L_compos = L_entity + L_bind + L_rel

realised here as the default active set ('binding', 'spatial', 'depth'). The
entity-level separation term is carried inside the depth objective's separation
component and by the contrastive negatives of the binding term. L_num is
implemented (see `vista.losses.numeracy`) but is DISABLED in every experiment
reported in the paper, hence its absence from `DEFAULT_ACTIVE_LOSSES`.
"""

from vista.losses.binding import conform_contrastive_loss
from vista.losses.depth import depth_occlusion_loss
from vista.losses.numeracy import numerical_cluster_count_loss
from vista.losses.spatial import pos_spatial_loss

# ── Active configuration (this is the one used for every reported number) ────
LOSS_REGISTRY = {
    'binding':  dict(fn=conform_contrastive_loss,     pairs_key='pairs'),
    'spatial':  dict(fn=pos_spatial_loss,             pairs_key='spatial_pairs'),
    'depth':    dict(fn=depth_occlusion_loss,         pairs_key='spatial_pairs'),
    'numeracy': dict(fn=numerical_cluster_count_loss, pairs_key='numerical_pairs'),
}

# Objectives that actually run unless `active_losses` is set on the config.
# 'numeracy' is deliberately excluded — see the module docstring.
DEFAULT_ACTIVE_LOSSES = ['binding', 'spatial', 'depth']

# Optional per-loss gradient tolerance override. Anything not listed here falls
# back to the global `grad_tol` from the VISTA config. Only calibrate entries
# you have actually inspected in logs; leave the rest on the shared default.
#
#   NOTE (Infinity-8B): the global default of 1e-3 is far too coarse for the 8B
#   model, whose per-token gradient norms are orders of magnitude smaller. It
#   silently zeroes out optimisation at the finer steered scales. Pass a much
#   smaller `grad_tol` (e.g. 1e-12) when steering 8B.
LOSS_GRAD_TOLS = {
    # 'binding': 0.03,
    # 'spatial': 0.005,
    # 'numeracy': 8e-3,
}


def available_losses():
    """Every implemented objective, grouped by category.

    Handy for ablations: pick a name from here and drop it into
    `LOSS_REGISTRY[<slot>]['fn']`, or build your own registry and pass it in.
    """
    from vista.losses import binding, depth, entity, numeracy, spatial

    return {
        'binding': [
            binding.conform_contrastive_loss,      # default
            binding.syngen_loss,
            binding.kl_binding_loss,
            binding.jsd_binding_loss,
            binding.composite_binding_loss,
            binding.peak_alignment_loss,
            binding.exclusive_binding_loss,
            binding.contrastive_binding_loss,
            binding.divide_and_bind_loss,
            binding.overlap_coefficient_loss,
            binding.focal_binding_loss,
        ],
        'entity': [
            entity.astar_segregation_loss,
            entity.astar_retention_loss,
            entity.attention_presence_balance_loss,
        ],
        'spatial': [
            spatial.pos_spatial_loss,              # default
            spatial.spatial_relative_centroid_loss,
            spatial.spatial_half_field_loss,
            spatial.spatial_separation_loss,
            spatial.spatial_gaussian_target_loss,
            spatial.spatial_iou_loss,
            spatial.composite_spatial_loss,
            spatial.cross_scale_consistency_loss,
        ],
        'depth': [
            depth.depth_occlusion_loss,            # default (Boundary Containment)
            depth.depth_proxy_loss,
            depth.combined_depth_loss,
        ],
        'numeracy': [
            numeracy.numerical_cluster_count_loss,  # registry default, but inactive
            numeracy.numerical_entropy_count_loss,
            numeracy.numerical_softmask_component_loss,
            numeracy.numerical_template_loss,
            numeracy.composite_numerical_loss,
        ],
    }
