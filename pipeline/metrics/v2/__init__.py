"""V2 causal metric entry points."""
from __future__ import annotations

from .bf_conf_calibrated_progression import SPEC as BF_CONF_SPEC
from .bf_patch_answer_transfer import SPEC as BF_PATCH_SPEC
from .bf_swap_latent_replacement import SPEC as BF_SWAP_SPEC
from .cf_stage_decay import SPEC as CF_STAGE_SPEC
from .pf_a_corruption_selectivity import SPEC as PF_A_SPEC
from .pf_b_patch_alignment import SPEC as PF_B_SPEC

V2_SPECS = [
    PF_A_SPEC,
    BF_PATCH_SPEC,
    BF_SWAP_SPEC,
    BF_CONF_SPEC,
    CF_STAGE_SPEC,
    PF_B_SPEC,
]

__all__ = [
    "BF_CONF_SPEC",
    "BF_PATCH_SPEC",
    "BF_SWAP_SPEC",
    "CF_STAGE_SPEC",
    "PF_A_SPEC",
    "PF_B_SPEC",
    "V2_SPECS",
]
