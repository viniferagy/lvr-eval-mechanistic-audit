"""V2 causal metric skeletons."""
from __future__ import annotations

from .bf_patch_answer_transfer import SPEC as BF_PATCH_SPEC
from .pf_a_corruption_selectivity import SPEC as PF_A_SPEC

V2_SPECS = [PF_A_SPEC, BF_PATCH_SPEC]

__all__ = ["BF_PATCH_SPEC", "PF_A_SPEC", "V2_SPECS"]
