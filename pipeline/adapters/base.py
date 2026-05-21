"""Model adapter interface for VLM backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn


@dataclass
class ModelBundle:
    """Loaded model objects plus adapter-specific metadata."""

    model: nn.Module
    processor: Any
    image_pad_id: int


class VLMAdapter(Protocol):
    """Adapter contract used by metrics and the runner."""

    arch: str

    def load(self, cfg_model: dict, *, dtype: torch.dtype, device: str) -> ModelBundle:
        """Load model/processor and move the model to the target device."""

    def build_inputs(self, wrapper, image, question: str):
        """Build a single-sample model input object."""

    def generate(self, wrapper, images: list, prompts: list[str],
                 max_new_tokens: int = 64) -> list[str]:
        """Generate text outputs for optional output-level checks."""

