"""Monet/Qwen2.5-VL adapter preflight.

Monet's paper-grade latent inference path is implemented through its modified
vLLM runner. This adapter loads the published Monet Transformers model for
architecture probes and standard-forward regression checks; it deliberately
does not claim to expose Monet's runtime latent feedback loop.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

from .base import ModelBundle
from .qwen_vl import QwenVLAdapter

logger = logging.getLogger("lvr_eval.adapters.monet_qwen")


def _resolve_path(value: Any, *, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _find_monet_source(cfg_model: dict, *, repo_root: Path) -> Path:
    candidates = [
        cfg_model.get("monet_source_path"),
        os.environ.get("MONET_SOURCE_PATH"),
        repo_root.parent / "Monet",
        repo_root.parent / "monet",
        Path("/home/pengguangyue/workspace/proj/Monet"),
        Path("/data/pengguangyue/proj/Monet"),
    ]
    tried: list[Path] = []
    for candidate in candidates:
        path = _resolve_path(candidate, base=repo_root)
        if path is None:
            continue
        tried.append(path)
        if (path / "monet_qwen_model" / "modeling_qwen2_5_vl_monet.py").is_file():
            return path
    raise RuntimeError(
        "Monet source checkout missing monet_qwen_model/modeling_qwen2_5_vl_monet.py. "
        "Set models.monet_7b.monet_source_path or MONET_SOURCE_PATH. Tried: "
        + ", ".join(str(path) for path in tried)
    )


def _load_monet_model_class(source_path: Path):
    source_text = str(source_path)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("monet_qwen_model.modeling_qwen2_5_vl_monet")
    return getattr(module, "Qwen2_5_VLForConditionalGeneration")


class MonetQwenAdapter(QwenVLAdapter):
    """Adapter for the public Monet-7B Transformers checkpoint."""

    def __init__(self, arch: str = "monet_qwen2_5_vl"):
        super().__init__("qwen2_5_vl")
        self.arch = arch

    def load(self, cfg_model: dict, *, dtype: torch.dtype, device: str) -> ModelBundle:
        try:
            from transformers import AutoProcessor
        except ValueError as exc:
            if "grouped_mm_fallback" in str(exc) or "infer_schema" in str(exc):
                raise RuntimeError(
                    "transformers 与 torch 版本不兼容。请运行: "
                    "pip install -U 'transformers>=4.51,<4.56'"
                ) from exc
            raise

        repo_root = Path.cwd()
        source_path = _find_monet_source(cfg_model, repo_root=repo_root)
        model_cls = _load_monet_model_class(source_path)

        path = cfg_model["path"]
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = model_cls.from_pretrained(
            path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=cfg_model.get("attn_implementation", "eager"),
        )
        model.to(device).eval()
        image_pad_id = self._image_pad_id(processor, cfg_model)
        logger.info("loaded Monet model via source checkout: %s", source_path)
        return ModelBundle(model=model, processor=processor, image_pad_id=image_pad_id)

    def get_spans(self, wrapper, inputs, model_outputs=None):
        spans = super().get_spans(wrapper, inputs, model_outputs=model_outputs)
        spans.notes.update({
            "adapter": "monet_qwen",
            "latent_tokens": "monet_runtime_abs_vis_tokens_not_exposed_in_standard_forward",
            "true_latent_inference": "requires_modified_vllm_runner",
        })
        return spans

    def generate_with_trace(self, wrapper, image, question: str, **kwargs) -> dict:
        raise NotImplementedError(
            "Monet true latent generation is implemented by the official modified vLLM "
            "runner, not by the standard Transformers generate path. Use the W12 Monet "
            "preflight first, then add a vLLM trace adapter before causal latent metrics."
        )
