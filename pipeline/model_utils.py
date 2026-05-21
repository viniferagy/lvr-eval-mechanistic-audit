"""
pipeline/model_utils.py
========================
模型 wrapper + decoder layer 自动探测 + final_norm/lm_head 缓存。

为支持 BF-3(logit lens)与 PF-3(attention KL),加载时:
  - 通过 adapter 加载模型并设置 attn_implementation="eager"
  - 缓存 final_norm / lm_head      (BF-3 logit lens 用)
  - 缓存 image_pad_id              (adapter image span 定位用)

上层(ablation / degradation / internal_metrics)只跟 VLMWrapper 打交道。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from .adapters import get_adapter

logger = logging.getLogger("lvr_eval.model")

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# --------------------------------------------------------------------------- #
#  结构探测
# --------------------------------------------------------------------------- #
def find_decoder_layers(model: nn.Module, layer_path: Optional[str] = None) -> nn.ModuleList:
    if layer_path:
        obj: Any = model
        for attr in layer_path.split("."):
            obj = getattr(obj, attr)
        assert isinstance(obj, nn.ModuleList), f"{layer_path} 不是 ModuleList"
        logger.info("decoder layers via explicit path '%s' (n=%d)", layer_path, len(obj))
        return obj

    common = [
        "model.language_model.layers", "model.model.layers",
        "language_model.model.layers", "model.layers", "language_model.layers",
    ]
    for path in common:
        obj, ok = model, True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, nn.ModuleList) and len(obj) > 0:
            logger.info("decoder layers via common path '%s' (n=%d)", path, len(obj))
            return obj

    best, best_path = None, ""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 4:
            has_attn = any("attn" in n.lower() for n, _ in mod[0].named_modules())
            if has_attn and (best is None or len(mod) > len(best)):
                best, best_path = mod, name
    if best is None:
        raise RuntimeError("无法自动定位 decoder layers，请在 config 填 layer_path")
    logger.info("decoder layers via scan '%s' (n=%d)", best_path, len(best))
    return best


def find_final_norm(model: nn.Module):
    """BF-3 logit lens 的 final norm。覆盖 transformers 4.49 的 model.model.norm。"""
    for path in ["model.norm", "model.language_model.norm",
                 "language_model.model.norm", "model.model.norm"]:
        obj, ok = model, True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, nn.Module):
            logger.info("final_norm via '%s'", path)
            return obj
    logger.warning("找不到 final norm，logit lens 将用 raw hidden state")
    return None


def find_lm_head(model: nn.Module):
    for path in ["lm_head", "language_model.lm_head", "model.lm_head"]:
        obj, ok = model, True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, nn.Module):
            logger.info("lm_head via '%s'", path)
            return obj
    raise RuntimeError("找不到 lm_head")


# --------------------------------------------------------------------------- #
#  封装
# --------------------------------------------------------------------------- #
@dataclass
class VLMWrapper:
    model: nn.Module
    processor: Any
    adapter: Any
    layers: nn.ModuleList
    device: str
    arch: str
    image_pad_id: int
    final_norm: Any
    lm_head: Any
    cfg: Optional[dict] = None

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    # ---- 构造单样本 inputs（BF-3 / PF-3 共用） ----
    def build_inputs(self, image, question: str):
        return self.adapter.build_inputs(self, image, question)

    def build_inputs_from_sample(self, sample):
        if hasattr(self.adapter, "build_inputs_from_sample"):
            return self.adapter.build_inputs_from_sample(self, sample)
        return self.adapter.build_inputs(self, sample.image, sample.question)

    # ---- 生成（可选, 用于 output-accuracy sanity） ----
    @torch.no_grad()
    def generate(self, images: list, prompts: list[str], max_new_tokens: int = 64) -> list[str]:
        return self.adapter.generate(self, images, prompts, max_new_tokens=max_new_tokens)


def load_model(cfg_model: dict, dtype: str = "bfloat16",
               device: str = "cuda:0", cfg: Optional[dict] = None) -> VLMWrapper:
    arch = cfg_model.get("arch", "auto")
    torch_dtype = _DTYPE[dtype]
    adapter = get_adapter(arch)
    logger.info("loading %s (%s) from %s", cfg_model["name"], arch, cfg_model["path"])
    bundle = adapter.load(cfg_model, dtype=torch_dtype, device=device)

    return VLMWrapper(
        model=bundle.model, processor=bundle.processor, adapter=adapter,
        layers=find_decoder_layers(bundle.model, cfg_model.get("layer_path")),
        device=device, arch=arch,
        image_pad_id=bundle.image_pad_id,
        final_norm=find_final_norm(bundle.model),
        lm_head=find_lm_head(bundle.model),
        cfg=cfg,
    )
