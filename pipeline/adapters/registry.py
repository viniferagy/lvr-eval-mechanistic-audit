"""Adapter registry."""
from __future__ import annotations

from .lvr_qwen import LVRQwenAdapter
from .lvr_qwen_traced import TracedLVRQwenAdapter
from .monet_qwen import MonetQwenAdapter
from .qwen_vl import QwenVLAdapter


def get_adapter(arch: str):
    normalized = (arch or "auto").strip().lower()
    if normalized in {"qwen2_5_vl", "qwen3_vl", "auto"}:
        return QwenVLAdapter(normalized)
    if normalized in {"lvr_qwen2_5_vl", "lvr", "qwen_lvr"}:
        return LVRQwenAdapter(normalized)
    if normalized in {"lvr_qwen2_5_vl_traced", "lvr_traced", "qwen_lvr_traced"}:
        return TracedLVRQwenAdapter(normalized)
    if normalized in {"monet_qwen2_5_vl", "monet", "monet_qwen"}:
        return MonetQwenAdapter(normalized)
    raise KeyError(f"unknown model adapter arch: {arch}")


def known_adapters() -> list[str]:
    return [
        "qwen2_5_vl",
        "qwen3_vl",
        "auto",
        "lvr_qwen2_5_vl",
        "lvr_qwen2_5_vl_traced",
        "lvr",
        "lvr_traced",
        "monet_qwen2_5_vl",
        "monet",
    ]
