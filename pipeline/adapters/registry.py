"""Adapter registry."""
from __future__ import annotations

from .lvr_qwen import LVRQwenAdapter
from .qwen_vl import QwenVLAdapter


def get_adapter(arch: str):
    normalized = (arch or "auto").strip().lower()
    if normalized in {"qwen2_5_vl", "qwen3_vl", "auto"}:
        return QwenVLAdapter(normalized)
    if normalized in {"lvr_qwen2_5_vl", "lvr", "qwen_lvr"}:
        return LVRQwenAdapter(normalized)
    raise KeyError(f"unknown model adapter arch: {arch}")


def known_adapters() -> list[str]:
    return ["qwen2_5_vl", "qwen3_vl", "auto", "lvr_qwen2_5_vl", "lvr"]
