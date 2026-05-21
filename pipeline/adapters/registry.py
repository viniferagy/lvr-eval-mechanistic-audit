"""Adapter registry."""
from __future__ import annotations

from .qwen_vl import QwenVLAdapter


def get_adapter(arch: str):
    normalized = (arch or "auto").strip().lower()
    if normalized in {"qwen2_5_vl", "qwen3_vl", "auto"}:
        return QwenVLAdapter(normalized)
    raise KeyError(f"unknown model adapter arch: {arch}")


def known_adapters() -> list[str]:
    return ["qwen2_5_vl", "qwen3_vl", "auto"]

