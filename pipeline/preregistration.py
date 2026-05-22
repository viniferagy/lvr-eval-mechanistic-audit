"""Preregistration manifest loading and run lock helpers."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MANIFEST_PATH = Path("prereg") / "manifest.yaml"


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(path or DEFAULT_MANIFEST_PATH)
    with manifest_path.open(encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"prereg manifest must be a mapping: {manifest_path}")
    return manifest


def canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    text = yaml.safe_dump(
        manifest,
        allow_unicode=True,
        sort_keys=True,
        default_flow_style=False,
    )
    return text.encode("utf-8")


def hash_manifest(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def _primary_metric_summary(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    primary = manifest.get("primary_metrics") or []
    out = []
    for item in primary:
        if not isinstance(item, dict):
            continue
        out.append({
            "metric_id": item.get("metric_id"),
            "family": item.get("family"),
            "primary_scalar": item.get("primary_scalar"),
            "version": item.get("version"),
            "status": item.get("status"),
        })
    return out


def build_lock_payload(
    manifest: dict[str, Any],
    *,
    manifest_path: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    return {
        "manifest_version": manifest.get("manifest_version"),
        "sha256": hash_manifest(manifest),
        "manifest_path": str(manifest_path or DEFAULT_MANIFEST_PATH),
        "primary_metrics": _primary_metric_summary(manifest),
        "analysis_blind_seed": manifest.get("analysis_blind_seed"),
        "stopping_rule": manifest.get("stopping_rule"),
        "statistics": manifest.get("statistics"),
        "created_at": created_at or dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def write_lock(
    out_dir: str | Path,
    manifest: dict[str, Any],
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = build_lock_payload(manifest, manifest_path=manifest_path)
    path = Path(out_dir) / "prereg.lock.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload
