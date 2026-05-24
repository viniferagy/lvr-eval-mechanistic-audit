#!/usr/bin/env python
"""Check local Monet model/source/data paths without loading model weights."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def resolve_path(value: Any, *, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def require(path: Path | None, label: str) -> Path:
    if path is None:
        fail(f"{label} is not configured")
    if not path.exists():
        fail(f"{label} missing: {path}")
    print(f"PASS {label}: {path}")
    return path


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        fail(f"config must be a YAML mapping: {path}")
    return data


def check_shards(model_dir: Path) -> None:
    index_candidates = [
        model_dir / "model.safetensors.index.json",
        model_dir / "pytorch_model.bin.index.json",
    ]
    index_path = next((path for path in index_candidates if path.is_file()), None)
    if index_path is None:
        direct = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
        if not direct:
            fail(f"no model index or direct weight shard found in {model_dir}")
        print(f"PASS direct weight files: {len(direct)} file(s)")
        return
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map") or {}
    missing = sorted({shard for shard in weight_map.values() if not (model_dir / shard).is_file()})
    if missing:
        fail(f"{index_path} references missing shard(s): {missing[:8]}")
    print(f"PASS weight index: {index_path} ({len(set(weight_map.values()))} shard(s))")


def find_monet_source(model_cfg: dict, *, repo_root: Path) -> tuple[Path | None, list[Path]]:
    candidates = [
        model_cfg.get("monet_source_path"),
        os.environ.get("MONET_SOURCE_PATH"),
        repo_root.parent / "Monet",
        repo_root.parent / "monet",
        Path("/home/pengguangyue/workspace/proj/Monet"),
        Path("/data/pengguangyue/proj/Monet"),
    ]
    tried: list[Path] = []
    for candidate in candidates:
        path = resolve_path(candidate, base=repo_root)
        if path is None:
            continue
        tried.append(path)
        if (path / "monet_qwen_model" / "modeling_qwen2_5_vl_monet.py").is_file():
            return path, tried
    return None, tried


def git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:  # noqa: BLE001
        return None
    return result.stdout.strip() or None


def git_dirty(path: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--short"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:  # noqa: BLE001
        return None
    return bool(result.stdout.strip())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Check local Monet-7B paths without loading model weights.")
    ap.add_argument("--config", default="config.monet.preflight.yaml")
    ap.add_argument("--model-key", default="monet_7b")
    ap.add_argument("--check-data", action="store_true", help="Also require the configured data manifest/image root.")
    args = ap.parse_args(argv)

    repo_root = Path.cwd()
    cfg_path = require(resolve_path(args.config, base=repo_root), "config")
    cfg = load_yaml(cfg_path)
    model_cfg = ((cfg.get("models") or {}).get(args.model_key) or {})
    if not model_cfg:
        fail(f"models.{args.model_key} missing in {cfg_path}")
    if model_cfg.get("arch") not in {"monet_qwen2_5_vl", "monet", "monet_qwen"}:
        fail(f"models.{args.model_key}.arch must be monet_qwen2_5_vl/monet, got {model_cfg.get('arch')!r}")

    model_dir = require(resolve_path(model_cfg.get("path"), base=repo_root), f"models.{args.model_key}.path")
    require(model_dir / "config.json", "Monet config.json")
    check_shards(model_dir)

    source_path, tried = find_monet_source(model_cfg, repo_root=repo_root)
    if source_path is None:
        fail(
            "Monet source checkout missing monet_qwen_model/modeling_qwen2_5_vl_monet.py. Tried: "
            + ", ".join(str(path) for path in tried)
        )
    require(source_path / "monet_qwen_model" / "modeling_qwen2_5_vl_monet.py", "Monet modeling_qwen2_5_vl_monet.py")
    require(source_path / "inference" / "vllm" / "monet_gpu_model_runner.py", "Monet vLLM latent runner")

    revision = git_revision(source_path)
    if revision:
        print(f"PASS Monet source git commit: {revision}")
    dirty = git_dirty(source_path)
    if dirty is not None:
        print(f"PASS Monet source dirty state recorded: {dirty}")

    if args.check_data:
        data_cfg = cfg.get("data") or {}
        manifest = require(resolve_path(data_cfg.get("jsonl_path"), base=repo_root), "data.jsonl_path")
        image_root = require(resolve_path(data_cfg.get("image_root"), base=repo_root), "data.image_root")
        if manifest.stat().st_size <= 0:
            fail(f"data manifest is empty: {manifest}")
        if not image_root.is_dir():
            fail(f"data.image_root is not a directory: {image_root}")

    print("MONET ENV CHECK PASSED")


if __name__ == "__main__":
    main()
