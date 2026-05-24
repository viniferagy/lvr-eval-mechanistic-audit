#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def resolve_path(value, *, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def find_lvr_source(cfg_model: dict, *, repo_root: Path) -> tuple[Path | None, list[Path]]:
    candidates = [
        cfg_model.get("lvr_source_path"),
        os.environ.get("LVR_SOURCE_PATH"),
        repo_root.parent / "lvr",
        Path("/home/pengguangyue/workspace/proj/lvr"),
        Path("/data/pengguangyue/proj/lvr"),
    ]
    tried = []
    for candidate in candidates:
        path = resolve_path(candidate, base=repo_root)
        if path is None:
            continue
        tried.append(path)
        if (path / "src" / "model" / "qwen_lvr_model.py").is_file():
            return path, tried
    return None, tried


def require(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"{label} missing: {path}")
    print(f"PASS {label}: {path}")


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
    missing = []
    for shard in sorted(set(weight_map.values())):
        if not (model_dir / shard).is_file():
            missing.append(shard)
    if missing:
        fail(f"{index_path} references missing shard(s): {missing[:8]}")
    print(f"PASS weight index: {index_path} ({len(set(weight_map.values()))} shard(s))")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Check local LVR model/source paths without loading model weights.")
    ap.add_argument("--config", default="config.lvr_latent_patch.stepsweep.yaml")
    args = ap.parse_args(argv)

    repo_root = Path.cwd()
    cfg_path = resolve_path(args.config, base=repo_root)
    require(cfg_path, "config")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_cfg = ((cfg.get("models") or {}).get("lvr_7b") or {})
    if not model_cfg:
        fail(f"models.lvr_7b missing in {cfg_path}")
    model_dir = resolve_path(model_cfg.get("path"), base=repo_root)
    require(model_dir, "models.lvr_7b.path")
    require(model_dir / "config.json", "LVR config.json")
    check_shards(model_dir)
    source_path, tried = find_lvr_source(model_cfg, repo_root=repo_root)
    if source_path is None:
        fail(
            "LVR source checkout missing src/model/qwen_lvr_model.py. Tried: "
            + ", ".join(str(path) for path in tried)
        )
    require(source_path / "src" / "model" / "qwen_lvr_model.py", "LVR qwen_lvr_model.py")
    print("LVR ENV CHECK PASSED")


if __name__ == "__main__":
    main()
