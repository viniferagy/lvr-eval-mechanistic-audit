#!/usr/bin/env python
"""Validate that a run directory has Main-paper readiness artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def _read_json(path: Path):
    if not path.is_file():
        fail(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_has_evidence_policy(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    required = [
        "qwen_query_span_control",
        "lvr_generation_trace_latent",
        "monet_transformers_latent_range_gate",
    ]
    return all(item in text for item in required)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--manifest", default="prereg/manifest.yaml")
    ap.add_argument("--require-margin-quality", action="store_true")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    manifest = Path(args.manifest)
    if not manifest.is_file():
        fail(f"missing manifest: {manifest}")
    if not _manifest_has_evidence_policy(manifest):
        fail("manifest missing explicit evidence_level policy keys")

    ci = _read_json(run_dir / "summary_with_ci.json")
    if not isinstance(ci, list) or not ci:
        fail("summary_with_ci.json is empty")

    mixed = _read_json(run_dir / "mixed_effects_summary.json")
    if not str(mixed.get("status", "")).startswith("implemented"):
        fail(f"mixed_effects_summary status is not implemented: {mixed.get('status')}")

    sanity = _read_json(run_dir / "sanity" / "summary_sanity.json")
    if sanity.get("overall_status") not in {"pass", "warn"}:
        fail(f"sanity overall_status={sanity.get('overall_status')}")

    if args.require_margin_quality:
        from tools.validate_trace_margin_quality import main as validate_margin

        validate_margin([str(run_dir)])

    print("MAIN PAPER READINESS VALIDATION PASSED")


if __name__ == "__main__":
    main()
