"""MetricResult envelope helpers and legacy compatibility loaders."""
from __future__ import annotations

import glob
import json
import os
from typing import Iterable

from .metrics import get_metric


LEGACY_RESULT_PREFIXES = {
    "bf1_latent_ablation": "bf1",
    "bf1_latent_ablation_legacy": "bf1_legacy",
    "bf1_layer_ablation_legacy": "bf1_layer_legacy",
    "cf2_pf_decay_curve": "cf2",
    "cf2_pf_decay_curve_legacy": "cf2_legacy",
}
PREFIX_TO_METRIC_ID = {prefix: metric_id for metric_id, prefix in LEGACY_RESULT_PREFIXES.items()}


def make_metric_result(metric_id: str, model_tag: str, payload: dict) -> dict:
    spec = get_metric(metric_id)
    return {
        "metric_id": spec.metric_id,
        "metric_name": spec.title,
        "legacy_name": spec.legacy_name,
        "model": model_tag,
        "result_type": spec.kind,
        "payload": payload,
    }


def legacy_result_path(out_dir: str, metric_id: str, model_tag: str) -> str | None:
    prefix = LEGACY_RESULT_PREFIXES.get(metric_id)
    if prefix is None:
        return None
    return os.path.join(out_dir, f"{prefix}_{model_tag}.json")


def write_metric_result(out_dir: str, metric_id: str, model_tag: str, payload: dict) -> dict:
    envelope = make_metric_result(metric_id, model_tag, payload)

    legacy_path = legacy_result_path(out_dir, metric_id, model_tag)
    if legacy_path is not None:
        with open(legacy_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, f"{metric_id}_{model_tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, ensure_ascii=False)
    return envelope


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _candidate_result_roots(root: str) -> list[str]:
    roots = [root]
    for path in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(path):
            roots.append(path)
    return roots


def load_metric_results(root: str, include_legacy: bool = True) -> list[dict]:
    """Load unified metrics/*.json and optionally legacy bf1_*/cf2_* results."""
    envelopes: list[dict] = []
    seen = set()

    for result_root in _candidate_result_roots(root):
        for path in sorted(glob.glob(os.path.join(result_root, "metrics", "*.json"))):
            envelope = _read_json(path)
            metric_id = envelope.get("metric_id")
            payload = envelope.get("payload")
            model = envelope.get("model")
            if not metric_id or not isinstance(payload, dict) or not model:
                continue
            key = (str(metric_id), str(model))
            if key in seen:
                continue
            envelopes.append(envelope)
            seen.add(key)

    if not include_legacy:
        return envelopes

    for result_root in _candidate_result_roots(root):
        for metric_id, prefix in LEGACY_RESULT_PREFIXES.items():
            for path in sorted(glob.glob(os.path.join(result_root, f"{prefix}_*.json"))):
                payload = _read_json(path)
                model = str(payload.get("model") or os.path.basename(path)[len(prefix) + 1:-5])
                key = (metric_id, model)
                if key in seen:
                    continue
                envelopes.append(make_metric_result(metric_id, model, payload))
                seen.add(key)

    return envelopes


def split_metric_results(metric_results: Iterable[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    ablation: dict[str, dict] = {}
    decay: dict[str, dict] = {}

    for envelope in metric_results:
        metric_id = envelope.get("metric_id")
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            continue
        model = str(payload.get("model") or envelope.get("model") or "unknown")
        if metric_id in {"bf1_latent_ablation", "bf1_latent_ablation_legacy"}:
            ablation[model] = payload
        elif metric_id in {"cf2_pf_decay_curve", "cf2_pf_decay_curve_legacy"}:
            decay[model] = payload

    return ablation, decay
