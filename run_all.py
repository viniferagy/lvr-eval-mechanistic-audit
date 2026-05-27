#!/usr/bin/env python
"""
run_all.py
==========
端到端编排:
  load config -> load probe set -> 对每个选定模型:
      按 metrics registry 执行选定指标
  -> dump 原始结果 json -> 跑 analysis(出图 + summary + radar)

用法:
  python run_all.py --config config.yaml --models qwen2_5_vl_7b lvr_7b
  python run_all.py --config config.yaml --models qwen2_5_vl_7b lvr_7b --only bf1
  python run_all.py --config config.yaml --models lvr_7b --only cf2_pf_decay_curve

多卡: 见 launch_sharded.sh（把不同模型/不同层段分配到不同 GPU 并行）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import threading
import time

import yaml

from pipeline.data import load_probe_set
from pipeline.analysis import run_analysis
from pipeline.metrics import (
    get_metric,
    list_runnable_metrics,
    normalize_metric_id,
    run_metric,
)
from pipeline.model_utils import load_model
from pipeline.preregistration import DEFAULT_MANIFEST_PATH, load_manifest, write_lock
from pipeline.results import write_metric_result
from pipeline.sanity import (
    has_failed_checks,
    run_sanity_for_metric_results,
    save_sanity_reports,
)


DEFAULT_MODEL_TAGS = ["qwen2_5_vl_7b", "lvr_7b"]
DEFAULT_PROGRESS_INTERVAL_SECONDS = 60.0
PROGRESS_INTERVAL_ENV = "LVR_PROGRESS_INTERVAL_SECONDS"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def progress_interval_seconds(cfg: dict) -> float | None:
    runtime = cfg.get("runtime", {}) or {}
    raw = os.environ.get(PROGRESS_INTERVAL_ENV, runtime.get("progress_interval_seconds", DEFAULT_PROGRESS_INTERVAL_SECONDS))
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise SystemExit(f"{PROGRESS_INTERVAL_ENV}/runtime.progress_interval_seconds must be numeric") from None
    if value <= 0:
        return None
    return value


class ProgressHeartbeat:
    """Log start/running/done lines around long operations."""

    def __init__(self, log: logging.Logger, label: str, interval_seconds: float | None):
        self.log = log
        self.label = label
        self.interval_seconds = interval_seconds
        self.started = 0.0
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.started = time.monotonic()
        self.log.info("[progress] START %s", self.label)
        if self.interval_seconds is not None:
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._loop, name="lvr-progress-heartbeat", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        elapsed = format_elapsed(time.monotonic() - self.started)
        status = "FAILED" if exc_type is not None else "DONE"
        self.log.info("[progress] %s %s elapsed=%s", status, self.label, elapsed)
        return False

    def _loop(self):
        assert self._stop is not None
        assert self.interval_seconds is not None
        while not self._stop.wait(self.interval_seconds):
            elapsed = format_elapsed(time.monotonic() - self.started)
            self.log.info("[progress] RUNNING %s elapsed=%s", self.label, elapsed)


def metric_payload_summary(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""
    reduction = payload.get("reduction")
    parts: list[str] = []
    for key in ("n", "n_paired", "n_success", "n_error", "total_success", "total_error"):
        value = payload.get(key)
        if value is None and isinstance(reduction, dict):
            value = reduction.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    if isinstance(reduction, dict):
        scalars = []
        for key, value in reduction.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                scalars.append(f"{key}={value:.6g}")
            if len(scalars) >= 4:
                break
        if scalars:
            parts.append("reduction{" + ", ".join(scalars) + "}")
    return "" if not parts else " " + " ".join(parts)


def model_aliases(cfg: dict) -> dict[str, str]:
    aliases = dict(cfg.get("model_aliases") or {})
    aliases.setdefault("M0", "qwen2_5_vl_7b")
    aliases.setdefault("M0_small", "qwen2_5_vl_3b")
    aliases.setdefault("M2", "lvr_7b")
    return aliases


def resolve_model_tags(tags: list[str], cfg: dict, log: logging.Logger) -> list[str]:
    aliases = model_aliases(cfg)
    resolved = []
    for tag in tags:
        new_tag = aliases.get(tag, tag)
        if new_tag != tag:
            log.info("模型别名 %s -> %s", tag, new_tag)
        resolved.append(new_tag)
    return resolved


def validate_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise SystemExit("config must be a YAML mapping")
    required_sections = ("models", "data", "inference", "output")
    missing = [name for name in required_sections if not isinstance(cfg.get(name), dict)]
    if missing:
        raise SystemExit(f"config missing required mapping section(s): {', '.join(missing)}")
    for key in ("dtype", "device"):
        if key not in cfg["inference"]:
            raise SystemExit(f"config.inference.{key} is required")
    for key in ("root",):
        if key not in cfg["output"]:
            raise SystemExit(f"config.output.{key} is required")


def metric_enabled(cfg: dict, metric_id: str) -> bool:
    """Prefer unified metrics.*.enabled, fallback to legacy section.enabled."""
    metrics_cfg = cfg.get("metrics") or {}
    if metric_id in metrics_cfg:
        item = metrics_cfg.get(metric_id) or {}
        if not isinstance(item, dict):
            raise SystemExit(f"config.metrics.{metric_id} must be a mapping or null")
        return bool(item.get("enabled", True))
    legacy_key = get_metric(metric_id).legacy_name
    return bool(cfg.get(legacy_key, {}).get("enabled", False))


def selected_metric_ids(tokens: list[str], cfg: dict) -> list[str]:
    runnable = {spec.metric_id: spec for spec in list_runnable_metrics()}
    normalized_tokens = [str(t).strip() for t in tokens if str(t).strip()]
    magic_tokens = {"all", "both"}
    if any(t in magic_tokens for t in normalized_tokens) and len(normalized_tokens) > 1:
        raise SystemExit("--only cannot mix all/both with explicit metric ids")
    if not normalized_tokens or any(t in magic_tokens for t in normalized_tokens):
        return [
            spec.metric_id
            for spec in runnable.values()
            if metric_enabled(cfg, spec.metric_id)
        ]

    out = []
    for token in normalized_tokens:
        try:
            metric_id = normalize_metric_id(token)
        except KeyError as exc:
            known = ", ".join(sorted(set(runnable) | {s.legacy_name for s in runnable.values()}))
            raise SystemExit(f"unknown --only metric '{token}'. Known runnable metrics: {known}") from exc
        if metric_id not in runnable:
            raise SystemExit(f"--only {token} is a readout metric, not a runnable sweep metric")
        out.append(metric_id)
    return list(dict.fromkeys(out))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODEL_TAGS,
                    help="要跑的模型 tag（对应 config.models 的 key）")
    ap.add_argument("--only", nargs="+", default=["all"],
                    help="要跑的指标: all/both、bf1、cf2 或完整 metric id")
    ap.add_argument("--device", default=None, help="覆盖 config 的 inference.device")
    ap.add_argument("--output-root", default=None, help="覆盖 config.output.root")
    ap.add_argument("--run-name", default=None, help="覆盖 config.output.run_name")
    ap.add_argument("--no-analysis", action="store_true")
    ap.add_argument("--no-sanity", action="store_true",
                    help="跳过 sanity report 生成")
    return ap.parse_args()


def main():
    setup_logging()
    log = logging.getLogger("lvr_eval.main")
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    validate_config(cfg)
    if args.device:
        cfg["inference"]["device"] = args.device
    if args.output_root:
        cfg["output"]["root"] = args.output_root
    if args.run_name:
        cfg["output"]["run_name"] = args.run_name

    # run dir
    run_name = cfg["output"].get("run_name") or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg["output"]["root"], run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_snapshot.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    prereg_cfg = cfg.get("preregistration", {}) or {}
    manifest_path = prereg_cfg.get("manifest_path", str(DEFAULT_MANIFEST_PATH))
    if prereg_cfg.get("enabled", True):
        manifest = load_manifest(manifest_path)
        lock = write_lock(out_dir, manifest, manifest_path=manifest_path)
        log.info("prereg lock = %s sha256=%s", manifest_path, lock["sha256"])
    log.info("run dir = %s", out_dir)

    # 数据只加载一次（所有模型共用同一 probe set，保证可比）
    samples = load_probe_set(cfg["data"])
    if not samples and not cfg["data"].get("allow_empty", False):
        raise SystemExit(
            "probe set is empty. Check data.json_path/image_root/require_lvr_placeholder "
            "or set data.allow_empty=true for debugging only."
        )

    metric_ids = selected_metric_ids(args.only, cfg)
    log.info("selected metrics = %s", " ".join(metric_ids) if metric_ids else "(none)")
    heartbeat_interval = progress_interval_seconds(cfg)
    if heartbeat_interval is None:
        log.info("[progress] heartbeat disabled")
    else:
        log.info("[progress] heartbeat interval = %s", format_elapsed(heartbeat_interval))
    task = str((cfg.get("data") or {}).get("source_type") or "unknown")
    validation_cfg = cfg.get("validation", {})
    do_sanity = validation_cfg.get("run_sanity", True) and not args.no_sanity

    ablation_results: dict[str, dict] = {}
    decay_results: dict[str, dict] = {}
    metric_results: list[dict] = []

    model_tags = resolve_model_tags(args.models, cfg, log)
    total_metric_jobs = sum(1 for tag in model_tags if tag in cfg["models"]) * len(metric_ids)
    completed_metric_jobs = 0
    for tag in model_tags:
        if tag not in cfg["models"]:
            log.warning("config 中无模型 %s，跳过", tag)
            continue
        with ProgressHeartbeat(log, f"load_model model={tag}", heartbeat_interval):
            wrapper = load_model(
                cfg["models"][tag],
                dtype=cfg["inference"]["dtype"],
                device=cfg["inference"]["device"],
                cfg=cfg,
            )

        for metric_index, metric_id in enumerate(metric_ids, start=1):
            label = (
                f"metric {completed_metric_jobs + 1}/{total_metric_jobs} "
                f"model={tag} metric={metric_id} samples={len(samples)} "
                f"model_metric={metric_index}/{len(metric_ids)}"
            )
            with ProgressHeartbeat(log, label, heartbeat_interval):
                res = run_metric(metric_id, wrapper, samples, cfg, tag)
            log.info("[progress] RESULT model=%s metric=%s%s", tag, metric_id, metric_payload_summary(res))
            metric_results.append(write_metric_result(out_dir, metric_id, tag, res, task=task))
            completed_metric_jobs += 1
            if metric_id in {"bf1_latent_ablation", "bf1_layer_ablation"}:
                ablation_results[tag] = res
            elif metric_id == "cf2_pf_decay_curve":
                decay_results[tag] = res

        # 释放显存，准备加载下一个模型
        del wrapper
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    if do_sanity:
        with ProgressHeartbeat(log, f"sanity metric_results={len(metric_results)}", heartbeat_interval):
            sanity_reports = run_sanity_for_metric_results(metric_results, cfg)
            sanity_dir = save_sanity_reports(sanity_reports, out_dir)
        if has_failed_checks(sanity_reports):
            msg = f"sanity checks reported failures -> {sanity_dir}"
            if validation_cfg.get("fail_fast", False):
                log.error(msg)
                raise SystemExit(1)
            log.warning(msg)
        else:
            log.info("sanity checks saved -> %s", sanity_dir)

    if not args.no_analysis:
        with ProgressHeartbeat(log, f"analysis metric_results={len(metric_results)}", heartbeat_interval):
            run_analysis(ablation_results, decay_results, out_dir, metric_results=metric_results)

    log.info("ALL DONE -> %s", out_dir)


if __name__ == "__main__":
    main()
