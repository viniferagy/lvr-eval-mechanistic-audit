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
import json
import logging
import os
from pathlib import Path
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
from pipeline.results import write_metric_result
from pipeline.sanity import (
    has_failed_checks,
    run_sanity_for_metric_results,
    save_sanity_reports,
)


LVR_MODEL_NAME = "LVR-7B"
MODEL_READY_POLL_SECONDS = 30
MODEL_READY_STABLE_SECONDS = 10
DOWNLOAD_MARKER_SUFFIXES = (".incomplete", ".tmp", ".part", ".crdownload")
WEIGHT_FILE_GLOBS = (
    "*.safetensors",
    "pytorch_model*.bin",
    "*.pt",
    "*.ckpt",
    "*.gguf",
)
WEIGHT_INDEX_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
DEFAULT_MODEL_TAGS = ["qwen2_5_vl_7b", "lvr_7b"]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def is_lvr_model(cfg_model: dict) -> bool:
    name = str(cfg_model.get("name", "")).lower()
    path_name = Path(str(cfg_model.get("path", ""))).name.lower()
    target = LVR_MODEL_NAME.lower()
    return target == name or target == path_name


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


def order_models_for_execution(tags: list[str], cfg_models: dict, log: logging.Logger) -> list[str]:
    non_lvr: list[str] = []
    lvr: list[str] = []
    for tag in tags:
        cfg_model = cfg_models.get(tag)
        if cfg_model and is_lvr_model(cfg_model):
            lvr.append(tag)
        else:
            non_lvr.append(tag)

    ordered = non_lvr + lvr
    if ordered != tags:
        log.info("LVR-7B 将最后执行: %s", " ".join(ordered))
    return ordered


def resolve_local_path(path_text: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(path_text)))
    if path.is_absolute():
        return path
    return Path.cwd() / path


def first_download_marker(path: Path) -> Path | None:
    if not path.exists() or not path.is_dir():
        return None
    for child in path.rglob("*"):
        if child.is_file() and child.name.endswith(DOWNLOAD_MARKER_SUFFIXES):
            return child
    return None


def first_weight_file(path: Path) -> Path | None:
    for pattern in WEIGHT_FILE_GLOBS:
        for child in path.glob(pattern):
            if child.is_file():
                return child
    return None


def missing_index_shards(path: Path) -> list[str] | None:
    for index_name in WEIGHT_INDEX_FILES:
        index_path = path / index_name
        if not index_path.is_file():
            continue
        try:
            weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        except (OSError, json.JSONDecodeError):
            return [f"{index_name} 还不能读取"]

        shard_names = sorted(set(weight_map.values()))
        if not shard_names:
            return [f"{index_name} 没有 weight_map"]

        missing = [name for name in shard_names if not (path / name).is_file()]
        return missing
    return None


def model_path_status(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "模型目录还不存在"
    if not path.is_dir():
        return False, "模型路径不是目录"

    marker = first_download_marker(path)
    if marker:
        return False, f"检测到下载中的临时文件: {marker}"

    if not (path / "config.json").is_file():
        return False, "缺少 config.json"

    missing_shards = missing_index_shards(path)
    if missing_shards:
        shown = ", ".join(missing_shards[:3])
        more = "" if len(missing_shards) <= 3 else f" 等 {len(missing_shards)} 个"
        return False, f"缺少权重分片: {shown}{more}"
    if missing_shards == []:
        return True, "ready"

    if not first_weight_file(path):
        return False, "缺少权重文件"
    return True, "ready"


def path_signature(path: Path) -> tuple[int, int, int]:
    total_size = 0
    file_count = 0
    newest_mtime_ns = 0
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        try:
            stat = child.stat()
        except OSError:
            continue
        total_size += stat.st_size
        file_count += 1
        newest_mtime_ns = max(newest_mtime_ns, stat.st_mtime_ns)
    return file_count, total_size, newest_mtime_ns


def wait_for_lvr_model_if_needed(tag: str, cfg_model: dict, log: logging.Logger):
    if not is_lvr_model(cfg_model):
        return

    model_path = resolve_local_path(cfg_model["path"])
    log.info("[%s] LVR-7B 排在最后；检查本地模型是否下载完成: %s", tag, model_path)

    started = time.monotonic()
    while True:
        ready, reason = model_path_status(model_path)
        if ready:
            signature = path_signature(model_path)
            time.sleep(MODEL_READY_STABLE_SECONDS)
            ready_after_wait, reason_after_wait = model_path_status(model_path)
            if ready_after_wait and path_signature(model_path) == signature:
                log.info("[%s] LVR-7B 已就绪，开始执行", tag)
                return
            reason = reason_after_wait if not ready_after_wait else "模型文件还在变化"

        elapsed_min = (time.monotonic() - started) / 60
        log.info("[%s] 等待 LVR-7B 下载完成（%.1f min）：%s", tag, elapsed_min, reason)
        time.sleep(MODEL_READY_POLL_SECONDS)


def metric_enabled(cfg: dict, metric_id: str) -> bool:
    """Prefer unified metrics.*.enabled, fallback to legacy section.enabled."""
    metrics_cfg = cfg.get("metrics") or {}
    if metric_id in metrics_cfg:
        return bool(metrics_cfg[metric_id].get("enabled", True))
    legacy_key = get_metric(metric_id).legacy_name
    return bool(cfg.get(legacy_key, {}).get("enabled", False))


def selected_metric_ids(tokens: list[str], cfg: dict) -> list[str]:
    runnable = {spec.metric_id: spec for spec in list_runnable_metrics()}
    normalized_tokens = [str(t).strip() for t in tokens if str(t).strip()]
    if not normalized_tokens or any(t in ("all", "both") for t in normalized_tokens):
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
        cfg = yaml.safe_load(f)
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
    log.info("run dir = %s", out_dir)

    # 数据只加载一次（所有模型共用同一 probe set，保证可比）
    samples = load_probe_set(cfg["data"])

    metric_ids = selected_metric_ids(args.only, cfg)
    log.info("selected metrics = %s", " ".join(metric_ids) if metric_ids else "(none)")
    validation_cfg = cfg.get("validation", {})
    do_sanity = validation_cfg.get("run_sanity", True) and not args.no_sanity

    ablation_results: dict[str, dict] = {}
    decay_results: dict[str, dict] = {}
    metric_results: list[dict] = []

    model_tags = resolve_model_tags(args.models, cfg, log)
    model_tags = order_models_for_execution(model_tags, cfg["models"], log)
    for tag in model_tags:
        if tag not in cfg["models"]:
            log.warning("config 中无模型 %s，跳过", tag)
            continue
        wait_for_lvr_model_if_needed(tag, cfg["models"][tag], log)
        wrapper = load_model(cfg["models"][tag],
                             dtype=cfg["inference"]["dtype"],
                             device=cfg["inference"]["device"])

        for metric_id in metric_ids:
            res = run_metric(metric_id, wrapper, samples, cfg, tag)
            metric_results.append(write_metric_result(out_dir, metric_id, tag, res))
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
        run_analysis(ablation_results, decay_results, out_dir, metric_results=metric_results)

    log.info("ALL DONE -> %s", out_dir)


if __name__ == "__main__":
    main()
