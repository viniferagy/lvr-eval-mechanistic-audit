#!/usr/bin/env python
"""
smoke_test.py
=============
不加载真实模型/不需要 GPU/torch, 验证非模型逻辑接线:
  - corruption 算子 (mask / blur / shuffle / 连续 severity)
  - curve 特征拟合 (AUC / char_severity, 升降型都覆盖)
  - BF-3 / PF-3 / BF-1 / CF-2 sanity report
  - BF-1 / CF-2 结果结构 -> analysis 出图 + radar + summary + spearman

真实 BF-3/PF-3 forward 逻辑需要 GPU+权重, 这里用合成 curve 注入。

  python smoke_test.py
"""
from __future__ import annotations

import copy
import hashlib
import os
import json
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from pipeline.adapters.lvr_qwen import LVRQwenAdapter
from pipeline.adapters.lvr_qwen_traced import TraceRecorder, TracedLVRQwenAdapter
from pipeline.adapters.probe_catalog import list_adapter_probes, validate_adapter_probes
from pipeline.adapters.qwen_vl import QwenVLAdapter
from pipeline.adapters.registry import get_adapter, known_adapters
from pipeline.corruptions import apply_mask, irrelevant_mask, random_mask, relevant_mask
from pipeline.data import load_probe_set
from pipeline.internal_metrics import corrupt_image, get_post_image_text_span
from pipeline.metrics import get_metric, list_metrics, list_runnable_metrics, normalize_metric_id, resolve_readout
from pipeline.metrics.v2.pf_a_corruption_selectivity import run as run_pf_a_metric
from pipeline.metrics.v2.bf_patch_answer_transfer import (
    answer_transfer_rate,
    patch_grid,
    patch_one_pair,
    run as run_bf_patch_metric,
)
from pipeline.metrics.v2.bf_conf_calibrated_progression import run as run_bf_conf_metric
from pipeline.metrics.v2.bf_swap_latent_replacement import run as run_bf_swap_metric
from pipeline.metrics.v2.cf_stage_decay import stage_reduce
from pipeline.metrics.v2.lvr_latent_patch_answer_transfer import (
    TracePolicyViolation,
    _assert_trace_ok,
    parse_candidate,
    reduce_records as reduce_w3_latent_records,
)
from pipeline.metrics.v2.monet_latent_patch_answer_transfer import (
    METRIC_ID as MONET_LATENT_METRIC_ID,
    reduce_records as reduce_monet_latent_records,
)
from pipeline.metrics.v2.output_accuracy_sanity import run as run_output_accuracy_metric
from pipeline.metrics.v2.output_accuracy_sanity import answer_hit
from pipeline.metrics.v2.pf_b_patch_alignment import run as run_pf_b_metric
from pipeline.metrics.v2 import trace_latent as TL
from pipeline.metrics.lvr_generation_trace import run as run_lvr_trace_metric
from pipeline.preregistration import build_lock_payload, hash_manifest, load_manifest
from pipeline.results import make_metric_result
from pipeline.stats.bf_usage_diagnostics import build_diagnostics
from pipeline.stats.bootstrap import paired_bootstrap
from pipeline.stats.mixed_effects import fit_mixed_effects
from pipeline.degradation import curve_features
from pipeline.analysis import build_summary_with_ci, run_analysis
from pipeline import ablation as ABL
from pipeline import internal_metrics as IM
from pipeline.sanity import (
    has_failed_checks,
    run_sanity_for_metric_result,
    run_sanity_for_metric_results,
    run_sanity_suite,
    save_sanity_reports,
)
from tools.validate_spd_range import main as validate_spd_range_main
from tools.validate_capacity_sweep import main as validate_capacity_sweep_main
from tools.build_evidence_pack import main as build_evidence_pack_main
from tools.build_findings_pack import main as build_findings_pack_main
from tools.prepare_maze_planning_hf import main as prepare_maze_planning_hf_main
from tools.prepare_blink_hf import main as prepare_blink_hf_main
from tools.prepare_vstar_hf import main as prepare_vstar_hf_main
from tools.prepare_vstar_local import main as prepare_vstar_local_main
from tools.prepare_vsi_bench_hf import main as prepare_vsi_bench_hf_main
from tools.prepare_vsi_visuals_hf import export_visuals as export_vsi_visuals
from tools.prepare_monet_sft_hf import extract_prompt_answer, strip_latent_tokens
from tools.validate_findings_gate import main as validate_findings_gate_main
from tools.validate_main_matrix import main as validate_main_matrix_main
from tools.validate_monet_latent import main as validate_monet_latent_main
from tools.validate_main_paper_readiness import main as validate_main_paper_readiness_main
from tools.validate_trace_margin_quality import main as validate_trace_margin_quality_main
from tools.validate_trace_latent_gate import main as validate_trace_latent_gate_main
from tools.validate_w3_latent import main as validate_w3_latent_main
from tools.validate_w4_stepsweep import main as validate_w4_stepsweep_main
from tools.merge_vsi_accuracy import main as merge_vsi_accuracy_main
from run_all import metric_enabled, selected_metric_ids
from run_all import ProgressHeartbeat, metric_payload_summary, progress_interval_seconds


def make_img(seed=0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((64, 64, 3)) * 255).astype("uint8")
    return Image.fromarray(arr)


def test_corruption():
    print("== 1. corruption 算子 ==")
    img = make_img()
    for mode in ["mask_50pct", "mask_80pct", "gaussian_blur", "patch_shuffle"]:
        out = corrupt_image(img, mode, seed=0)
        assert out.size == img.size, mode
        print(f"  {mode:14s} -> ok")
    # 连续 severity
    assert corrupt_image(img, "mask", seed=0, severity=0.3).size == img.size
    assert corrupt_image(img, "gaussian_blur", severity=15).size == img.size
    assert np.array_equal(np.asarray(corrupt_image(img, "mask", seed=0, severity=0)), np.asarray(img.convert("RGB")))
    assert np.array_equal(np.asarray(corrupt_image(img, "gaussian_blur", severity=0)), np.asarray(img.convert("RGB")))
    print("  连续 severity (mask=0.3, blur=15) -> ok")
    print("  severity=0 clean baseline -> ok")


def test_progress_helpers():
    print("\n== 1b. progress observability helpers ==")
    import logging

    logger = logging.getLogger("smoke.progress")
    assert progress_interval_seconds({"runtime": {"progress_interval_seconds": 0}}) is None
    assert progress_interval_seconds({"runtime": {"progress_interval_seconds": 3}}) == 3.0
    payload = {
        "reduction": {
            "selectivity": 0.25,
            "native_alignment": 0.9,
            "ignored_text": "x",
        },
        "n": 4,
        "n_error": 0,
    }
    summary = metric_payload_summary(payload)
    assert "n=4" in summary and "n_error=0" in summary and "selectivity=0.25" in summary
    with ProgressHeartbeat(logger, "smoke_short_operation", None):
        pass
    print("  run_all progress heartbeat config and payload summary -> ok")


def test_spans():
    print("\n== 2. adapter span semantics ==")
    try:
        import torch
    except ImportError:
        print("  torch unavailable; skip span unit checks")
        return

    image_pad_id = 99
    qwen_inputs = {"input_ids": torch.tensor([[1, image_pad_id, image_pad_id, 10, 11, 12]])}
    qwen_wrapper = SimpleNamespace(image_pad_id=image_pad_id)
    qwen_spans = QwenVLAdapter("qwen2_5_vl").get_spans(qwen_wrapper, qwen_inputs)
    assert qwen_spans.image_tokens.start == 1 and qwen_spans.image_tokens.end == 3
    assert qwen_spans.latent_tokens is None
    assert qwen_spans.answer_probe_pos == 5
    qwen_query = qwen_spans.preferred_query_span()
    assert (qwen_query.start, qwen_query.end, qwen_query.kind) == (5, 6, "answer_probe_pos")
    print("  Qwen baseline span -> ok")

    lvr_start_id, lvr_id, lvr_latent_end_id, lvr_end_id = 201, 202, 203, 204
    lvr_inputs = {"input_ids": torch.tensor([[
        image_pad_id, image_pad_id,
        lvr_start_id,
        lvr_id, lvr_id, lvr_id,
        lvr_end_id,
        300,
    ]])}
    lvr_wrapper = SimpleNamespace(
        image_pad_id=image_pad_id,
        cfg={},
        model=SimpleNamespace(config=SimpleNamespace(
            lvr_start_id=lvr_start_id,
            lvr_id=lvr_id,
            lvr_latent_end_id=lvr_latent_end_id,
            lvr_end_id=lvr_end_id,
        )),
    )
    lvr_spans = LVRQwenAdapter().get_spans(lvr_wrapper, lvr_inputs)
    assert lvr_spans.image_tokens.start == 0 and lvr_spans.image_tokens.end == 2
    assert lvr_spans.lvr_placeholder_tokens is not None
    assert lvr_spans.lvr_placeholder_tokens.start == 3
    assert lvr_spans.lvr_placeholder_tokens.end == 6
    lvr_query = lvr_spans.preferred_query_span()
    assert (lvr_query.start, lvr_query.end, lvr_query.kind) == (3, 6, "lvr_placeholder_tokens")
    print("  LVR teacher-forced span -> ok")

    no_lvr_inputs = {"input_ids": torch.tensor([[image_pad_id, image_pad_id, 300]])}
    try:
        LVRQwenAdapter().get_spans(lvr_wrapper, no_lvr_inputs)
        raise AssertionError("LVR teacher-forced span fallback should fail")
    except ValueError as exc:
        assert "expected <|lvr|>" in str(exc)
    print("  LVR teacher-forced missing <|lvr|> -> fail-fast ok")

    lvr_wrapper.cfg = {"audit": {"allow_multiple_lvr_placeholders": True}}
    lvr_spans = LVRQwenAdapter().get_spans(lvr_wrapper, lvr_inputs)
    assert lvr_spans.notes["allow_multiple_lvr_placeholders"] is True
    assert lvr_spans.notes["span_semantics"] == "continuous_span_may_include_interleaving_text"
    print("  LVR multi-placeholder debug notes -> ok")

    assert get_post_image_text_span(qwen_inputs["input_ids"], image_pad_id) == (3, 6)
    for rel in ["pipeline/internal_metrics.py", "pipeline/metrics/bf3_confidence_progression.py",
                "pipeline/metrics/pf3_attention_distance.py", "pipeline/ablation.py"]:
        with open(rel, encoding="utf-8") as f:
            source = f.read()
        assert ("get_" + "latent_span") not in source, rel
    print("  legacy latent helper removed from production usage -> ok")


def test_reductions():
    print("\n== 3. curve 归约 + 特征 ==")
    metric_ids = {m.metric_id for m in list_metrics()}
    runnable_ids = {m.metric_id for m in list_runnable_metrics()}
    assert "bf3_confidence_progression" in metric_ids
    assert "pf3_attention_distance" in metric_ids
    assert normalize_metric_id("bf3_legacy") == "bf3_confidence_progression_legacy"
    assert get_metric("pf3_legacy").metric_id == "pf3_attention_distance_legacy"
    assert "lvr_generation_trace_legacy" in runnable_ids
    legacy_summary = [
        {"metric_id": get_metric(name).metric_id, "legacy_name": get_metric(name).legacy_name,
         "has_run_fn": get_metric(name).run_fn is not None}
        for name in ["bf3_legacy", "pf3_legacy", "bf1_legacy", "bf1_layer_legacy", "cf2_legacy", "lvr_trace_legacy"]
    ]
    assert all(row["has_run_fn"] for row in legacy_summary)
    legacy_path = os.path.join(tempfile.gettempdir(), "legacy_registry_test.json")
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump(legacy_summary, f, indent=2)
    assert "bf3_confidence_progression" in runnable_ids
    assert "pf3_attention_distance" in runnable_ids
    bf3_reduce = resolve_readout("bf3").require_reduce()
    pf3_reduce = resolve_readout("pf3").require_reduce()
    bf3 = np.linspace(6.0, 1.0, 28)            # entropy 早高末低
    r = bf3_reduce(bf3)
    print(f"  bf3_reduce: {r}")
    assert r["early_to_late_drop"] > 0
    pf3 = np.concatenate([np.linspace(0.1, 0.5, 14), np.linspace(0.5, 0.2, 14)])
    rp = pf3_reduce(pf3)
    print(f"  pf3_reduce: {rp}")
    assert rp["peak_kl"] >= rp["mean_kl"]
    # 升型曲线(PF-3 decay): severity↑ KL↑
    up = curve_features([0, 0.2, 0.4, 0.6, 0.8], [0.1, 0.2, 0.35, 0.5, 0.6])
    print(f"  curve_features(up):   {up}")
    assert up["rel_change"] > 0 and up["char_severity"] is not None
    # 降型曲线
    down = curve_features([0, 1, 2, 3, 4], [1.0, 0.8, 0.6, 0.4, 0.2])
    print(f"  curve_features(down): {down}")
    assert down["rel_change"] < 0


def test_data_field_mapping():
    print("\n== 4. data field mapping ==")
    out_dir = tempfile.mkdtemp(prefix="lvr_data_")
    img_path = os.path.join(out_dir, "sample.png")
    make_img(seed=7).save(img_path)
    jsonl_path = os.path.join(out_dir, "data.jsonl")
    row = {
        "meta": {"uid": "sample-7"},
        "img_path": "sample.png",
        "prompt_text": "What is shown?",
        "target_text": "noise image",
    }
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    samples = load_probe_set({
        "source_type": "jsonl",
        "jsonl_path": jsonl_path,
        "image_root": out_dir,
        "field_map": {
            "id": "meta.uid",
            "image": "img_path",
            "question": "prompt_text",
            "answer": "target_text",
        },
    })
    assert len(samples) == 1
    assert samples[0].id == "sample-7"
    assert samples[0].question == "What is shown?"
    assert samples[0].answer == "noise image"
    assert samples[0].image.size == (64, 64)
    print("  jsonl custom field_map -> ok")


def test_spd_faith_and_maze_loaders():
    print("\n== 4b. SPD-Faith / Maze loaders ==")
    out_dir = tempfile.mkdtemp(prefix="lvr_new_loaders_")
    img_a = os.path.join(out_dir, "a.png")
    img_b = os.path.join(out_dir, "b.png")
    maze_img = os.path.join(out_dir, "maze.png")
    make_img(seed=10).save(img_a)
    make_img(seed=11).save(img_b)
    make_img(seed=12).save(maze_img)

    spd_path = os.path.join(out_dir, "spd.jsonl")
    with open(spd_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "pair_id": "p0",
            "image_clean": "a.png",
            "image_counterfactual": "b.png",
            "question": "What changed?",
            "gold_answer_clean": "red",
            "gold_answer_counterfactual": "blue",
            "rationale": "color changed",
        }) + "\n")
        f.write(json.dumps({
            "pair_id": "missing",
            "image_clean": "missing.png",
            "image_counterfactual": "b.png",
            "question": "Missing?",
        }) + "\n")
    spd = load_probe_set({
        "source_type": "spd_faith",
        "jsonl_path": spd_path,
        "image_root": out_dir,
        "skip_missing_images": True,
    })
    assert len(spd) == 1
    assert spd[0].paired_id == "p0"
    assert spd[0].counterfactual_image is not None
    assert spd[0].counterfactual_answer == "blue"

    maze_path = os.path.join(out_dir, "maze.json")
    with open(maze_path, "w", encoding="utf-8") as f:
        json.dump([{
            "maze_id": "m0",
            "maze_image": "maze.png",
            "instruction": "Find the path.",
            "path": "RRDD",
            "steps": ["R", "R", "D", "D"],
            "difficulty": "easy",
        }], f)
    maze = load_probe_set({
        "source_type": "maze",
        "json_path": maze_path,
        "image_root": out_dir,
    })
    assert len(maze) == 1
    assert maze[0].task_metadata["steps"] == ["R", "R", "D", "D"]
    summary = {
        "n_spd_pairs": len(spd),
        "n_maze_samples": len(maze),
        "missing_image_count": 1,
        "paired_field_valid_rate": 1.0,
    }
    with open(os.path.join(out_dir, "data_loader_smoke_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("  spd_faith + maze fixtures -> ok")


def test_prepare_maze_planning_fixture():
    print("\n== 4c. MazePlanning prepare fixture ==")
    out_dir = Path(tempfile.mkdtemp(prefix="maze_prepare_fixture_"))
    src_root = out_dir / "source"
    image_root = src_root / "imgs"
    image_root.mkdir(parents=True)
    make_img(seed=13).save(image_root / "maze_000.png")
    raw_path = src_root / "updated_data.json"
    raw_path.write_text(json.dumps([
        {
            "input_text": "USER: Given the maze in the input image <image>, find a path.",
            "input_img": ["imgs/maze_000.png"],
            "label_actions": ["go forward", "turn left", "go forward"],
        }
    ]), encoding="utf-8")
    prepared = out_dir / "prepared"
    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "prepare_maze_planning_hf.py",
            "--input-json", str(raw_path),
            "--image-root", str(src_root),
            "--out", str(prepared),
            "--strict-actions",
        ]
        prepare_maze_planning_hf_main()
    finally:
        sys.argv = old_argv
    manifest = prepared / "manifest.jsonl"
    stats = json.loads((prepared / "prepare_stats.json").read_text(encoding="utf-8"))
    assert stats["n_written"] == 1
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["answer"] == "go forward, turn left, go forward"
    maze = load_probe_set({
        "source_type": "maze",
        "jsonl_path": str(manifest),
        "image_root": str(prepared / "images"),
        "skip_missing_images": False,
    })
    assert len(maze) == 1
    assert maze[0].task_metadata["steps"] == ["go forward", "turn left", "go forward"]
    assert maze[0].bboxes and len(maze[0].bboxes[0]) == 4
    print("  MazePlanning HF/local converter fixture -> ok")


def test_blink_vsi_loaders_and_prepare_fixtures():
    print("\n== 4d. BLINK / VSI loaders and converters ==")
    out_dir = Path(tempfile.mkdtemp(prefix="blink_vsi_fixture_"))
    source = out_dir / "source"
    source.mkdir()
    make_img(seed=140).save(source / "blink.png")
    make_img(seed=141).save(source / "vsi_0.png")
    make_img(seed=142).save(source / "vsi_1.png")

    blink_raw = source / "blink.json"
    blink_raw.write_text(json.dumps([
        {
            "id": "blink0",
            "image": "blink.png",
            "question": "Which object is red?",
            "choices": {"A": "cube", "B": "sphere"},
            "answer": "A",
        }
    ]), encoding="utf-8")
    blink_out = out_dir / "blink_prepared"
    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "prepare_blink_hf.py",
            "--input-json", str(blink_raw),
            "--image-root", str(source),
            "--out", str(blink_out),
            "--max-samples", "1",
        ]
        prepare_blink_hf_main()
    finally:
        sys.argv = old_argv
    blink = load_probe_set({
        "source_type": "blink",
        "jsonl_path": str(blink_out / "manifest.jsonl"),
        "image_root": str(blink_out / "images"),
        "skip_missing_images": False,
    })
    assert len(blink) == 1
    assert "Choices:" in blink[0].question
    assert blink[0].task_metadata["weak_oracle"] is True

    vsi_raw = source / "vsi.json"
    vsi_raw.write_text(json.dumps([
        {
            "id": "vsi0",
            "frames": ["vsi_0.png", "vsi_1.png"],
            "question": "Where did the marker move?",
            "choices": ["left", "right"],
            "answer": "right",
        }
    ]), encoding="utf-8")
    vsi_out = out_dir / "vsi_prepared"
    old_argv = sys.argv
    try:
        sys.argv = [
            "prepare_vsi_bench_hf.py",
            "--input-json", str(vsi_raw),
            "--image-root", str(source),
            "--out", str(vsi_out),
            "--max-samples", "1",
        ]
        prepare_vsi_bench_hf_main()
    finally:
        sys.argv = old_argv
    vsi = load_probe_set({
        "source_type": "vsi",
        "jsonl_path": str(vsi_out / "manifest.jsonl"),
        "image_root": str(vsi_out / "images"),
        "skip_missing_images": False,
    })
    assert len(vsi) == 1
    assert vsi[0].image.size[0] >= 256
    assert "Choices:" in vsi[0].question
    assert vsi[0].answer == "right"

    visual_out = out_dir / "vsi_visuals"
    visual_rows = [
        {
            "id": "vsi_scene_0",
            "idx": 7,
            "dataset": "arkitscenes",
            "scene_name": "scene_000",
            "image": make_img(seed=143),
        },
        {
            "id": "vsi_scene_1",
            "dataset": "scannet",
            "scene_name": "scene_001",
            "frames": [make_img(seed=144), make_img(seed=145), make_img(seed=146)],
        },
    ]
    visual_stats = export_vsi_visuals(
        visual_rows,
        dataset="fixture/VSI",
        split="test",
        out_root=visual_out,
        image_root=None,
        thumb=64,
        quality=90,
    )
    assert visual_stats["n_written"] == 2
    assert (visual_out / "vsi_scene_0.jpg").is_file()
    assert (visual_out / "arkitscenes" / "scene_000.jpg").is_file()
    assert (visual_out / "scannet" / "scene_001.jpg").is_file()

    lookup_raw = source / "vsi_lookup.json"
    lookup_raw.write_text(json.dumps([
        {
            "id": "lookup0",
            "dataset": "arkitscenes",
            "scene_name": "scene_000",
            "question": "Which room is shown?",
            "options": ["kitchen", "garage"],
            "ground_truth": "kitchen",
        }
    ]), encoding="utf-8")
    lookup_out = out_dir / "vsi_lookup_prepared"
    old_argv = sys.argv
    try:
        sys.argv = [
            "prepare_vsi_bench_hf.py",
            "--input-json", str(lookup_raw),
            "--image-root", str(visual_out),
            "--out", str(lookup_out),
            "--max-samples", "1",
        ]
        prepare_vsi_bench_hf_main()
    finally:
        sys.argv = old_argv
    lookup_stats = json.loads((lookup_out / "prepare_stats.json").read_text(encoding="utf-8"))
    assert lookup_stats["n_written"] == 1
    letter_manifest = out_dir / "vsi_letter_manifest.jsonl"
    letter_manifest.write_text(json.dumps({
        "id": "letter0",
        "image": str(source / "vsi_0.png"),
        "question": "Which object?",
        "choices": ["A. table", "B. toilet", "C. door"],
        "answer": "B",
    }) + "\n", encoding="utf-8")
    letter_samples = load_probe_set({
        "source_type": "vsi",
        "jsonl_path": str(letter_manifest),
        "image_root": str(source),
        "skip_missing_images": False,
    })
    assert letter_samples[0].answer["answer"] == "B"
    assert "toilet" in letter_samples[0].answer["aliases"]
    assert answer_hit("toilet", letter_samples[0].answer)
    assert answer_hit("B. toilet", letter_samples[0].answer)
    assert not answer_hit("table", letter_samples[0].answer)
    print("  BLINK/VSI prepare tools and loaders -> ok")


def test_vstar_loader_and_prepare_fixture():
    print("\n== 4e. V*Bench loader and converter ==")
    out_dir = Path(tempfile.mkdtemp(prefix="vstar_fixture_"))
    source = out_dir / "source"
    source.mkdir()
    high_res = Image.new("RGB", (1600, 1200), "white")
    arr = np.asarray(high_res).copy()
    arr[200:360, 300:520] = [255, 0, 0]
    Image.fromarray(arr).save(source / "vstar.png")
    raw = source / "vstar.json"
    raw.write_text(json.dumps([
        {
            "id": "vstar0",
            "image": {"path": "vstar.png"},
            "question": "Which patch contains the red target?",
            "options": ["A", "B", "C", "D"],
            "label": "A",
            "bbox": [300, 200, 520, 360],
            "category": "spatial_relationship_reasoning",
        }
    ]), encoding="utf-8")
    prepared = out_dir / "prepared"
    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "prepare_vstar_hf.py",
            "--input-json", str(raw),
            "--image-root", str(source),
            "--out", str(prepared),
            "--max-samples", "1",
        ]
        prepare_vstar_hf_main()
    finally:
        sys.argv = old_argv
    stats = json.loads((prepared / "prepare_stats.json").read_text(encoding="utf-8"))
    assert stats["n_written"] == 1
    assert stats["n_missing_bbox"] == 0
    assert stats["n_high_resolution"] == 1
    vstar = load_probe_set({
        "source_type": "vstar",
        "jsonl_path": str(prepared / "manifest.jsonl"),
        "image_root": str(prepared / "images"),
        "skip_missing_images": False,
        "require_bbox": True,
    })
    assert len(vstar) == 1
    assert vstar[0].bboxes and vstar[0].task_metadata["high_resolution"] is True
    assert "Choices:" in vstar[0].question
    rel = relevant_mask(vstar[0].image, bboxes=vstar[0].bboxes)
    assert rel.oracle_source == "bbox" and rel.coverage > 0
    print("  V*Bench prepare tool, loader, bbox oracle, and high-res metadata -> ok")


def test_vstar_snapshot_prepare_fixture():
    print("\n== 4f. V*Bench snapshot JSON-bbox prepare path ==")
    out_dir = Path(tempfile.mkdtemp(prefix="vstar_snapshot_fixture_"))
    repo = out_dir / "repo"
    direct = repo / "direct_attributes"
    relative = repo / "relative_position"
    direct.mkdir(parents=True)
    relative.mkdir(parents=True)
    for subdir, target_dir in (("direct_attributes", direct), ("relative_position", relative)):
        image = Image.new("RGB", (1600, 1200), "white")
        arr = np.asarray(image).copy()
        arr[100:180, 240:320] = [0, 0, 255]
        stem = f"sa_{subdir}_00000"
        Image.fromarray(arr).save(target_dir / f"{stem}.jpg")
        (target_dir / f"{stem}.json").write_text(json.dumps({
            "target_object": ["blue square"],
            "bbox": [[240, 100, 80, 80]],  # V*Bench xywh; preparer must convert.
            "question": "What color is the target object?",
            "options": ["blue", "red", "green", "yellow"],
        }), encoding="utf-8")

    prepared = out_dir / "prepared"
    old_argv = sys.argv
    try:
        sys.argv = [
            "prepare_vstar_local.py",
            "--input-dir", str(repo),
            "--out", str(prepared),
            "--copy-images",
            "--expected-min-samples", "2",
        ]
        prepare_vstar_local_main()
    finally:
        sys.argv = old_argv

    stats = json.loads((prepared / "prepare_stats.json").read_text(encoding="utf-8"))
    assert stats["n_written"] == 2
    assert stats["per_category"]["attribute_recognition"] == 1
    assert stats["per_category"]["spatial_relationship_reasoning"] == 1
    assert stats["bbox_area_ratio"]["median"] > 0
    first = json.loads((prepared / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["bboxes"] == [[240.0, 100.0, 320.0, 180.0]]
    assert first["answer"] == "A" and first["answer_text"] == "blue"

    samples = load_probe_set({
        "source_type": "vstar",
        "jsonl_path": str(prepared / "manifest.jsonl"),
        "image_root": str(prepared / "images"),
        "skip_missing_images": False,
        "require_bbox": True,
    })
    assert len(samples) == 2
    assert all(s.bboxes and s.answer == "A" for s in samples)
    assert "A. blue" in samples[0].question
    rel = relevant_mask(samples[0].image, bboxes=samples[0].bboxes)
    dilated = relevant_mask(samples[0].image, bboxes=samples[0].bboxes, dilate_px=32)
    assert rel.oracle_source == "bbox"
    assert dilated.oracle_source == "bbox_dilated_32px"
    assert dilated.coverage > rel.coverage
    print("  V*Bench full-repo snapshot prepare, xywh->xyxy bbox, and dilation -> ok")


def test_image_resize_metadata():
    print("\n== 5. image resize metadata ==")
    adapter = QwenVLAdapter("qwen2_5_vl")
    img = Image.new("RGB", (2000, 1000), color=(128, 128, 128))
    wrapper = SimpleNamespace(cfg={"audit": {"max_image_side": 1000, "max_image_pixels": None}})
    processed, meta = adapter.prepare_image_for_audit(wrapper, img)
    assert processed.size == (1000, 500)
    assert meta["image_original_size"] == [2000, 1000]
    assert meta["image_processed_size"] == [1000, 500]
    assert meta["image_resize_applied"] is True

    wrapper = SimpleNamespace(cfg={"audit": {"max_image_side": None, "max_image_pixels": None}})
    processed, meta = adapter.prepare_image_for_audit(wrapper, img)
    assert processed.size == (2000, 1000)
    assert meta["image_resize_applied"] is False
    print("  resize metadata and default no-resize -> ok")


def test_pf3_corrupt_after_resize():
    print("\n== 6. PF-3 corruption after resize ==")
    adapter = QwenVLAdapter("qwen2_5_vl")
    wrapper = SimpleNamespace(
        adapter=adapter,
        cfg={"audit": {"max_image_side": 1000, "max_image_pixels": None}},
    )
    img = Image.new("RGB", (2000, 1000), color=(255, 255, 255))
    processed, meta = IM.prepare_image_for_audit(wrapper, img)
    corr = corrupt_image(processed, "mask", seed=0, severity=0)
    assert processed.size == (1000, 500)
    assert corr.size == processed.size
    assert np.array_equal(np.asarray(corr), np.asarray(processed))
    assert meta["image_resize_applied"] is True
    print("  corruption operates on processed clean image -> ok")


def test_lvr_json_loader():
    print("\n== 7. LVR JSON loader ==")
    out_dir = tempfile.mkdtemp(prefix="lvr_json_")
    os.makedirs(os.path.join(out_dir, "viscot/flickr30k"), exist_ok=True)
    img_rel = "viscot/flickr30k/sample.png"
    img_path = os.path.join(out_dir, img_rel)
    make_img(seed=9).save(img_path)

    data = [{
        "dataset": "flickr30k",
        "split": "train",
        "question_id": 31593,
        "image": [img_rel],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nCan you describe the lower apparel?",
            },
            {
                "from": "gpt",
                "value": "<lvr>\n<answer>dark blue denim shorts</answer>",
            },
        ],
        "bboxes": [[0.382, 0.456, 0.718, 0.656]],
    }]

    json_path = os.path.join(out_dir, "lvr.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    samples = load_probe_set({
        "source_type": "lvr_json",
        "json_path": json_path,
        "image_root": out_dir,
        "max_samples": 10,
        "skip_missing_images": False,
    })

    assert len(samples) == 1
    s = samples[0]
    assert s.id == "31593"
    assert "lower apparel" in s.question
    assert s.answer == "dark blue denim shorts"
    assert s.lvr_assistant.startswith("<lvr>")
    assert s.bboxes == [[0.382, 0.456, 0.718, 0.656]]
    assert s.source_dataset == "flickr30k"
    print("  LVR JSON list -> ProbeSample with lvr metadata ok")

    missing_lvr_path = os.path.join(out_dir, "missing_lvr.json")
    missing_lvr_data = copy.deepcopy(data)
    missing_lvr_data[0]["conversations"][1]["value"] = "<answer>dark blue denim shorts</answer>"
    with open(missing_lvr_path, "w", encoding="utf-8") as f:
        json.dump(missing_lvr_data, f)
    samples = load_probe_set({
        "source_type": "lvr_json",
        "json_path": missing_lvr_path,
        "image_root": out_dir,
        "max_samples": 10,
        "skip_missing_images": False,
    })
    assert samples == []
    print("  missing assistant-side <lvr> skipped by default -> ok")

    limited_path = os.path.join(out_dir, "limited_lvr.json")
    bad = copy.deepcopy(data[0])
    bad["question_id"] = 31592
    bad["conversations"][1]["value"] = "<answer>dark blue denim shorts</answer>"
    good = copy.deepcopy(data[0])
    good["question_id"] = 31593
    good2 = copy.deepcopy(data[0])
    good2["question_id"] = 31594
    records = [bad, good, good2]
    with open(limited_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    samples = load_probe_set({
        "source_type": "lvr_json",
        "json_path": limited_path,
        "image_root": out_dir,
        "max_scan_records": 1,
        "skip_missing_images": False,
    })
    assert len(samples) == 0
    samples = load_probe_set({
        "source_type": "lvr_json",
        "json_path": limited_path,
        "image_root": out_dir,
        "max_scan_records": 2,
        "skip_missing_images": False,
    })
    assert len(samples) == 1 and samples[0].id == "31593"
    print("  max_scan_records limits LVR JSON scan -> ok")


def test_lvr_assistant_expansion():
    print("\n== 8. LVR assistant expansion ==")
    sample = SimpleNamespace(
        lvr_assistant="<lvr>\n<answer>B</answer>",
        answer="B",
    )
    wrapper = SimpleNamespace(cfg={"audit": {"lvr_num_tokens": 3}})
    adapter = LVRQwenAdapter()
    text = adapter._teacher_forced_assistant_text(wrapper, sample)
    expected = "<|lvr_start|><|lvr|><|lvr|><|lvr|><|lvr_end|>\n<answer>B</answer>"
    assert "<lvr>" not in text
    assert text == expected
    assert text.count("<|lvr|>") == 3
    assert text.startswith("<|lvr_start|>")
    assert "<|lvr_end|>" in text
    assert "<|lvr_latent_end|>" not in text
    assert "<answer>B</answer>" in text

    multi_sample = SimpleNamespace(
        lvr_assistant="<lvr>\nmid\n<lvr>\n<answer>B</answer>",
        answer="B",
    )
    try:
        adapter._teacher_forced_assistant_text(wrapper, multi_sample)
        raise AssertionError("multiple <lvr> blocks should fail by default")
    except ValueError as exc:
        assert "exactly one <lvr> block" in str(exc)

    multi_wrapper = SimpleNamespace(
        cfg={"audit": {"lvr_num_tokens": 3, "allow_multiple_lvr_placeholders": True}}
    )
    multi_text = adapter._teacher_forced_assistant_text(multi_wrapper, multi_sample)
    assert "<lvr>" not in multi_text
    assert multi_text.count("<|lvr_start|>") == 2
    assert multi_text.count("<|lvr|>") == 6

    missing_sample = SimpleNamespace(
        lvr_assistant="<answer>B</answer>",
        answer="B",
    )
    try:
        adapter._teacher_forced_assistant_text(wrapper, missing_sample)
        raise AssertionError("missing <lvr> should fail by default")
    except ValueError as exc:
        assert "requires sample.lvr_assistant containing <lvr>" in str(exc)

    debug_wrapper = SimpleNamespace(
        cfg={"audit": {"lvr_num_tokens": 3, "allow_synthetic_lvr_assistant": True}}
    )
    debug_text = adapter._teacher_forced_assistant_text(debug_wrapper, missing_sample)
    assert debug_text == expected

    bad_wrapper = SimpleNamespace(
        cfg={
            "audit": {
                "lvr_expansion_mode": "fixed",
                "lvr_num_tokens": 3,
                "lvr_include_latent_end_token": True,
            }
        }
    )
    try:
        adapter._teacher_forced_assistant_text(bad_wrapper, sample)
        raise AssertionError("fixed LVR expansion must reject latent_end insertion")
    except ValueError as exc:
        assert "does not insert <|lvr_latent_end|>" in str(exc)

    print("  <lvr> -> official fixed special token sequence ok")


def test_lvr_trace_position_extraction():
    print("\n== 9. LVR generation trace position extraction ==")
    try:
        import torch
    except ImportError:
        print("  torch unavailable; skip LVR trace extraction check")
        return

    adapter = LVRQwenAdapter()
    lvr_start_id, lvr_id, lvr_latent_end_id, lvr_end_id = 201, 202, 203, 204
    seq = torch.tensor([
        10, 11,
        lvr_start_id,
        lvr_id,
        lvr_id,
        lvr_latent_end_id,
        lvr_end_id,
        99,
    ])

    info = adapter._extract_lvr_positions_from_sequence(
        seq,
        prompt_len=2,
        lvr_start_id=lvr_start_id,
        lvr_id=lvr_id,
        lvr_latent_end_id=lvr_latent_end_id,
        lvr_end_id=lvr_end_id,
    )

    assert info["lvr_token_positions"] == [3, 4]
    assert info["lvr_generated_positions"] == [3, 4]
    assert info["lvr_latent_end_positions"] == [5]
    assert info["lvr_block_spans"] == [[2, 7]]
    assert info["unexpected_lvr_inner_positions"] == []
    print("  latent_end marker excluded from <|lvr|> positions -> ok")


def test_lvr_trace_metric_payload():
    print("\n== 10. LVR trace metric payload fields ==")

    class FakeAdapter:
        def generate_with_trace(self, wrapper, image, question, **kwargs):
            return {
                "generated_text": "B",
                "lvr_generated_positions": [3, 4],
                "lvr_token_positions": [3, 4],
                "lvr_latent_end_positions": [5],
                "lvr_block_spans": [[2, 7]],
                "unexpected_lvr_inner_positions": [],
                "trace_quality": "unit",
                "notes": {"kwargs": kwargs},
            }

    wrapper = SimpleNamespace(adapter=FakeAdapter())
    sample = SimpleNamespace(id="s0", image=make_img(), question="Q?")
    result = run_lvr_trace_metric(
        wrapper,
        [sample],
        {"audit": {"lvr_decoding_strategy": "steps", "lvr_steps": 3}},
        "fake_lvr",
    )
    rec = result["samples"][0]
    assert rec["lvr_generated_positions"] == [3, 4]
    assert rec["lvr_token_positions"] == [3, 4]
    assert rec["lvr_latent_end_positions"] == [5]
    assert rec["lvr_block_spans"] == [[2, 7]]
    assert rec["unexpected_lvr_inner_positions"] == []
    print("  trace metric preserves latent_end/block metadata -> ok")


def test_lvr_trace_required_hard_fail():
    print("\n== 10a. LVR trace v2 required hard fail ==")

    class FallbackAdapter:
        def generate_with_trace(self, wrapper, image, question, **kwargs):
            return {
                "generated_text": "fallback",
                "trace_quality": "approx_from_generated_token_ids",
                "trace_v2_error": "hook failed",
                "missing_modules": ["embed_tokens"],
            }

    wrapper = SimpleNamespace(adapter=FallbackAdapter())
    sample = SimpleNamespace(id="s0", image=make_img(), question="Q?")
    try:
        run_lvr_trace_metric(
            wrapper,
            [sample],
            {"trace_v2": {"required": True, "forbid_fallback": True}},
            "fake_lvr",
        )
        raise AssertionError("required trace v2 should hard-fail on fallback")
    except RuntimeError as exc:
        assert "trace v2 required" in str(exc)
    print("  required trace v2 raises before fallback artifact can pass -> ok")


def test_trace_recorder_fake_model():
    print("\n== 10b. LVR trace v2 fake hooks ==")
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  torch unavailable; skip trace recorder fake model")
        return

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.model = nn.Module()
            self.model.model.embed_tokens = nn.Embedding(8, 4)
            self.model.model.norm = nn.LayerNorm(4)
            self.model.visual = nn.Module()
            self.model.visual.merger = nn.Linear(4, 4)
            self.lm_head = nn.Linear(4, 8)

        def forward(self):
            input_ids = torch.tensor([[1, 2]])
            x = self.model.model.embed_tokens(input_ids)
            x = self.model.visual.merger(x)
            for layer in self.layers:
                x = layer(x)
            x = self.model.model.norm(x)
            return self.lm_head(x)

    model = FakeModel()
    model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(8)])
    wrapper = SimpleNamespace(model=model, layers=model.layers, n_layers=len(model.layers))
    with TraceRecorder(wrapper) as rec:
        model.forward()
    summary = rec.summary()
    assert summary["trace_quality"] == "instrumented_sparse_v0"
    assert summary["sampled_layers"] == [0, 2, 4, 6, 7]
    assert summary["lm_head_called"] is True
    assert summary["last_hidden_state_shape"] == [1, 1, 4]
    with open(os.path.join(tempfile.gettempdir(), "trace_validation.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("  TraceRecorder captures/removes sparse hooks -> ok")


def test_trace_recorder_tensor_capture_and_patch():
    print("\n== 10b2. LVR trace v3 tensor capture + patch ==")
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  torch unavailable; skip trace recorder tensor capture")
        return

    class Output:
        def __init__(self, hidden):
            self.last_position_hidden_state = hidden
            self.logits = torch.zeros(1, 1, 4)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(4, 4)
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.embed_tokens = nn.Embedding(8, 4)
            self.model.language_model.norm = nn.LayerNorm(4)
            self.model.visual = nn.Module()
            self.model.visual.merger = nn.Linear(4, 4)

        def forward(self, input_ids=None, lvr_mode_switch=None, last_position_hidden_state=None, **_):
            hidden = torch.ones(1, 4)
            if last_position_hidden_state is not None:
                hidden = last_position_hidden_state + 1.0
            self.lm_head(hidden)
            return Output(hidden)

    model = FakeModel()
    model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
    wrapper = SimpleNamespace(model=model, layers=model.layers, n_layers=len(model.layers))
    with TraceRecorder(wrapper, capture_tensors=True) as rec:
        model(input_ids=torch.tensor([[1]]), lvr_mode_switch=torch.tensor([False]))
        model(
            input_ids=torch.tensor([[2]]),
            lvr_mode_switch=torch.tensor([True]),
            last_position_hidden_state=torch.zeros(1, 4),
        )
    summary = rec.summary()
    assert summary["n_captured_latent_states"] >= 2
    assert summary["hidden_size"] == 4

    patch_state = [torch.full((1, 4), 5.0)]
    with TraceRecorder(wrapper, capture_tensors=True, patch_states=patch_state, patch_steps={0}) as patched:
        out = model(
            input_ids=torch.tensor([[2]]),
            lvr_mode_switch=torch.tensor([True]),
            last_position_hidden_state=torch.zeros(1, 4),
        )
    assert patched.summary()["n_patch_applied"] == 1
    assert float(out.last_position_hidden_state[0, 0]) == 6.0
    try:
        with TraceRecorder(
            wrapper,
            capture_tensors=True,
            patch_states=[torch.full((1, 3), 5.0)],
            patch_steps={0},
        ):
            model(
                input_ids=torch.tensor([[2]]),
                lvr_mode_switch=torch.tensor([True]),
                last_position_hidden_state=torch.zeros(1, 4),
            )
        raise AssertionError("strict patch shape policy should reject mismatched tensors")
    except RuntimeError as exc:
        assert "shape mismatch" in str(exc)
    with TraceRecorder(
        wrapper,
        capture_tensors=True,
        patch_states=[torch.full((1, 3), 7.0)],
        patch_steps={0},
        patch_shape_policy="slice",
    ) as sliced:
        out = model(
            input_ids=torch.tensor([[2]]),
            lvr_mode_switch=torch.tensor([True]),
            last_position_hidden_state=torch.zeros(1, 4),
        )
    assert sliced.summary()["n_patch_applied"] == 1
    assert float(out.last_position_hidden_state[0, 0]) == 8.0
    print("  TraceRecorder captures latent tensors and patches selected steps -> ok")


def test_preregistration_and_bootstrap():
    print("\n== 10c. preregistration + bootstrap CI ==")
    manifest = load_manifest("prereg/manifest.yaml")
    digest = hash_manifest(manifest)
    assert digest == hash_manifest(yaml.safe_load(yaml.safe_dump(manifest, sort_keys=True)))
    changed = dict(manifest)
    changed["manifest_version"] = "changed"
    assert hash_manifest(changed) != digest
    lock = build_lock_payload(manifest, created_at="2026-05-23T00:00:00+00:00")
    assert lock["sha256"] == digest
    assert len(lock["primary_metrics"]) == 6
    assert lock["experimental_metrics"]
    assert {gate["gate_id"] for gate in lock["gates"]} >= {"W5_capacity_sweep", "W8_evidence_pack"}
    assert {"qwen_query_span_control", "lvr_generation_trace_latent", "monet_transformers_latent_range_gate"} <= set(manifest["evidence_level_policy"])

    ci = paired_bootstrap([1, 2, 3, 4], seed=1, n_resamples=200)
    assert ci is not None and ci.n == 4
    ci2 = paired_bootstrap([2, 4, 6], [1, 2, 3], seed=1, n_resamples=200)
    assert ci2 is not None and abs(ci2.mean - 2.0) < 1e-9
    rows = build_summary_with_ci([
        make_metric_result("bf3_confidence_progression_legacy", "fake", {
            "samples": [
                {"id": "a", "reduction": {"early_to_late_drop": 1.0}},
                {"id": "b", "reduction": {"early_to_late_drop": 2.0}},
            ]
        })
    ], seed=1)
    assert rows and rows[0]["scalar"] == "early_to_late_drop"
    grouped_rows = build_summary_with_ci([
        make_metric_result("bf_patch_answer_transfer", "fake", {
            "samples": [
                {"id": "a0", "paired_id": "a", "reduction": {"continuous_margin_shift": 1.0}},
                {"id": "a1", "paired_id": "a", "reduction": {"continuous_margin_shift": 3.0}},
                {"id": "b0", "paired_id": "b", "reduction": {"continuous_margin_shift": 5.0}},
            ]
        }, task="spd_faith")
    ], seed=1)
    row = next(r for r in grouped_rows if r["scalar"] == "continuous_margin_shift")
    assert row["n"] == 2 and abs(row["mean"] - 3.5) < 1e-9
    assert row["task"] == "spd_faith"
    transfer_rows = build_summary_with_ci([
        make_metric_result("bf_patch_answer_transfer", "fake", {
            "samples": [
                {"id": "a", "paired_id": "a", "reduction": {"answer_transfer_rate": 1.0}},
                {"id": "b", "paired_id": "b", "reduction": {"answer_transfer_rate": 0.0}},
            ]
        }),
        make_metric_result("bf_swap_latent_replacement", "fake", {
            "samples": [
                {"id": "a", "paired_id": "a", "reduction": {"swap_answer_transfer_rate": 1.0}},
                {"id": "b", "paired_id": "b", "reduction": {"swap_answer_transfer_rate": 0.0}},
            ]
        }),
    ], seed=1)
    assert any(r["scalar"] == "answer_transfer_rate" for r in transfer_rows)
    assert any(r["scalar"] == "swap_answer_transfer_rate" for r in transfer_rows)
    mixed = fit_mixed_effects(metric_results=[
        make_metric_result("pf_a_corruption_selectivity", "m0", {
            "task": "spd_faith",
            "reduction": {"selectivity": 0.1, "n": 3},
        }),
        make_metric_result("pf_a_corruption_selectivity", "m1", {
            "task": "maze",
            "reduction": {"selectivity": 0.2, "n": 3},
        }),
    ])
    assert mixed["status"].startswith("implemented") and "pf_a_corruption_selectivity" in mixed["models"]
    for item in manifest["primary_metrics"]:
        spec = get_metric(item["metric_id"])
        schema = spec.require_run().__globals__.get("build_schema", lambda: {})()
        scalars = set(schema.get("scalars") or [])
        assert item["primary_scalar"] in scalars, item
        assert item["status"] in {"runnable_v0_validated", "main_shortest_path_primary"}
    exp = manifest.get("experimental_metrics", [])
    latent_scalars = {
        item["primary_scalar"]
        for item in exp
        if item["metric_id"] == "lvr_latent_patch_answer_transfer"
    }
    assert {"latent_answer_transfer_rate", "best_step_transfer_rate"} <= latent_scalars
    assert get_metric("latent_patch").metric_id == "lvr_latent_patch_answer_transfer"
    trace_cfg = yaml.safe_load(Path("config.trace_v2.yaml").read_text())
    assert trace_cfg["models"]["lvr_7b"]["arch"] == "lvr_qwen2_5_vl_traced"
    assert trace_cfg["metrics"]["lvr_generation_trace"]["enabled"] is True
    assert metric_enabled({"metrics": {"bf_patch_answer_transfer": None}}, "bf_patch_answer_transfer") is True
    try:
        selected_metric_ids(["all", "bf_patch"], {"metrics": {}})
        raise AssertionError("--only all bf_patch should fail")
    except SystemExit as exc:
        assert "cannot mix" in str(exc)
    print("  manifest hash/lock + bootstrap summary_with_ci -> ok")


def test_bf_usage_diagnostics_fixture():
    print("\n== 10c2. BF usage diagnostics ==")

    def rec(idx, clean, shift, transfer, source="candidate_sequence_logprob"):
        patched = clean + shift
        return {
            "id": f"s{idx}",
            "paired_id": f"p{idx}",
            "clean_margin": clean,
            "patched_margin": patched,
            "continuous_margin_shift": shift,
            "logprob_margin_shift": shift,
            "margin_source": source,
            "clean_source_logprob": -2.0,
            "clean_target_logprob": -2.0 - clean,
            "patched_source_logprob": -2.0,
            "patched_target_logprob": -2.0 - patched,
            "answer_transferred": transfer,
            "source_answer_token_ids": [1],
            "target_answer_token_ids": [2],
        }

    patch_payload = {
        "model": "fake",
        "task": "spd_faith",
        "cells": [{
            "layer": 7,
            "position_bucket": "query",
            "records": [
                rec(0, -0.05, 0.20, True),
                rec(1, -0.20, -0.30, False),
                rec(2, -0.80, 0.10, False, source="latent_logit_lens"),
            ],
        }],
    }
    swap_payload = {
        "model": "fake",
        "task": "spd_faith",
        "cells": [{
            "layer": 7,
            "position_bucket": "query",
            "records": [
                rec(0, -0.05, 0.40, True),
                rec(1, -0.20, 0.20, True),
            ],
        }],
        "control_cells": [{
            "layer": 7,
            "position_bucket": "query",
            "control": "random_pair_swap",
            "records": [
                rec(0, -0.05, 0.10, False),
                rec(1, -0.20, -0.10, False),
            ],
        }],
    }
    diagnostics = build_diagnostics([
        make_metric_result("bf_patch_answer_transfer", "fake", patch_payload, task="spd_faith"),
        make_metric_result("bf_swap_latent_replacement", "fake", swap_payload, task="spd_faith"),
    ])
    patch_summary = next(
        row for row in diagnostics["summaries"]
        if row["metric_id"] == "bf_patch_answer_transfer"
    )
    assert patch_summary["n_shift"] == 3
    assert abs(patch_summary["mean_signed_shift"] - 0.0) < 1e-9
    assert patch_summary["mean_abs_shift"] > 0.19
    assert abs(patch_summary["directional_accuracy"] - (2 / 3)) < 1e-9
    assert patch_summary["agreement"]["transfer_yes_direction_correct"] == 1
    bins = {row["boundary_bin"]: row for row in patch_summary["boundary_bins"]}
    assert bins["near"]["n_shift"] == 1
    assert bins["medium"]["n_shift"] == 1
    assert bins["far"]["n_shift"] == 1
    sources = {row["margin_source"]: row for row in patch_summary["margin_sources"]}
    assert sources["candidate_sequence_logprob"]["n_shift"] == 2
    assert sources["latent_logit_lens"]["n_shift"] == 1
    control_rows = diagnostics["control_normalized_rows"]
    assert len(control_rows) == 1
    assert control_rows[0]["control"] == "random_pair_swap"
    assert control_rows[0]["abs_shift_minus_control"] > 0
    print("  BF usage diagnostics direction/abs/boundary/source/control -> ok")


def test_v2_corruptions_and_patch_schema():
    print("\n== 10d. v2 corruption + BF-Patch schema fixtures ==")
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    rel = relevant_mask(img, bboxes=[[0.0, 0.0, 0.5, 0.5]])
    irr = irrelevant_mask(img, rel, seed=0)
    rnd1 = random_mask(img, coverage=0.2, seed=3)
    rnd2 = random_mask(img, coverage=0.2, seed=3)
    fallback = relevant_mask(img)
    assert rel.oracle_source == "bbox"
    assert irr.oracle_source == "irrelevant_from_bbox"
    assert rnd1.oracle_source == "random"
    assert fallback.oracle_source == "center_fallback"
    assert int((rel.data & irr.data).sum()) == 0
    assert np.array_equal(rnd1.data, rnd2.data)
    assert np.array_equal(np.asarray(apply_mask(img, rel, severity=0)), np.asarray(img))
    half = apply_mask(img, rel, fill=(0, 0, 0), severity=0.5)
    assert 120 <= int(np.asarray(half)[0, 0, 0]) <= 135
    grid = patch_grid()
    assert len(grid) == 15
    assert grid[0] == {"layer": 0, "position_bucket": "image"}
    rate = answer_transfer_rate([
        {"answer_transferred": True},
        {"answer_transferred": False},
    ])
    assert rate == 0.5
    schema_path = os.path.join(tempfile.gettempdir(), "v2_metric_schema_smoke.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump({"pf_a_fixture": True, "bf_patch_grid_cells": len(grid)}, f, indent=2)
    print("  PF-A masks + BF-Patch 5x3 grid -> ok")


def test_pf_a_metric_fixture(monkeypatch=None):
    print("\n== 10e. PF-A runnable reduction fixture ==")
    from pipeline.data import ProbeSample

    class FakeInternal:
        calls = []

        @staticmethod
        def query_image_attention_kl_with_meta_from_sample(_wrapper, sample, corrupted_image):
            arr = np.asarray(corrupted_image.convert("RGB"))
            dark = float((arr == 0).all(axis=2).mean())
            FakeInternal.calls.append(dark)
            return {
                "curve": np.asarray([dark, dark * 2.0, dark * 3.0]),
                "seq_len": 8,
                "n_total": 1,
                "n_success": 1,
                "n_skipped": 0,
                "skip_reasons": {},
                "query_target_kind": "fixture",
                "query_span": {"start": 5, "end": 6, "length": 1},
                "image_span": {"start": 1, "end": 5, "length": 4},
            }

        @staticmethod
        def pf3_reduce(curve):
            return IM.pf3_reduce(curve)

    import pipeline.metrics.v2.pf_a_corruption_selectivity as pf_a_mod

    old_im = pf_a_mod.IM
    pf_a_mod.IM = FakeInternal
    try:
        sample = ProbeSample(
            id="fixture",
            image=Image.new("RGB", (16, 16), color=(255, 255, 255)),
            question="Where is the target?",
            answer="top-left",
            bboxes=[[0.0, 0.0, 0.5, 0.5]],
        )
        result = run_pf_a_metric(None, [sample], {"pf_a": {"seed": 7}}, "fake")
    finally:
        pf_a_mod.IM = old_im

    rec = result["samples"][0]
    assert result["schema"]["status"] == "runnable_v0"
    assert result["config"]["comparison"] == "clean_vs_region_masked_attention_kl"
    assert rec["selectivity"] is not None
    assert rec["reduction"]["selectivity"] == rec["irrelevant_kl"] - rec["relevant_kl"]
    assert result["reduction"]["n"] == 1
    assert len(FakeInternal.calls) == 3
    print("  PF-A metric emits selectivity/reductions from explicit masked-image KL -> ok")


def test_bf_patch_metric_fixture():
    print("\n== 10f. BF-Patch hook fixture ==")
    import torch
    import torch.nn as nn
    from pipeline.adapters.spans import AuditSpans, TokenSpan
    from pipeline.data import ProbeSample

    class TinyTokenizer:
        vocab = {"red": 1, "blue": 2, "light": 1, "green": 3}

        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

        def encode(self, text, add_special_tokens=False):
            return [self.vocab[part] for part in str(text).strip().split()]

        def decode(self, ids, skip_special_tokens=True):
            inv = {v: k for k, v in self.vocab.items()}
            return inv.get(int(ids[0]), str(ids[0]))

    class TinyLayer(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.idx = idx

        def forward(self, hidden):
            return hidden + (self.idx + 1) * 0.1

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([TinyLayer(0), TinyLayer(1)])
            self.device = torch.device("cpu")

        def forward(self, input_ids, **kwargs):
            hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
            for layer in self.layers:
                hidden = layer(hidden)
            logits = torch.zeros(hidden.shape[0], hidden.shape[1], 4, device=hidden.device)
            logits[..., 1] = hidden[..., 0] * 0.1
            logits[..., 2] = hidden[..., 0] * 2.0
            logits[..., 3] = hidden[..., 0] * 0.5
            return SimpleNamespace(logits=logits)

    class TinyAdapter:
        def build_inputs_from_sample(self, wrapper, sample):
            base = 5.0 if sample.answer == "red" else 1.0
            ids = torch.tensor([[base, base + 1, base + 2, base + 3]])
            return {"input_ids": ids}

        def get_spans(self, wrapper, inputs, model_outputs=None):
            return AuditSpans(
                image_tokens=TokenSpan(0, 1, "image_tokens"),
                lvr_placeholder_tokens=TokenSpan(1, 3, "lvr_placeholder_tokens"),
                answer_probe_pos=3,
            )

    model = TinyModel()
    class TinyWrapper:
        def __init__(self, model):
            self.model = model
            self.processor = SimpleNamespace(tokenizer=TinyTokenizer())
            self.adapter = TinyAdapter()
            self.layers = model.layers
            self.n_layers = len(model.layers)

        def build_inputs_from_sample(self, sample):
            return self.adapter.build_inputs_from_sample(self, sample)

    wrapper = TinyWrapper(model)
    sample = ProbeSample(
        id="pair0",
        image=Image.new("RGB", (8, 8), "white"),
        question="color?",
        answer="blue",
        counterfactual_image=Image.new("RGB", (8, 8), "black"),
        counterfactual_answer="red",
        paired_id="pair0",
    )
    one = patch_one_pair(wrapper, sample, layer=0, position_bucket="query")
    assert one["logprob_margin_shift"] is not None
    assert one["logit_margin_shift"] is not None
    assert one["patch_length"] == 2
    multi = ProbeSample(
        id="pair1",
        image=Image.new("RGB", (8, 8), "white"),
        question="color?",
        answer="blue",
        counterfactual_image=Image.new("RGB", (8, 8), "black"),
        counterfactual_answer="light green",
        paired_id="pair1",
    )
    multi_one = patch_one_pair(wrapper, multi, layer=0, position_bucket="query")
    assert multi_one["source_answer_token_ids"] == [1, 3]
    assert multi_one["patched_source_logprob"] is not None
    assert len(model.layers[0]._forward_hooks) == 0
    result = run_bf_patch_metric(
        wrapper,
        [sample],
        {"bf_patch": {"layers": [0], "position_buckets": ["image", "query", "latent"]}},
        "tiny",
    )
    assert result["schema"]["status"] == "runnable_v0"
    assert len(result["cells"]) == 3
    assert result["n_paired"] == 1
    assert result["reduction"]["n_cells"] == 3
    assert len(result["samples"]) == 3
    assert "answer_transfer" in result["samples"][0]["reduction"]
    print("  BF-Patch captures source hidden states, patches target, and cleans hooks -> ok")


def test_bf_swap_and_conf_fixtures():
    print("\n== 10g. BF-Swap / BF-Conf fixtures ==")
    import torch
    import torch.nn as nn
    from pipeline.adapters.spans import AuditSpans, TokenSpan
    from pipeline.data import ProbeSample

    class TinyTokenizer:
        vocab = {"red": 1, "blue": 2}

        def encode(self, text, add_special_tokens=False):
            return [self.vocab[str(text).strip()]]

        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            return {"input_ids": [self.vocab[str(text).strip()]]}

        def decode(self, ids, skip_special_tokens=True):
            inv = {v: k for k, v in self.vocab.items()}
            return inv.get(int(ids[0]), str(ids[0]))

    class TinyLayer(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.idx = idx

        def forward(self, hidden):
            return hidden + (self.idx + 1) * 0.2

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([TinyLayer(0), TinyLayer(1), TinyLayer(2)])
            self.device = torch.device("cpu")

        def forward(self, input_ids, **kwargs):
            hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
            for layer in self.layers:
                hidden = layer(hidden)
            logits = torch.zeros(hidden.shape[0], hidden.shape[1], 4)
            logits[..., 1] = hidden[..., 0] * 0.3
            logits[..., 2] = hidden[..., 0] * 1.2
            return SimpleNamespace(logits=logits)

    class TinyNorm(nn.Module):
        def forward(self, hidden):
            return hidden

    class TinyHead(nn.Module):
        def forward(self, hidden):
            logits = torch.zeros(hidden.shape[0], 4)
            logits[:, 1] = hidden[:, 0] * 0.3
            logits[:, 2] = hidden[:, 0] * 1.2
            return logits

    class TinyAdapter:
        def build_inputs(self, wrapper, image, question):
            return {"input_ids": torch.tensor([[1.0, 2.0, 3.0, 4.0]])}

        def build_inputs_from_sample(self, wrapper, sample):
            base = 5.0 if sample.answer == "red" else 1.0
            return {"input_ids": torch.tensor([[base, base + 1, base + 2, base + 3]])}

        def get_spans(self, wrapper, inputs, model_outputs=None):
            return AuditSpans(
                image_tokens=TokenSpan(0, 1, "image_tokens"),
                lvr_placeholder_tokens=TokenSpan(1, 3, "lvr_placeholder_tokens"),
                answer_probe_pos=3,
            )

    class TinyWrapper:
        def __init__(self):
            self.model = TinyModel()
            self.processor = SimpleNamespace(tokenizer=TinyTokenizer())
            self.adapter = TinyAdapter()
            self.layers = self.model.layers
            self.n_layers = len(self.layers)
            self.final_norm = TinyNorm()
            self.lm_head = TinyHead()
            self.device = "cpu"

        def build_inputs_from_sample(self, sample):
            return self.adapter.build_inputs_from_sample(self, sample)

    wrapper = TinyWrapper()
    sample = ProbeSample(
        id="pair0",
        image=Image.new("RGB", (8, 8), "white"),
        question="color?",
        answer="blue",
        counterfactual_image=Image.new("RGB", (8, 8), "black"),
        counterfactual_answer="red",
        paired_id="pair0",
    )
    assert normalize_metric_id("bf_swap") == "bf_swap_latent_replacement"
    assert normalize_metric_id("bf_conf") == "bf_conf_calibrated_progression"
    swap = run_bf_swap_metric(
        wrapper,
        [sample],
        {"bf_swap": {"layers": [0], "position_buckets": ["query"], "max_pairs": 1}},
        "tiny",
    )
    assert swap["schema"]["status"] == "runnable_v0"
    assert swap["reduction"]["n_cells"] == 1
    assert len(swap["samples"]) == 1
    assert {cell["control"] for cell in swap["control_cells"]} == {
        "self_swap", "reverse_swap", "random_pair_swap"
    }
    p1 = ProbeSample(
        id="pair1",
        image=Image.new("RGB", (8, 8), "white"),
        question="color?",
        answer="blue",
        counterfactual_image=Image.new("RGB", (8, 8), "black"),
        counterfactual_answer="red",
        paired_id="pair1",
    )
    swap_controls = run_bf_swap_metric(
        wrapper,
        [sample, p1],
        {
            "bf_swap": {
                "layers": [0],
                "position_buckets": ["query"],
                "max_pairs": 2,
                "controls": ["random_pair_swap"],
            }
        },
        "tiny",
    )
    random_records = swap_controls["control_cells"][0]["records"]
    assert random_records
    assert all(record["id"] != record["random_pair_source_id"] for record in random_records)
    conf = run_bf_conf_metric(
        wrapper,
        [sample],
        {"bf_conf": {"text_only_control": True}},
        "tiny",
    )
    assert conf["schema"]["status"] == "runnable_v0"
    assert conf["reduction"]["n"] == 1
    assert "gold_logit_slope" in conf["samples"][0]["reduction"]
    assert "text_only_control" in conf["samples"][0]
    print("  BF-Swap and BF-Conf emit runnable reductions and registry aliases -> ok")


def test_cf_stage_and_pf_b_fixtures():
    print("\n== 10h. CF-Stage / PF-B fixtures ==")
    from pipeline.data import ProbeSample

    stages = stage_reduce(np.arange(9, dtype=float))
    assert stages == {"early": 1.0, "mid": 4.0, "late": 7.0}
    assert normalize_metric_id("cf_stage") == "cf_stage_decay"
    assert normalize_metric_id("pf_b") == "pf_b_patch_alignment"

    class FakeInternal:
        calls = []

        @staticmethod
        def query_image_attention_kl_with_meta_from_sample(_wrapper, sample, masked_image):
            arr = np.asarray(masked_image.convert("RGB"))
            dark = float((arr == 0).all(axis=2).mean())
            FakeInternal.calls.append(dark)
            return {
                "curve": np.asarray([dark, dark * 2.0, dark * 3.0]),
                "skip_reasons": {},
                "query_target_kind": "fixture",
                "query_span": [1, 2],
                "image_span": [0, 1],
            }

        @staticmethod
        def pf3_reduce(curve):
            return IM.pf3_reduce(curve)

    import pipeline.metrics.v2.pf_b_patch_alignment as pf_b_mod

    old_im = pf_b_mod.IM
    pf_b_mod.IM = FakeInternal
    try:
        sample = ProbeSample(
            id="fixture",
            image=Image.new("RGB", (16, 16), color=(255, 255, 255)),
            question="Where is the target?",
            answer="top-left",
            bboxes=[[0.0, 0.0, 0.5, 0.5]],
        )
        result = run_pf_b_metric(None, [sample], {"pf_b": {"seed": 3}}, "fake")
    finally:
        pf_b_mod.IM = old_im

    assert result["schema"]["status"] == "runnable_native_v0"
    assert result["reduction"]["n"] == 1
    assert result["samples"][0]["dino"]["available"] is False
    assert result["samples"][0]["native_alignment"] is not None
    assert len(FakeInternal.calls) == 3

    FakeInternal.calls = []
    import pipeline.metrics.v2.pf_b_patch_alignment as pf_b_mod

    old_im = pf_b_mod.IM
    pf_b_mod.IM = FakeInternal
    try:
        dino_result = run_pf_b_metric(
            None,
            [sample],
            {
                "pf_b": {
                    "seed": 3,
                    "use_dino": True,
                    "dino_backend": "mock",
                },
                "validation": {"v2": {"min_samples": 1}},
            },
            "fake",
        )
    finally:
        pf_b_mod.IM = old_im

    dino_sample = dino_result["samples"][0]
    assert dino_sample["dino"]["available"] is True
    assert dino_sample["dino"]["backend"] == "mock"
    assert dino_result["reduction"]["dino_alignment"] is not None
    assert dino_result["reduction"]["dino_region_selectivity"] is not None
    summary_rows = build_summary_with_ci([make_metric_result("pf_b_patch_alignment", "fake", dino_result)])
    assert any(row["scalar"] == "dino_region_selectivity" for row in summary_rows)
    sanity = run_sanity_for_metric_results(
        [make_metric_result("pf_b_patch_alignment", "fake", dino_result)],
        {"validation": {"v2": {"min_samples": 1}}},
    )
    assert not has_failed_checks(sanity), sanity
    print("  CF-Stage stage reducer + PF-B native and mock-DINO alignment fixture -> ok")


def test_v2_sanity_and_validator_fixtures():
    print("\n== 10i. v2 sanity dispatch + SPD validator fixtures ==")
    cfg = {"validation": {"v2": {"min_samples": 1, "min_pairs": 1, "allow_center_fallback": True},
                          "bf3": {"min_layers": 2}}}
    payloads = {
        "lvr_generation_trace": {
            "model": "fake",
            "samples": [{
                "trace_quality": "instrumented_sparse_v0",
                "missing_modules": [],
                "n_lvr_mode_steps": 1,
                "n_hidden_feedback_steps": 1,
                "lm_head_called": True,
            }],
        },
        "pf_a_corruption_selectivity": {
            "model": "fake",
            "reduction": {"selectivity": 1.0, "relevant_kl": 0.1, "irrelevant_kl": 1.1, "random_kl": 0.5, "n": 1},
            "samples": [{"id": "s", "relevant_irrelevant_overlap": 0, "relevant_oracle_source": "bbox"}],
        },
        "pf_b_patch_alignment": {
            "model": "fake",
            "config": {"use_dino": False},
            "reduction": {"native_alignment": 0.8, "relevant_alignment": 0.8, "irrelevant_alignment": 0.5, "random_alignment": 0.6, "n": 1},
            "samples": [{"id": "s", "relevant_oracle_source": "bbox", "dino": {"available": False}}],
        },
        "bf_patch_answer_transfer": {
            "model": "fake",
            "n_paired": 1,
            "cells": [{"n_success": 1, "n_error": 0, "records": [{
                "source_answer_token_ids": [1, 2],
                "target_answer_token_ids": [3],
                "clean_margin": -1.0,
                "patched_margin": 0.5,
            }]}],
        },
        "bf_swap_latent_replacement": {
            "model": "fake",
            "n_paired": 1,
            "cells": [{"n_success": 1, "n_error": 0, "records": [{
                "source_answer_token_ids": [1],
                "target_answer_token_ids": [2],
                "clean_margin": -0.2,
                "patched_margin": 0.3,
            }]}],
            "control_cells": [
                {"control": "self_swap", "swap_margin_shift": 0.0, "n_success": 1, "n_error": 0},
                {"control": "reverse_swap", "swap_margin_shift": -0.1, "n_success": 1, "n_error": 0},
                {"control": "random_pair_swap", "swap_margin_shift": 0.1, "n_success": 1, "n_error": 0},
            ],
        },
        "bf_conf_calibrated_progression": {
            "model": "fake",
            "reduction": {"gold_logit_slope": 0.1},
            "samples": [{"id": "s", "curve": [1.0, 0.5], "text_only_control": {"available": True}}],
        },
        "cf_stage_decay": {
            "model": "fake",
            "reduction": {"late_delta": -0.4, "late_retention": 0.6},
            "families": {"mask": {"records": [
                {"severity": 0.0, "stages": {"early": 1.0, "mid": 1.0, "late": 1.0}},
                {"severity": 0.8, "stages": {"early": 0.8, "mid": 0.7, "late": 0.6}},
            ]}},
        },
    }
    for metric_id, payload in payloads.items():
        reports = run_sanity_for_metric_result(metric_id, payload, cfg)
        assert reports and not has_failed_checks(reports), metric_id
    bad_trace = dict(payloads["lvr_generation_trace"])
    bad_trace["samples"] = [{"trace_quality": "approximate_legacy", "missing_modules": [], "lm_head_called": True}]
    assert has_failed_checks(run_sanity_for_metric_result("lvr_generation_trace", bad_trace, cfg))

    run_dir = Path(tempfile.mkdtemp(prefix="spd_validator_"))
    metrics_dir = run_dir / "metrics"
    sanity_dir = run_dir / "sanity"
    metrics_dir.mkdir()
    sanity_dir.mkdir()
    primary_fixture = {
        "pf_a_corruption_selectivity": ("selectivity", 1.0),
        "pf_b_patch_alignment": ("native_alignment", 0.8),
        "bf_patch_answer_transfer": ("logprob_margin_shift", 0.2),
        "bf_swap_latent_replacement": ("swap_margin_shift", 0.3),
        "bf_conf_calibrated_progression": ("gold_logit_slope", 0.1),
        "cf_stage_decay": ("late_delta", -0.4),
    }
    model_fixture = ("qwen2_5_vl_3b", "qwen2_5_vl_7b", "lvr_7b")
    for model in model_fixture:
        for metric_id, scalar in {
            "pf_a_corruption_selectivity": ("selectivity", 1.0),
            "pf_b_patch_alignment": ("native_alignment", 0.8),
            "bf_patch_answer_transfer": ("logprob_margin_shift", 0.2),
            "bf_swap_latent_replacement": ("swap_margin_shift", 0.3),
            "bf_conf_calibrated_progression": ("gold_logit_slope", 0.1),
            "cf_stage_decay": ("late_delta", -0.4),
        }.items():
            payload = {"reduction": {scalar[0]: scalar[1]}}
            if metric_id in {"bf_patch_answer_transfer", "bf_swap_latent_replacement"}:
                payload.update({"n_paired": 1, "cells": [{"n_success": 1, "n_error": 0}]})
            if metric_id == "bf_swap_latent_replacement":
                payload["control_cells"] = [
                    {"control": "self_swap", "swap_margin_shift": 0.0},
                    {"control": "reverse_swap", "swap_margin_shift": -0.1},
                    {"control": "random_pair_swap", "swap_margin_shift": 0.1},
                ]
            (metrics_dir / f"{metric_id}__{model}.json").write_text(json.dumps({
                "metric_id": metric_id,
                "model": model,
                "payload": payload,
            }))
    (run_dir / "summary_with_ci.json").write_text(json.dumps([
        {"metric_id": metric_id, "model": model, "scalar": scalar, "n": 1}
        for metric_id, (scalar, _value) in primary_fixture.items()
        for model in model_fixture
    ]))
    (sanity_dir / "summary_sanity.json").write_text(json.dumps({"overall_status": "pass"}))
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["validate_spd_range.py", str(run_dir), "--min-pairs", "1"]
        validate_spd_range_main()
        low_dir = Path(tempfile.mkdtemp(prefix="spd_validator_low_"))
        (low_dir / "metrics").mkdir()
        (low_dir / "summary_with_ci.json").write_text("[]")
        sys.argv = ["validate_spd_range.py", str(low_dir), "--min-pairs", "1"]
        try:
            validate_spd_range_main()
            raise AssertionError("validator should reject missing metrics")
        except SystemExit as exc:
            assert "FAIL:" in str(exc)
    finally:
        sys.argv = old_argv
    print("  v2 sanity dispatch and range validator accept/reject fixtures -> ok")


def test_w3_latent_sanity_and_validator_fixtures():
    print("\n== 10i2. W3 latent patch sanity + validator fixtures ==")
    assert parse_candidate("<|im_start|> Modified.") == "modified"
    reduction = reduce_w3_latent_records([
        {
            "answer_transferred": True,
            "latent_margin_shift": 0.5,
            "n_patch_applied": 1,
            "n_lvr_mode_steps": 2,
            "n_captured_latent_states": 2,
        }
    ], 1)
    assert reduction["latent_answer_transfer_rate"] == 1.0
    payload = {
        "model": "lvr_7b",
        "n_paired": 1,
        "reduction": {
            "latent_answer_transfer_rate": 1.0,
            "latent_margin_shift": 0.5,
            "n_paired": 1,
            "n_success": 1,
            "n_error": 0,
            "n_patch_applied": 1,
            "n_with_lvr_mode": 1,
            "n_with_captured_state": 1,
        },
        "samples": [{
            "id": "p0",
            "paired_id": "p0",
            "trace_quality": "instrumented_sparse_v0",
            "source_trace_quality": "instrumented_sparse_v0",
            "missing_modules": [],
            "trace_v2_error": None,
            "clean_answer": "original",
            "patched_answer": "modified",
            "captured_state_shapes": [[1, 4]],
            "patched_captured_state_shapes": [[1, 4]],
            "reduction": {"latent_answer_transfer_rate": 1.0, "latent_margin_shift": 0.5},
        }],
    }
    cfg = {"validation": {"w3": {"min_pairs": 1, "max_error_ratio": 0.2}}}
    reports = run_sanity_for_metric_result("lvr_latent_patch_answer_transfer", payload, cfg)
    assert reports and not has_failed_checks(reports)
    bad = dict(payload)
    bad["samples"] = [{**payload["samples"][0], "trace_quality": "approximate_legacy"}]
    assert has_failed_checks(run_sanity_for_metric_result("lvr_latent_patch_answer_transfer", bad, cfg))

    run_dir = Path(tempfile.mkdtemp(prefix="w3_latent_validator_"))
    metrics_dir = run_dir / "metrics"
    sanity_dir = run_dir / "sanity"
    metrics_dir.mkdir()
    sanity_dir.mkdir()
    (run_dir / "prereg.lock.json").write_text(json.dumps({"manifest_version": "fixture"}))
    (metrics_dir / "lvr_latent_patch_answer_transfer_lvr_7b.json").write_text(json.dumps({
        "metric_id": "lvr_latent_patch_answer_transfer",
        "model": "lvr_7b",
        "payload": payload,
    }))
    (run_dir / "summary_with_ci.json").write_text(json.dumps([{
        "metric_id": "lvr_latent_patch_answer_transfer",
        "model": "lvr_7b",
        "scalar": "latent_answer_transfer_rate",
        "n": 1,
    }]))
    (sanity_dir / "summary_sanity.json").write_text(json.dumps({"overall_status": "pass"}))
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["validate_w3_latent.py", str(run_dir), "--min-pairs", "1"]
        validate_w3_latent_main()
        low_dir = Path(tempfile.mkdtemp(prefix="w3_latent_validator_low_"))
        (low_dir / "metrics").mkdir()
        sys.argv = ["validate_w3_latent.py", str(low_dir), "--min-pairs", "1"]
        try:
            validate_w3_latent_main()
            raise AssertionError("W3 validator should reject missing metric")
        except SystemExit as exc:
            assert "FAIL:" in str(exc)
    finally:
        sys.argv = old_argv
    print("  W3 latent sanity and validator accept/reject fixtures -> ok")


def test_w4_latent_step_sweep_fixtures():
    print("\n== 10i3. W4 latent step sweep sanity + validator fixtures ==")
    records = [
        {
            "answer_transferred": True,
            "latent_margin_shift": 0.3,
            "n_patch_applied": 2,
            "n_lvr_mode_steps": 2,
            "n_captured_latent_states": 2,
            "step_results": [
                {
                    "step_index": 0,
                    "answer_transferred": False,
                    "latent_margin_shift": -0.1,
                    "n_patch_applied": 1,
                    "trace_quality": "instrumented_sparse_v0",
                    "missing_modules": [],
                },
                {
                    "step_index": 1,
                    "answer_transferred": True,
                    "latent_margin_shift": 0.3,
                    "n_patch_applied": 1,
                    "trace_quality": "instrumented_sparse_v0",
                    "missing_modules": [],
                },
            ],
            "reduction": {
                "latent_answer_transfer_rate": 1.0,
                "latent_margin_shift": 0.3,
                "best_step_transfer_rate": 1.0,
                "best_step_index": 1,
                "step_transfer_auc": 0.5,
                "last_step_transfer_rate": 1.0,
                "n_steps_evaluated": 2,
            },
        }
    ]
    reduction = reduce_w3_latent_records(records, 1)
    assert reduction["best_step_transfer_rate"] == 1.0
    assert reduction["best_step_index"] == 1
    assert reduction["n_steps_evaluated"] == 2
    assert abs(reduction["step_transfer_auc"] - 0.5) < 1e-9

    payload = {
        "model": "lvr_7b",
        "n_paired": 1,
        "reduction": {
            **reduction,
            "n_success": 1,
            "n_error": 0,
            "n_patch_applied": 1,
            "n_with_lvr_mode": 1,
            "n_with_captured_state": 1,
        },
        "samples": [{
            "id": "p0",
            "paired_id": "p0",
            "trace_quality": "instrumented_sparse_v0",
            "source_trace_quality": "instrumented_sparse_v0",
            "missing_modules": [],
            "trace_v2_error": None,
            "clean_answer": "original",
            "patched_answer": "modified",
            "captured_state_shapes": [[1, 4], [1, 4]],
            "patched_captured_state_shapes": [[1, 4]],
            "step_results": records[0]["step_results"],
            "reduction": records[0]["reduction"],
        }],
    }
    cfg = {"validation": {"w3": {"min_pairs": 1}, "w4": {"min_pairs": 1, "min_steps": 2}}}
    reports = run_sanity_for_metric_result("lvr_latent_patch_answer_transfer", payload, cfg)
    assert reports and not has_failed_checks(reports)
    bad = dict(payload)
    bad["reduction"] = {**payload["reduction"], "n_steps_evaluated": 1, "per_step": payload["reduction"]["per_step"][:1]}
    assert has_failed_checks(run_sanity_for_metric_result("lvr_latent_patch_answer_transfer", bad, cfg))

    run_dir = Path(tempfile.mkdtemp(prefix="w4_stepsweep_validator_"))
    metrics_dir = run_dir / "metrics"
    sanity_dir = run_dir / "sanity"
    metrics_dir.mkdir()
    sanity_dir.mkdir()
    (run_dir / "prereg.lock.json").write_text(json.dumps({"manifest_version": "fixture"}))
    (metrics_dir / "lvr_latent_patch_answer_transfer_lvr_7b.json").write_text(json.dumps({
        "metric_id": "lvr_latent_patch_answer_transfer",
        "model": "lvr_7b",
        "payload": payload,
    }))
    (run_dir / "summary_with_ci.json").write_text(json.dumps([
        {
            "metric_id": "lvr_latent_patch_answer_transfer",
            "model": "lvr_7b",
            "scalar": "best_step_transfer_rate",
            "n": 1,
        },
        {
            "metric_id": "lvr_latent_patch_answer_transfer",
            "model": "lvr_7b",
            "scalar": "step_transfer_auc",
            "n": 1,
        },
    ]))
    (sanity_dir / "summary_sanity.json").write_text(json.dumps({"overall_status": "pass"}))
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["validate_w4_stepsweep.py", str(run_dir), "--min-pairs", "1", "--min-steps", "2"]
        validate_w4_stepsweep_main()
        low_dir = Path(tempfile.mkdtemp(prefix="w4_stepsweep_validator_low_"))
        (low_dir / "metrics").mkdir()
        sys.argv = ["validate_w4_stepsweep.py", str(low_dir), "--min-pairs", "1", "--min-steps", "2"]
        try:
            validate_w4_stepsweep_main()
            raise AssertionError("W4 validator should reject missing metric")
        except SystemExit as exc:
            assert "FAIL:" in str(exc)
    finally:
        sys.argv = old_argv
    print("  W4 latent step sweep sanity and validator accept/reject fixtures -> ok")


def test_w5_w8_tooling_fixtures():
    print("\n== 10i4. W5-W8 capacity/evidence tooling fixtures ==")
    w7_scalars = {
        "pf_a_corruption_selectivity": "selectivity",
        "pf_b_patch_alignment": "native_alignment",
        "bf_patch_answer_transfer": "logprob_margin_shift",
        "bf_swap_latent_replacement": "swap_margin_shift",
        "bf_conf_calibrated_progression": "gold_logit_slope",
        "cf_stage_decay": "late_delta",
    }
    w7_models = ("qwen2_5_vl_3b", "qwen2_5_vl_7b", "lvr_7b")

    def write_latent_run(root: Path, n_steps: int):
        metrics_dir = root / "metrics"
        sanity_dir = root / "sanity"
        metrics_dir.mkdir(parents=True)
        sanity_dir.mkdir()
        reduction = {
            "latent_answer_transfer_rate": 0.5,
            "best_step_transfer_rate": 0.6,
            "best_step_index": n_steps - 1,
            "step_transfer_auc": 0.55,
            "last_step_transfer_rate": 0.5,
            "n_steps_evaluated": n_steps,
            "n_paired": 1,
            "n_success": 1,
            "n_error": 0,
        }
        payload = {
            "model": "lvr_7b",
            "n_paired": 1,
            "reduction": reduction,
        }
        (metrics_dir / "lvr_latent_patch_answer_transfer_lvr_7b.json").write_text(json.dumps({
            "metric_id": "lvr_latent_patch_answer_transfer",
            "model": "lvr_7b",
            "payload": payload,
        }))
        (root / "summary_with_ci.json").write_text(json.dumps([
            {"metric_id": "lvr_latent_patch_answer_transfer", "model": "lvr_7b", "scalar": scalar, "n": 1}
            for scalar in ["best_step_transfer_rate", "step_transfer_auc", "last_step_transfer_rate"]
        ]))
        (sanity_dir / "summary_sanity.json").write_text(json.dumps({"overall_status": "pass"}))
        (root / "prereg.lock.json").write_text(json.dumps({"manifest_version": "fixture"}))

    run_a = Path(tempfile.mkdtemp(prefix="w5_capacity_a_"))
    run_b = Path(tempfile.mkdtemp(prefix="w5_capacity_b_"))
    run_w7 = Path(tempfile.mkdtemp(prefix="w7_scale_"))
    write_latent_run(run_a, 2)
    write_latent_run(run_b, 4)
    (run_w7 / "sanity").mkdir()
    (run_w7 / "metrics").mkdir()
    (run_w7 / "sanity" / "summary_sanity.json").write_text(json.dumps({"overall_status": "pass"}))
    ci_rows = []
    for metric_id, scalar in w7_scalars.items():
        for model in w7_models:
            payload = {"model": model, "reduction": {scalar: 0.1}, "n_paired": 1}
            (run_w7 / "metrics" / f"{metric_id}_{model}.json").write_text(json.dumps({
                "metric_id": metric_id,
                "model": model,
                "payload": payload,
            }))
            ci_rows.append({"metric_id": metric_id, "model": model, "scalar": scalar, "n": 1})
    (run_w7 / "summary_with_ci.json").write_text(json.dumps(ci_rows))

    import sys

    old_argv = sys.argv
    try:
        missing_dir = Path(tempfile.mkdtemp(prefix="w8_pack_missing_"))
        missing_out = missing_dir / "pack.md"
        sys.argv = [
            "build_evidence_pack.py",
            "--out",
            str(missing_out),
            "--w3",
            str(missing_dir),
            "--w4",
            str(run_b),
        ]
        try:
            build_evidence_pack_main()
            raise AssertionError("evidence pack builder should hard-fail on missing metric")
        except SystemExit as exc:
            assert "FAIL:" in str(exc)
        sys.argv = [
            "validate_capacity_sweep.py",
            str(run_a),
            str(run_b),
            "--min-pairs",
            "1",
            "--min-steps",
            "1",
            "--expected-steps",
            "2",
            "4",
        ]
        validate_capacity_sweep_main()
        out_path = Path(tempfile.mkdtemp(prefix="w8_pack_")) / "pack.md"
        sys.argv = [
            "build_evidence_pack.py",
            "--out",
            str(out_path),
            "--w3",
            str(run_a),
            "--w4",
            str(run_b),
            "--w5",
            str(run_a),
            str(run_b),
            "--w6",
            str(run_b),
            "--w7",
            str(run_w7),
        ]
        build_evidence_pack_main()
        assert out_path.is_file()
        assert "W5 Capacity Sweep" in out_path.read_text(encoding="utf-8")
    finally:
        sys.argv = old_argv
    print("  capacity sweep validator + evidence pack builder -> ok")


def test_findings_gate_tooling_fixtures():
    print("\n== 10i6. Findings gate tooling fixtures ==")
    w7_scalars = {
        "pf_a_corruption_selectivity": "selectivity",
        "pf_b_patch_alignment": "native_alignment",
        "bf_patch_answer_transfer": "logprob_margin_shift",
        "bf_swap_latent_replacement": "swap_margin_shift",
        "bf_conf_calibrated_progression": "gold_logit_slope",
        "cf_stage_decay": "late_delta",
    }
    maze_scalars = {
        "pf_a_corruption_selectivity": "selectivity",
        "pf_b_patch_alignment": "native_alignment",
        "bf_conf_calibrated_progression": "gold_logit_slope",
        "cf_stage_decay": "late_delta",
    }
    models = ("qwen2_5_vl_3b", "qwen2_5_vl_7b", "lvr_7b")

    def write_latent(root: Path, *, step: bool = False):
        (root / "metrics").mkdir(parents=True)
        (root / "sanity").mkdir()
        reduction = {
            "latent_answer_transfer_rate": 0.5,
            "latent_margin_shift": None,
            "n_paired": 1,
            "n_success": 1,
            "n_error": 0,
            "n_patch_applied": 1,
            "n_with_lvr_mode": 1,
            "n_with_captured_state": 1,
        }
        if step:
            reduction.update({
                "best_step_transfer_rate": 0.5,
                "step_transfer_auc": 0.5,
                "last_step_transfer_rate": 0.5,
                "n_steps_evaluated": 2,
                "per_step": [{"step_index": 0}, {"step_index": 1}],
            })
        (root / "metrics" / "lvr_latent_patch_answer_transfer_lvr_7b.json").write_text(json.dumps({
            "metric_id": "lvr_latent_patch_answer_transfer",
            "model": "lvr_7b",
            "payload": {"model": "lvr_7b", "n_paired": 1, "reduction": reduction},
        }))
        scalars = ["latent_answer_transfer_rate"]
        if step:
            scalars.append("best_step_transfer_rate")
        (root / "summary_with_ci.json").write_text(json.dumps([
            {"metric_id": "lvr_latent_patch_answer_transfer", "model": "lvr_7b", "scalar": scalar, "n": 1}
            for scalar in scalars
        ]))
        (root / "sanity" / "summary_sanity.json").write_text(json.dumps({"overall_status": "pass"}))
        (root / "prereg.lock.json").write_text(json.dumps({"sha256": "fixture"}))

    def write_matrix(root: Path, scalars: dict[str, str]):
        (root / "metrics").mkdir(parents=True)
        (root / "sanity").mkdir()
        rows = []
        for metric_id, scalar in scalars.items():
            for model in models:
                payload = {
                    "model": model,
                    "reduction": {scalar: 0.2, "n": 1},
                    "samples": [{"id": "s0", "reduction": {scalar: 0.2}}],
                }
                if metric_id in {"bf_patch_answer_transfer", "bf_swap_latent_replacement"}:
                    payload["n_paired"] = 1
                    payload["reduction"]["n_paired"] = 1
                (root / "metrics" / f"{metric_id}_{model}.json").write_text(json.dumps({
                    "metric_id": metric_id,
                    "model": model,
                    "payload": payload,
                }))
                rows.append({"metric_id": metric_id, "model": model, "scalar": scalar, "n": 1})
        (root / "summary_with_ci.json").write_text(json.dumps(rows))
        (root / "sanity" / "summary_sanity.json").write_text(json.dumps({"overall_status": "pass"}))
        (root / "prereg.lock.json").write_text(json.dumps({"sha256": "fixture"}))

    w3 = Path(tempfile.mkdtemp(prefix="findings_w3_"))
    w4 = Path(tempfile.mkdtemp(prefix="findings_w4_"))
    w6 = Path(tempfile.mkdtemp(prefix="findings_w6_"))
    spd = Path(tempfile.mkdtemp(prefix="findings_spd_"))
    maze = Path(tempfile.mkdtemp(prefix="findings_maze_"))
    write_latent(w3)
    write_latent(w4, step=True)
    write_latent(w6)
    write_matrix(spd, w7_scalars)
    write_matrix(maze, maze_scalars)

    import sys

    old_argv = sys.argv
    try:
        common = [
            "--w3", str(w3),
            "--w4", str(w4),
            "--w6", str(w6),
            "--spd", str(spd),
            "--maze", str(maze),
            "--min-latent-pairs", "1",
            "--min-spd-samples", "1",
            "--min-maze-samples", "1",
            "--min-steps", "2",
        ]
        sys.argv = ["validate_findings_gate.py", *common]
        validate_findings_gate_main()
        out_path = Path(tempfile.mkdtemp(prefix="findings_pack_")) / "pack.md"
        sys.argv = ["build_findings_pack.py", "--out", str(out_path), *common]
        build_findings_pack_main()
        assert out_path.is_file()
        assert "Findings Evidence Pack" in out_path.read_text(encoding="utf-8")
        sys.argv = [
            "validate_findings_gate.py",
            "--w3", str(w3),
            "--w4", str(w4),
            "--w6", str(w6),
            "--spd", str(spd),
            "--maze", str(maze / "missing"),
            "--min-latent-pairs", "1",
            "--min-spd-samples", "1",
            "--min-maze-samples", "1",
        ]
        try:
            validate_findings_gate_main()
            raise AssertionError("Findings validator should reject missing Maze by default")
        except SystemExit as exc:
            assert "FAIL:" in str(exc)
    finally:
        sys.argv = old_argv
    print("  Findings validator and pack accept/reject fixtures -> ok")


def test_latent_trace_policy_violation():
    print("\n== 10i5. W3/W4 trace policy hard fail fixture ==")
    cfg = {"trace_v2": {"required": True, "forbid_fallback": True}}
    try:
        _assert_trace_ok({"trace_quality": "approx_from_generated_token_ids"}, cfg, "bad")
        raise AssertionError("trace_v2.required should reject fallback trace quality")
    except TracePolicyViolation as exc:
        assert "instrumented_sparse_v0" in str(exc)
    try:
        _assert_trace_ok(
            {
                "trace_quality": "instrumented_sparse_v0",
                "trace_v2_error": "legacy fallback",
                "missing_modules": [],
            },
            cfg,
            "bad",
        )
        raise AssertionError("trace_v2.forbid_fallback should reject trace_v2_error")
    except TracePolicyViolation as exc:
        assert "forbidden" in str(exc)
    _assert_trace_ok(
        {"trace_quality": "instrumented_sparse_v0", "missing_modules": [], "trace_v2_error": None},
        cfg,
        "ok",
    )
    adapter = TracedLVRQwenAdapter()
    wrapper = SimpleNamespace(
        model=object(),
        layers=[],
        n_layers=0,
        cfg={},
    )
    try:
        adapter.generate_with_trace(
            wrapper,
            None,
            "q?",
            trace_capture={"patch_states": [object()], "patch_steps": [0]},
        )
        raise AssertionError("patched traced generation must not fallback after hook failure")
    except Exception as exc:  # noqa: BLE001
        assert "register" in repr(exc) or "Failed" in repr(exc) or "object" in repr(exc)
    print("  latent metric rejects trace fallback under required policy -> ok")


def test_adapter_probe_catalog():
    print("\n== 10j. adapter probe catalog ==")
    probes = list_adapter_probes()
    summary = validate_adapter_probes(probes)
    assert summary["all_complete"] is True
    assert summary["n_models"] == 3
    assert set(summary["model_ids"]) == {"monet", "latent_sketchpad", "crystal"}
    assert summary["main_pool_ready"] == []
    monet = next(row for row in probes if row["model_id"] == "monet")
    assert monet["main_pool_status"] == "candidate_w13_transformers_latent_gate"
    assert "NOVAglow646/Monet-7B" in monet["public_weights"]
    assert "ce_patch_vec" in monet["hookability"]
    print("  Monet / Latent Sketchpad / CrystaL go-no-go metadata -> ok")


def test_monet_preflight_wiring():
    print("\n== 10k. Monet W12 preflight wiring ==")
    assert "monet_qwen2_5_vl" in known_adapters()
    adapter = get_adapter("monet_qwen2_5_vl")
    assert adapter.__class__.__name__ == "MonetQwenAdapter"
    try:
        adapter.generate_with_trace(None, None, "q?")
        raise AssertionError("Monet standard-forward adapter should not claim trace support")
    except NotImplementedError as exc:
        assert "modified vLLM" in str(exc)

    record = {
        "conversations": [
            {"from": "human", "value": "<image>\nSolve this."},
            {"from": "gpt", "value": "<abs_vis_token>hidden</abs_vis_token> Final: blue"},
        ]
    }
    prompt, answer = extract_prompt_answer(record)
    assert prompt == "Solve this."
    assert strip_latent_tokens(answer) == "<latent> Final: blue"

    cfg = yaml.safe_load(Path("config.monet.preflight.yaml").read_text(encoding="utf-8"))
    assert cfg["models"]["monet_7b"]["arch"] == "monet_qwen2_5_vl"
    assert cfg["models"]["monet_7b"]["latent_start_id"] == 151666
    print("  Monet adapter registry, data parser, and config boundary -> ok")


def _monet_latent_payload(n=3, *, n_success=None, include_ci=True):
    n_success = n if n_success is None else n_success
    samples = []
    for idx in range(n):
        if idx >= n_success:
            samples.append({"id": f"s{idx}", "paired_id": f"p{idx}", "error": "RuntimeError('boom')"})
            continue
        samples.append({
            "id": f"s{idx}",
            "paired_id": f"p{idx}",
            "source_answer": "modified",
            "target_answer": "original",
            "clean_answer": "original",
            "patched_answer": "modified",
            "answer_transferred": True,
            "latent_margin_shift": 1.0,
            "trace_quality": "monet_transformers_latent_mode_v0",
            "source_trace_quality": "monet_transformers_latent_mode_v0",
            "target_trace_quality": "monet_transformers_latent_mode_v0",
            "latent_mode_path": "transformers_ce_patch_vec",
            "vllm_scheduler_native": False,
            "n_captured_latent_states": 10,
            "n_patch_applied": 10,
            "captured_state_shapes": [[10, 3584]],
            "target_captured_state_shapes": [[10, 3584]],
            "patch_state_shapes": [[10, 3584]],
            "reduction": {
                "latent_answer_transfer_rate": 1.0,
                "latent_margin_shift": 1.0,
            },
        })
    payload = {
        "model": "monet_7b",
        "config": {
            "latent_mode_path": "transformers_ce_patch_vec",
            "vllm_scheduler_native": False,
        },
        "samples": samples,
        "reduction": reduce_monet_latent_records(samples, n),
        "n_paired": n,
    }
    ci = []
    if include_ci:
        ci = [{
            "metric_id": MONET_LATENT_METRIC_ID,
            "model": "monet_7b",
            "scalar": "latent_answer_transfer_rate",
            "n": n_success,
            "mean": 1.0,
            "ci_low": 1.0,
            "ci_high": 1.0,
        }]
    return payload, ci


def test_monet_latent_gate_wiring():
    print("\n== 10l. Monet W13 latent gate wiring ==")
    spec = get_metric("monet_latent_patch")
    assert spec.metric_id == MONET_LATENT_METRIC_ID
    payload, _ci = _monet_latent_payload(n=3)
    cfg = {"validation": {"monet": {"min_pairs": 3, "max_error_ratio": 0.2}}}
    reports = run_sanity_for_metric_result(MONET_LATENT_METRIC_ID, payload, cfg)
    assert not has_failed_checks(reports)
    bad, _ = _monet_latent_payload(n=3, n_success=1)
    assert has_failed_checks(run_sanity_for_metric_result(MONET_LATENT_METRIC_ID, bad, cfg))
    config = yaml.safe_load(Path("config.monet_latent_patch.range.yaml").read_text(encoding="utf-8"))
    assert config["metrics"][MONET_LATENT_METRIC_ID]["enabled"] is True
    assert config["monet_latent_patch"]["latent_size"] == 10
    print("  Monet latent metric registry, config, and sanity fixtures -> ok")


def test_validate_monet_latent_fixture():
    print("\n== 10m. Monet W13 validator fixtures ==")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run = root / "run"
        (run / "metrics").mkdir(parents=True)
        (run / "sanity").mkdir()
        payload, ci = _monet_latent_payload(n=3)
        (run / "metrics" / f"{MONET_LATENT_METRIC_ID}_monet_7b.json").write_text(json.dumps({
            "metric_id": MONET_LATENT_METRIC_ID,
            "model": "monet_7b",
            "payload": payload,
        }), encoding="utf-8")
        (run / "sanity" / "summary_sanity.json").write_text(json.dumps({"overall_status": "pass"}), encoding="utf-8")
        (run / "summary_with_ci.json").write_text(json.dumps(ci), encoding="utf-8")
        (run / "prereg.lock.json").write_text(json.dumps({
            "sha256": "fake",
            "manifest": {
                "experimental_metrics": [
                    {"metric_id": MONET_LATENT_METRIC_ID, "primary_scalar": "latent_answer_transfer_rate"}
                ],
            },
        }), encoding="utf-8")
        validate_monet_latent_main([str(run), "--min-pairs", "3"])

        missing = root / "missing"
        (missing / "metrics").mkdir(parents=True)
        try:
            validate_monet_latent_main([str(missing), "--min-pairs", "3"])
            raise AssertionError("Monet validator should reject missing artifacts")
        except SystemExit as exc:
            assert "FAIL" in str(exc)
    print("  Monet latent validator accepts/rejects fixtures -> ok")


def test_trace_latent_w14_w15_fixtures():
    print("\n== 10n. W14/W15 trace-latent fixtures ==")
    import torch
    import torch.nn as nn
    from pipeline.data import ProbeSample

    class TinyTokenizer:
        vocab = {"original": 1, "modified": 2, "multi": 3, "piece": 4, "other": 5}

        def encode(self, text, add_special_tokens=False):
            text = str(text).strip().lower()
            if text == "multi":
                return [3, 4]
            return [self.vocab.get(text, 1)]

        def decode(self, ids, skip_special_tokens=True):
            inv = {v: k for k, v in self.vocab.items()}
            return inv.get(int(ids[0]), str(ids[0]))

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device("cpu")
            self.lm_head = nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                self.lm_head.weight.zero_()
                self.lm_head.weight[1, 0] = -1.0
                self.lm_head.weight[2, 0] = 1.0

    class TinyTraceAdapter:
        def generate_with_trace(self, wrapper, image, question, **kwargs):
            patch_states = (kwargs.get("trace_capture") or {}).get("patch_states")
            is_source = float(np.asarray(image.convert("RGB")).mean()) < 128.0
            value = 2.0 if is_source else -1.0
            if patch_states:
                value = float(patch_states[0].reshape(-1)[0])
            tensors = [
                torch.full((1, 4), value - 0.2),
                torch.full((1, 4), value),
                torch.full((1, 4), value + 0.2),
            ]
            include_scores = "with_scores" in str(question)
            scores = None
            generated_ids = [2 if value > 0 else 1]
            if include_scores:
                scores = [torch.zeros(1, 4)]
                scores[-1][0, 1] = -value
                scores[-1][0, 2] = value
                if "mismatch" in str(question):
                    generated_ids = [5]
            if "monet_trace" in str(question):
                trace_quality = "monet_vllm_latent_v0"
            else:
                trace_quality = "instrumented_sparse_v0"
            generated = "modified" if value > 0 else "original"
            return {
                "generated_text": generated,
                "scores": scores,
                "prompt_len": 0,
                "generated_ids": generated_ids,
                "decoded_generated_tokens": [wrapper.processor.tokenizer.decode([token_id]) for token_id in generated_ids],
                "score_token_alignment": [
                    {
                        "score_index": idx,
                        "generated_token_index": idx,
                        "generated_token_id": token_id,
                        "generated_token_text": wrapper.processor.tokenizer.decode([token_id]),
                    }
                    for idx, token_id in enumerate(generated_ids[: len(scores or [])])
                ],
                "trace_quality": trace_quality,
                "missing_modules": [],
                "trace_v2_error": None,
                "n_lvr_mode_steps": 1,
                "n_hidden_feedback_steps": 1,
                "n_captured_latent_states": len(tensors),
                "n_patch_applied": 1 if patch_states else 0,
                "captured_state_metadata": [
                    {
                        "kind": "output_last_position_hidden_state",
                        "step_index": idx,
                        "shape": [1, 4],
                    }
                    for idx, _tensor in enumerate(tensors)
                ],
                "_captured_states": [
                    {
                        "kind": "output_last_position_hidden_state",
                        "step_index": idx,
                        "shape": [1, 4],
                        "tensor": tensor,
                    }
                    for idx, tensor in enumerate(tensors)
                ],
            }

    wrapper = SimpleNamespace(
        model=TinyModel(),
        processor=SimpleNamespace(tokenizer=TinyTokenizer()),
        adapter=TinyTraceAdapter(),
        final_norm=None,
        lm_head=None,
    )
    wrapper.lm_head = wrapper.model.lm_head
    sample = ProbeSample(
        id="pair0",
        image=Image.new("RGB", (8, 8), "white"),
        question="q target",
        answer="original",
        counterfactual_image=Image.new("RGB", (8, 8), "black"),
        counterfactual_answer="modified",
        paired_id="pair0",
        bboxes=[[0, 0, 0.5, 0.5]],
    )
    cfg = {
        "trace_v2": {"required": True, "forbid_fallback": True},
        "trace_latent": {
            "enabled": True,
            "required": True,
            "forbid_fallback": True,
            "patch_steps": ["last"],
            "apply_to": "all",
        },
        "validation": {
            "v2": {
                "min_samples": 1,
                "min_pairs": 1,
                "allow_center_fallback": True,
                "self_swap_max_abs_shift": 10.0,
            },
            "bf3": {"min_layers": 1},
        },
        "pf_a": {"seed": 1},
        "pf_b": {"seed": 1, "use_dino": False},
        "bf_patch": {"max_pairs": 1},
        "bf_swap": {"max_pairs": 1, "controls": ["self_swap", "reverse_swap", "random_pair_swap"]},
        "cf_stage": {"families": {"mask": [0.4]}},
    }
    assert TL.enabled(cfg, "pf_a_corruption_selectivity")
    monet_trace = wrapper.adapter.generate_with_trace(
        wrapper,
        sample.image,
        "monet_trace",
        trace_capture={"capture_tensors": True},
    )
    TL.assert_trace_ok(monet_trace, cfg, "monet_future_trace")
    aligned_trace = wrapper.adapter.generate_with_trace(
        wrapper,
        sample.counterfactual_image,
        "with_scores",
        trace_capture={"capture_tensors": True},
    )
    aligned_diag = TL.score_margin_with_diagnostics(
        aligned_trace,
        wrapper,
        "modified",
        "original",
        label="aligned",
    )
    assert aligned_diag["status"] == "pass"
    assert aligned_diag["source"] == "generation_scores_aligned_first_token"
    assert aligned_diag["used_score_index"] == 0

    mismatch_trace = wrapper.adapter.generate_with_trace(
        wrapper,
        sample.counterfactual_image,
        "with_scores mismatch",
        trace_capture={"capture_tensors": True},
    )
    mismatch_diag = TL.score_margin_with_diagnostics(
        mismatch_trace,
        wrapper,
        "modified",
        "original",
        label="mismatch",
    )
    assert mismatch_diag["status"] == "fail"
    assert mismatch_diag["reason"] == "decision_index_mismatch"

    multi_diag = TL.score_margin_with_diagnostics(
        aligned_trace,
        wrapper,
        "multi",
        "original",
        label="multi",
    )
    assert multi_diag["status"] == "fail"
    assert multi_diag["reason"] == "multi_token_candidate"

    missing_scores_diag = TL.score_margin_with_diagnostics(
        wrapper.adapter.generate_with_trace(wrapper, sample.image, "no scores"),
        wrapper,
        "modified",
        "original",
        label="missing_scores",
    )
    assert missing_scores_diag["status"] == "fail"
    assert missing_scores_diag["reason"] == "missing_scores"

    patch_rec = TL.patch_pair(wrapper, sample, cfg)
    assert patch_rec["answer_transferred"] is True
    assert patch_rec["logprob_margin_shift"] is not None
    assert patch_rec["margin_source"] == "latent_logit_lens"
    assert patch_rec["continuous_margin_shift"] is not None
    assert patch_rec["latent_logit_margin_shift"] is not None
    assert patch_rec["clean_score_diagnostic"]["reason"] == "missing_scores"
    assert patch_rec["patched_score_diagnostic"]["reason"] == "missing_scores"

    payloads = {
        "pf_a_corruption_selectivity": run_pf_a_metric(wrapper, [sample], cfg, "lvr_7b"),
        "pf_b_patch_alignment": run_pf_b_metric(wrapper, [sample], cfg, "lvr_7b"),
        "bf_patch_answer_transfer": run_bf_patch_metric(wrapper, [sample], cfg, "lvr_7b"),
        "bf_swap_latent_replacement": run_bf_swap_metric(wrapper, [sample, ProbeSample(
            id="pair1",
            image=Image.new("RGB", (8, 8), "white"),
            question="q target",
            answer="original",
            counterfactual_image=Image.new("RGB", (8, 8), "black"),
            counterfactual_answer="modified",
            paired_id="pair1",
            bboxes=[[0, 0, 0.5, 0.5]],
        )], cfg, "lvr_7b"),
        "bf_conf_calibrated_progression": run_bf_conf_metric(wrapper, [sample], cfg, "lvr_7b"),
    }
    cfg_cf = {**cfg, "trace_latent": {**cfg["trace_latent"], "apply_to": ["cf_stage_decay"]}}
    payloads["cf_stage_decay"] = __import__(
        "pipeline.metrics.v2.cf_stage_decay",
        fromlist=["run"],
    ).run(wrapper, [sample], cfg_cf, "lvr_7b")

    for metric_id, payload in payloads.items():
        assert (payload.get("config") or {}).get("trace_latent") is True, metric_id
        if metric_id in {"bf_patch_answer_transfer", "bf_swap_latent_replacement"}:
            diag = payload["cells"][0]["generation_score_diagnostics"]
            assert diag["diagnostic_records"] >= 1
            assert diag["generation_score_failure_reason_counts"]
        reports = run_sanity_for_metric_result(metric_id, payload, cfg)
        assert reports and not has_failed_checks(reports), metric_id

    run_dir = Path(tempfile.mkdtemp(prefix="trace_latent_gate_"))
    (run_dir / "metrics").mkdir()
    (run_dir / "sanity").mkdir()
    metric_results = []
    for metric_id, payload in payloads.items():
        envelope = make_metric_result(metric_id, "lvr_7b", payload)
        metric_results.append(envelope)
        (run_dir / "metrics" / f"{metric_id}_lvr_7b.json").write_text(
            json.dumps(envelope, indent=2),
            encoding="utf-8",
        )
    (run_dir / "summary_with_ci.json").write_text(
        json.dumps(build_summary_with_ci(metric_results, seed=1), indent=2),
        encoding="utf-8",
    )
    sanity = []
    for metric_id, payload in payloads.items():
        sanity.extend(run_sanity_for_metric_result(metric_id, payload, cfg))
    save_sanity_reports(sanity, str(run_dir))
    run_analysis({}, {}, str(run_dir), metric_results=metric_results)
    validate_trace_latent_gate_main([str(run_dir), "--min-pairs", "1", "--min-samples", "1"])
    validate_trace_margin_quality_main([str(run_dir), "--min-records", "1", "--require-score-diagnostics"])

    bad_dir = Path(tempfile.mkdtemp(prefix="trace_latent_bad_margin_"))
    (bad_dir / "metrics").mkdir()
    (bad_dir / "sanity").mkdir()
    bad_payload = copy.deepcopy(payloads["bf_patch_answer_transfer"])
    for cell in bad_payload.get("cells") or []:
        for record in cell.get("records") or []:
            record["margin_source"] = "parsed_answer_fallback"
            record["effective_margin_shift"] = record.get("logprob_margin_shift")
    bad_env = make_metric_result("bf_patch_answer_transfer", "lvr_7b", bad_payload)
    (bad_dir / "metrics" / "bf_patch_answer_transfer_lvr_7b.json").write_text(
        json.dumps(bad_env, indent=2),
        encoding="utf-8",
    )
    try:
        validate_trace_margin_quality_main([str(bad_dir), "--min-records", "1", "--max-parsed-fallback-ratio", "0.2"])
        raise AssertionError("trace margin validator should reject parsed fallback dominance")
    except SystemExit as exc:
        assert (
            "parsed fallback ratio too high" in str(exc)
            or "has no continuous margin-source records" in str(exc)
        )

    cfg_file = yaml.safe_load(Path("config.lvr_trace_latent.w14_w15.yaml").read_text(encoding="utf-8"))
    assert cfg_file["models"]["lvr_7b"]["arch"] == "lvr_qwen2_5_vl_traced"
    assert cfg_file["trace_latent"]["enabled"] is True
    print("  trace-latent metrics, sanity, validator, and plots -> ok")


def test_output_accuracy_and_main_matrix_fixtures():
    print("\n== 10o. Output accuracy + main-matrix tooling fixtures ==")
    from pipeline.data import ProbeSample

    class TinyAccuracyWrapper:
        def generate(self, images, prompts, max_new_tokens=64):
            return ["The answer is right." for _ in prompts]

    sample = ProbeSample(
        id="acc0",
        image=Image.new("RGB", (8, 8), "white"),
        question="Where?",
        answer="right",
        paired_id="acc0",
        source_dataset="vsi_bench",
    )
    cfg = {
        "output_accuracy": {"batch_size": 1, "max_new_tokens": 8},
        "validation": {"output_accuracy": {"min_samples": 1}},
    }
    payload = run_output_accuracy_metric(TinyAccuracyWrapper(), [sample], cfg, "qwen2_5_vl_3b")
    assert payload["reduction"]["accuracy"] == 1.0
    assert payload["reduction"]["n_total"] == 1
    assert answer_hit("about 12.4 meters", {"numeric_value": 12.5, "abs_tolerance": 0.2})
    assert not answer_hit("about 12.0 meters", {"numeric_value": 12.5, "abs_tolerance": 0.2})
    assert answer_hit("Answer: A", "A")
    assert answer_hit("(B) chair", "B")
    assert not answer_hit("D. pillow", "A")
    assert not answer_hit("The answer appears to be pillow.", "A")
    reports = run_sanity_for_metric_result("output_accuracy_sanity", payload, cfg)
    assert reports and not has_failed_checks(reports)

    run_root = Path(tempfile.mkdtemp(prefix="main_matrix_fixture_"))
    task_metrics = {
        "spd_faith": [
            "pf_a_corruption_selectivity",
            "pf_b_patch_alignment",
            "bf_patch_answer_transfer",
            "bf_swap_latent_replacement",
            "bf_conf_calibrated_progression",
            "cf_stage_decay",
        ],
        "maze": [
            "pf_a_corruption_selectivity",
            "pf_b_patch_alignment",
            "bf_conf_calibrated_progression",
            "cf_stage_decay",
        ],
        "blink": [
            "pf_a_corruption_selectivity",
            "pf_b_patch_alignment",
            "bf_conf_calibrated_progression",
            "cf_stage_decay",
        ],
    }
    models = ["qwen2_5_vl_3b", "qwen2_5_vl_7b", "lvr_7b"]
    metric_results = []
    for task, metrics in task_metrics.items():
        for model in models:
            run_dir = run_root / f"{task}_{model}"
            (run_dir / "metrics").mkdir(parents=True)
            (run_dir / "config_snapshot.yaml").write_text(
                yaml.safe_dump({"data": {"source_type": task}}),
                encoding="utf-8",
            )
            for metric_id in metrics:
                if metric_id == "bf_patch_answer_transfer":
                    p = {"model": model, "task": task, "n_paired": 3, "cells": [], "samples": [{"id": "a", "paired_id": "a", "reduction": {"continuous_margin_shift": 0.1}}], "reduction": {"continuous_margin_shift": 0.1, "n_paired": 3}}
                elif metric_id == "bf_swap_latent_replacement":
                    p = {"model": model, "task": task, "n_paired": 3, "cells": [], "control_cells": [], "samples": [{"id": "a", "paired_id": "a", "reduction": {"continuous_margin_shift": 0.1}}], "reduction": {"continuous_margin_shift": 0.1, "n_paired": 3}}
                else:
                    scalar = {
                        "pf_a_corruption_selectivity": "selectivity",
                        "pf_b_patch_alignment": "native_alignment",
                        "bf_conf_calibrated_progression": "gold_logit_slope",
                        "cf_stage_decay": "late_delta",
                    }[metric_id]
                    p = {"model": model, "task": task, "samples": [{"id": str(i), "reduction": {scalar: 0.1}} for i in range(3)], "reduction": {scalar: 0.1, "n": 3}}
                env = make_metric_result(metric_id, model, p, task=task)
                metric_results.append(env)
                (run_dir / "metrics" / f"{metric_id}_{model}.json").write_text(
                    json.dumps(env, indent=2),
                    encoding="utf-8",
                )
    merged = run_root / "merged"
    merged.mkdir()
    (merged / "summary_with_ci.json").write_text(
        json.dumps(build_summary_with_ci(metric_results), indent=2),
        encoding="utf-8",
    )
    assert {
        ("maze", "qwen2_5_vl_3b", "pf_a_corruption_selectivity", "selectivity"),
        ("spd_faith", "lvr_7b", "bf_patch_answer_transfer", "continuous_margin_shift"),
    } <= {
        (row.get("task"), row.get("model"), row.get("metric_id"), row.get("scalar"))
        for row in build_summary_with_ci(metric_results)
        if int(row.get("n") or 0) > 0
    }
    (merged / "sanity").mkdir()
    (merged / "sanity" / "summary_sanity.json").write_text(
        json.dumps({"overall_status": "pass"}),
        encoding="utf-8",
    )
    fit_mixed_effects(metric_results, out_path=merged / "mixed_effects_summary.json")
    validate_main_matrix_main([
        str(run_root),
        "--tasks", "maze,spd_faith,blink",
        "--models", ",".join(models),
        "--min-samples", "3",
        "--bf-min-pairs", "3",
    ])
    validate_main_paper_readiness_main([str(merged), "--mode", "full_main_matrix"])

    acc_root = Path(tempfile.mkdtemp(prefix="accuracy_matrix_fixture_"))
    for model in models:
        acc_run = acc_root / f"vsi_{model}"
        (acc_run / "metrics").mkdir(parents=True)
        (acc_run / "config_snapshot.yaml").write_text(yaml.safe_dump({"data": {"source_type": "vsi"}}), encoding="utf-8")
        model_payload = copy.deepcopy(payload)
        model_payload["model"] = model
        acc_env = make_metric_result("output_accuracy_sanity", model, model_payload, task="vsi")
        (acc_run / "metrics" / f"output_accuracy_sanity_{model}.json").write_text(
            json.dumps(acc_env, indent=2),
            encoding="utf-8",
        )
    validate_main_matrix_main([str(acc_root), "--accuracy-only", "--min-samples", "1"])
    acc_merged = acc_root / "merged"
    merge_vsi_accuracy_main([str(acc_root), "--out", str(acc_merged), "--min-samples", "1"])
    assert (acc_merged / "vsi_accuracy_summary.json").is_file()
    duplicate_root = acc_root / "duplicate_lvr_failed"
    (duplicate_root / "metrics").mkdir(parents=True)
    failed_payload = copy.deepcopy(payload)
    failed_payload["model"] = "lvr_7b"
    failed_payload["samples"] = [
        {"id": "failed", "answer": "A", "error": "RuntimeError('old failure')"}
    ]
    failed_payload["reduction"] = {"accuracy": None, "n": 0, "n_total": 1, "n_error": 1}
    failed_env = make_metric_result("output_accuracy_sanity", "lvr_7b", failed_payload, task="vsi")
    (duplicate_root / "metrics" / "output_accuracy_sanity_lvr_7b.json").write_text(
        json.dumps(failed_env, indent=2),
        encoding="utf-8",
    )
    acc_merged_dupes = acc_root / "merged_dupes"
    merge_vsi_accuracy_main([
        str(acc_root),
        str(duplicate_root),
        "--out", str(acc_merged_dupes),
        "--min-samples", "1",
    ])
    duplicate_summary = json.loads((acc_merged_dupes / "vsi_accuracy_summary.json").read_text(encoding="utf-8"))
    assert duplicate_summary["models"]["lvr_7b"]["n_samples"] >= 1
    assert duplicate_summary["skipped_duplicate_candidates"]
    for env in metric_results:
        metric_path = merged / "metrics" / f"{env['task']}_{env['metric_id']}_{env['model']}.json"
        metric_path.parent.mkdir(exist_ok=True)
        metric_path.write_text(json.dumps(env, indent=2), encoding="utf-8")
    validate_main_paper_readiness_main([
        str(merged),
        "--mode", "paper_ready",
        "--t4-accuracy-summary", str(acc_merged / "vsi_accuracy_summary.json"),
        "--t4-min-samples", "1",
    ])

    dry = os.popen(
        "bash tools/launch_main_matrix.sh --configs config.main_maze_n1000.yaml "
        "--models qwen2_5_vl_3b --metrics pf_a_corruption_selectivity "
        "--gpus 0 --run-root /tmp/lvr_matrix_dry --dry-run"
    ).read()
    assert "CUDA_VISIBLE_DEVICES=0" in dry and "run_all.py" in dry
    print("  output accuracy, matrix validator, and launch dry-run -> ok")


def test_bf1_does_not_cache_gpu_inputs_static():
    print("\n== 11. BF-1 targeted cache policy ==")
    import inspect

    source = inspect.getsource(ABL.run_targeted_ablation_sweep)
    assert '"inputs": inputs' not in source
    assert '"inputs"' not in source.split("prepared.append", 1)[1].split("})", 1)[0]
    assert "baseline_pf3_by_id" in source
    assert "n_paired" in source
    print("  targeted BF-1 does not store prepared GPU inputs; PF3 delta is paired -> ok")


def fake_bf1(tag, n_layers=28, strength=1.0):
    seed = int.from_bytes(hashlib.sha256(str(tag).encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    base_bf3 = np.linspace(6, 1, n_layers)
    base_pf3 = np.concatenate([np.linspace(0.1, 0.5, n_layers // 2),
                               np.linspace(0.5, 0.2, n_layers - n_layers // 2)])
    layers = []
    for li in range(n_layers):
        crit = np.exp(-((li - n_layers / 2) ** 2) / 8) * strength
        layers.append({
            "layer": li,
            "bf3": {"early_to_late_drop": 5 - 2 * crit},
            "pf3": {"mean_kl": 0.35 - 0.2 * crit},
            "bf3_curve": (base_bf3 + rng.normal(0, 0.05, n_layers)).tolist(),
            "pf3_curve": (base_pf3 + rng.normal(0, 0.01, n_layers)).tolist(),
            "delta": {"bf3": 2 * crit, "pf3": 0.2 * crit},
        })
    return {"model": tag, "mode": "identity", "n_layers": n_layers,
            "bf3_scalar": "early_to_late_drop", "pf3_scalar": "mean_kl",
            "baseline": {"bf3": {"early_to_late_drop": 5.0},
                         "pf3": {"mean_kl": 0.35},
                         "bf3_curve": base_bf3.tolist(),
                         "pf3_curve": base_pf3.tolist()},
            "layers": layers}


def fake_cf2(tag, robust=1.0):
    fams = {}
    for fam, sev in [("mask", [0.2, 0.4, 0.6, 0.8]),
                     ("gaussian_blur", [2, 5, 10, 15, 20])]:
        severities = [0.0] + sev
        curve = [0.1 + robust * 0.5 * (s / max(severities)) for s in severities]
        fams[fam] = {"severities": severities, "curve": curve,
                     "features": curve_features(severities, curve)}
    return {"model": tag, "readout": "pf3", "families": fams}


def test_end_to_end():
    print("\n== 12. sanity + 端到端 analysis(合成数据) ==")
    ablation = {
        "qwen2_5_vl_7b": fake_bf1("qwen2_5_vl_7b", strength=1.4),
        "lvr_7b": fake_bf1("lvr_7b", strength=0.7),
    }
    decay = {
        "qwen2_5_vl_7b": fake_cf2("qwen2_5_vl_7b", robust=1.3),
        "lvr_7b": fake_cf2("lvr_7b", robust=0.8),
    }
    metric_results = [
        make_metric_result("bf1_latent_ablation", tag, result)
        for tag, result in ablation.items()
    ] + [
        make_metric_result("cf2_pf_decay_curve", tag, result)
        for tag, result in decay.items()
    ]
    out_dir = tempfile.mkdtemp(prefix="lvr_smoke_")
    sanity_reports = run_sanity_suite(ablation, decay, cfg={})
    sanity_dir = save_sanity_reports(sanity_reports, out_dir)
    assert sanity_reports, "sanity reports should not be empty"
    assert not has_failed_checks(sanity_reports), "fake sanity reports should pass"
    assert "summary_sanity.json" in sorted(os.listdir(sanity_dir))

    run_analysis(ablation, decay, out_dir, metric_results=metric_results)
    produced = sorted(os.listdir(out_dir))
    print(f"  产出 ({out_dir}):")
    for p in produced:
        print("   ", p)
    for need in ["bf1_layerwise_bf3.png", "bf1_layerwise_pf3.png",
                 "bf1_baseline_bf3_curve.png", "bf1_baseline_pf3_curve.png",
                 "cf2_decay_mask.png", "cf2_decay_gaussian_blur.png",
                 "radar_4metric.png", "summary.json", "rank_correlation.json",
                 "metric_results_summary.json", "metric_plots"]:
        assert need in produced, f"缺少 {need}"
    assert os.listdir(os.path.join(out_dir, "metric_plots"))


def main():
    test_corruption()
    test_progress_helpers()
    test_spans()
    test_reductions()
    test_data_field_mapping()
    test_spd_faith_and_maze_loaders()
    test_prepare_maze_planning_fixture()
    test_blink_vsi_loaders_and_prepare_fixtures()
    test_vstar_loader_and_prepare_fixture()
    test_vstar_snapshot_prepare_fixture()
    test_image_resize_metadata()
    test_pf3_corrupt_after_resize()
    test_lvr_json_loader()
    test_lvr_assistant_expansion()
    test_lvr_trace_position_extraction()
    test_lvr_trace_metric_payload()
    test_lvr_trace_required_hard_fail()
    test_trace_recorder_fake_model()
    test_trace_recorder_tensor_capture_and_patch()
    test_preregistration_and_bootstrap()
    test_bf_usage_diagnostics_fixture()
    test_v2_corruptions_and_patch_schema()
    test_pf_a_metric_fixture()
    test_bf_patch_metric_fixture()
    test_bf_swap_and_conf_fixtures()
    test_cf_stage_and_pf_b_fixtures()
    test_v2_sanity_and_validator_fixtures()
    test_w3_latent_sanity_and_validator_fixtures()
    test_w4_latent_step_sweep_fixtures()
    test_w5_w8_tooling_fixtures()
    test_findings_gate_tooling_fixtures()
    test_latent_trace_policy_violation()
    test_adapter_probe_catalog()
    test_monet_preflight_wiring()
    test_monet_latent_gate_wiring()
    test_validate_monet_latent_fixture()
    test_trace_latent_w14_w15_fixtures()
    test_output_accuracy_and_main_matrix_fixtures()
    test_bf1_does_not_cache_gpu_inputs_static()
    test_end_to_end()
    print("\nSMOKE TEST PASSED ✅")


if __name__ == "__main__":
    main()
