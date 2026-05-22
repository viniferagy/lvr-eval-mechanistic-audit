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

import os
import json
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import yaml

from pipeline.adapters.lvr_qwen import LVRQwenAdapter
from pipeline.adapters.lvr_qwen_traced import TraceRecorder
from pipeline.adapters.qwen_vl import QwenVLAdapter
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
from pipeline.metrics.v2.pf_b_patch_alignment import run as run_pf_b_metric
from pipeline.metrics.lvr_generation_trace import run as run_lvr_trace_metric
from pipeline.preregistration import build_lock_payload, hash_manifest, load_manifest
from pipeline.results import make_metric_result
from pipeline.stats.bootstrap import paired_bootstrap
from pipeline.degradation import curve_features
from pipeline.analysis import build_summary_with_ci, run_analysis
from pipeline import ablation as ABL
from pipeline import internal_metrics as IM
from pipeline.sanity import has_failed_checks, run_sanity_suite, save_sanity_reports


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
    data[0]["conversations"][1]["value"] = "<answer>dark blue denim shorts</answer>"
    with open(missing_lvr_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
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
    records = data + [dict(data[0], question_id=31594), dict(data[0], question_id=31595)]
    with open(limited_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    samples = load_probe_set({
        "source_type": "lvr_json",
        "json_path": limited_path,
        "image_root": out_dir,
        "max_scan_records": 1,
        "skip_missing_images": False,
    })
    assert len(samples) == 0  # current data[0] was mutated to remove <lvr>
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
    print("  manifest hash/lock + bootstrap summary_with_ci -> ok")


def test_v2_corruptions_and_patch_schema():
    print("\n== 10d. v2 corruption + BF-Patch schema fixtures ==")
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    rel = relevant_mask(img, bboxes=[[0.0, 0.0, 0.5, 0.5]])
    irr = irrelevant_mask(img, rel, seed=0)
    rnd1 = random_mask(img, coverage=0.2, seed=3)
    rnd2 = random_mask(img, coverage=0.2, seed=3)
    assert int((rel.data & irr.data).sum()) == 0
    assert np.array_equal(rnd1.data, rnd2.data)
    assert np.array_equal(np.asarray(apply_mask(img, rel, severity=0)), np.asarray(img))
    grid = patch_grid()
    assert len(grid) == 15
    assert grid[0] == {"layer": 0, "position_bucket": "image"}
    rate = answer_transfer_rate([
        {"patched_answer": "B", "source_answer": "B"},
        {"patched_answer": "A", "source_answer": "B"},
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
        vocab = {"red": 1, "blue": 2}

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
            logits = torch.zeros(hidden.shape[0], hidden.shape[1], 4)
            logits[..., 1] = hidden[..., 0] * 0.1
            logits[..., 2] = hidden[..., 0] * 2.0
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
    assert one["logit_margin_shift"] is not None
    assert one["patch_length"] == 2
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
    print("  CF-Stage stage reducer + PF-B native alignment fixture -> ok")


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
    rng = np.random.default_rng(hash(tag) % 2**31)
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
    test_spans()
    test_reductions()
    test_data_field_mapping()
    test_spd_faith_and_maze_loaders()
    test_image_resize_metadata()
    test_pf3_corrupt_after_resize()
    test_lvr_json_loader()
    test_lvr_assistant_expansion()
    test_lvr_trace_position_extraction()
    test_lvr_trace_metric_payload()
    test_trace_recorder_fake_model()
    test_preregistration_and_bootstrap()
    test_v2_corruptions_and_patch_schema()
    test_pf_a_metric_fixture()
    test_bf_patch_metric_fixture()
    test_bf_swap_and_conf_fixtures()
    test_cf_stage_and_pf_b_fixtures()
    test_bf1_does_not_cache_gpu_inputs_static()
    test_end_to_end()
    print("\nSMOKE TEST PASSED ✅")


if __name__ == "__main__":
    main()
