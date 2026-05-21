# LVR-Eval :: LVR Mechanistic Audit

VLM 内部机制审计 pipeline。在你已有的两个 **model-internal、逐层** 指标
（BF-3 / PF-3）之上，新增 **BF-1（targeted span ablation）** 与 **CF-2（PF 衰减曲线）**，
并把它们联动成 radar + 相关性分析。

模型矩阵：`qwen2_5_vl_7b`、`lvr_7b`、`qwen2_5_vl_3b`。
旧 tag `M0`、`M2`、`M0_small` 仍会通过 `model_aliases` 自动映射。

---

## Correct LVR Audit Semantics

- Qwen baseline has no latent tokens; it uses `answer_probe_pos` as a control query span.
- LVR teacher-forced audit uses `<|lvr|>` placeholder positions.
- LVR inference-time audit uses generation trace: `lvr_mode_switch` and `last_position_hidden_state`; the current trace metric stores an approximate first-pass trace from generated token ids and marks it as such.
- The legacy post-image text span is not a latent span and is retained only as `get_post_image_text_span()` for debugging old results.

---

## BF-3 / PF-3 是怎么接进来的

你上传的 `bf3_today.py` / `pf3_today.py` 的**核心**已移植进
`pipeline/internal_metrics.py`（去掉 CLI 和数据加载，重构成作用于 `VLMWrapper`
的可复用函数）。现在正式入口在 `pipeline/metrics/` registry，`internal_metrics.py`
保留为兼容 facade 和底层实现，逐项对应如下：

| 原文件 | 移植后 | 说明 |
|---|---|---|
| `logit_lens_entropy` | `internal_metrics.logit_lens_entropy` | 用 wrapper 缓存的 `final_norm`/`lm_head`（探测 `model.model.norm` 等多路径） |
| `compute_bf3_curve` | `internal_metrics.bf3_curve` | adapter query span 末端的逐层 entropy curve；`output_hidden_states=True` |
| image/query token span | `adapter.get_spans()` | Qwen=`answer_probe_pos` control；LVR teacher-forced=`<|lvr|>` placeholder |
| `random_patch_mask`/`patch_shuffle`/`corrupt_image` | 同名 | 另加 `severity` 形参供 CF-2 连续扫描 |
| `kl_divergence_safe`/`compute_attention_distance` | `_kl_safe`/`pf3_curve` | intact vs corrupted 的 query_span→image attention KL；eager attention |

关键设计：BF-3/PF-3 的 forward 都是**普通 forward**。BF-1 把 ablation hook
挂上后再调用它们，算出的 curve 自动反映“消融后”的内部状态 —— 这就是 BF-1 与
BF-3/PF-3 的联动机制。模型加载时强制 `attn_implementation="eager"`（PF-3 取
attention 必须），并缓存 `final_norm`/`lm_head`/`image_pad_id`。

---

## BF-1 与 CF-2 的读出方式

**BF-1 Targeted Span Ablation** — 默认 `bf1_latent_ablation` 先由 adapter 定位
`preferred_query_span()`，再只在该 span 上逐层挂 hook（`identity`/`zero`/`mean`/`noise`），
第一版 targeted readout 重算 BF-3 curve：
- `Δbf3 = baseline.early_to_late_drop − ablated.early_to_late_drop`（该层对 confidence sharpening 的贡献）

正值越大 = 该层越关键。同时保留每个消融配置下重算的完整 curve。旧整层 sweep
保留为 `bf1_layer_ablation` / `--only bf1_layer`，不再作为默认 latent ablation。

**CF-2 PF Decay Curve** — 把 PF-3 的离散 corruption 推广到 severity 连续轴：
- `readout: pf3`（默认，也可写 `pf3_attention_distance`）：x=severity，y=PF-3 `mean_kl`（intact vs corrupted-at-s）。severity↑ 通常 KL↑。
- `readout: bf3`（也可写 `bf3_confidence_progression`）：把退化图喂给 BF-3，y=final-layer entropy。
拟合 `auc` / `char_severity`（半程 severity）/ `rel_change`。

**联动 radar**：BF-1 criticality（层 Δ 的最大绝对值）/ BF-3 sharpening（baseline）/
PF-3 modality-dep（baseline）/ CF-2 AUC（各 family 均值）；外加关键层
Δbf3~Δpf3 的 Spearman。

---

## 目录

```
lvr_eval/
├── config.yaml                # 模型路径 / 数据 / PF-3 / BF-1 / CF-2 设置
├── run_all.py                 # registry-driven 端到端入口
├── launch_sharded.sh          # 4×4090：模型×阶段 4 个 job
├── merge_and_analyze.py       # 收集 sharded 产出统一出图
├── smoke_test.py              # 无真实模型验证非模型逻辑、span 语义、severity=0
├── tools/                     # GPU reservation wrapper / holder
│   ├── run_and_hold.sh
│   └── hold_gpu.py
├── docs/
│   └── validation_report.md
└── pipeline/
    ├── metrics/               # ★ 平级指标模块 + registry
    │   ├── bf3_confidence_progression.py
    │   ├── pf3_attention_distance.py
    │   ├── bf1_latent_ablation.py
    │   ├── bf1_layer_ablation.py
    │   ├── lvr_generation_trace.py
    │   └── cf2_pf_decay_curve.py
    ├── adapters/              # VLMAdapter：模型加载/输入构造/generate 扩展点
    ├── internal_metrics.py    # BF-3 / PF-3 底层兼容 facade + corruption
    ├── model_utils.py         # 加载(eager) + layer/norm/head/image_pad 探测 + forward
    ├── ablation.py            # BF-1：hook + 逐层重算 BF-3/PF-3
    ├── degradation.py         # CF-2：severity 扫描 + 曲线拟合
    ├── sanity/                # BF-3/PF-3/BF-1/CF-2 sanity reports
    ├── data.py                # probe 数据集（jsonl / HF）
    └── analysis.py            # 折线/热力图/decay/radar/spearman
```

---

## 上手

```bash
pip install -r requirements.txt
python smoke_test.py                      # 先验证接线（不需要 GPU）
```

改 `config.yaml`：
- `models.*.path`：`qwen2_5_vl_7b` / `lvr_7b` / `qwen2_5_vl_3b` 权重路径；LVR 使用 `arch: lvr_qwen2_5_vl`
- `data`：指向你的 audit 集（jsonl 或 HF；**PF-3 不能用 synthetic 图**）
- `data.field_map`：配置数据集字段映射，支持 `image` / `question` / `answer` / `id`，也支持 `meta.id` 点路径和字段 fallback 列表
- 如 LVR 的 image/LVR special token 不同，改 `models.lvr_7b.image_pad_token` 与 `lvr_*_token`
- `metrics.*.enabled`：统一指标开关；旧的 `bf1.enabled` / `cf2.enabled` 仍作为 fallback 兼容一轮

```bash
python run_all.py --config config.yaml --models qwen2_5_vl_7b lvr_7b
python run_all.py --config config.yaml --models qwen2_5_vl_7b lvr_7b --only bf1
python run_all.py --config config.yaml --models lvr_7b --only cf2_pf_decay_curve
python run_all.py --config config.yaml --models qwen2_5_vl_7b --no-sanity
bash launch_sharded.sh
```

服务器上需要 GPU 独占时，所有 GPU 命令通过 wrapper 入口运行：

```bash
bash tools/run_and_hold.sh 0,1,2,3 launch_sharded.sh
bash tools/run_and_hold.sh 0,1,2,3 ../.venv/bin/python merge_and_analyze.py --dir runs/<run>
```

`tools/hold_gpu.py` 会在命令结束后重新占住可见 GPU，并周期性自动扩张 ballast，
因此其他进程释放显存后会被 holder 在后续扩张周期内占用。

每个 metric 会同时写旧格式结果（如 `bf1_lvr_7b.json`）和统一 envelope：
`runs/<run_name>/metrics/<metric_id>_<model>.json`。
analysis 会优先理解这些统一 envelope，并额外写出:
- `metric_results_summary.json`
- `metric_plots/*.png`（按 `metric_id/model` 组织的通用曲线/标量图）

每次 run 默认会在 `runs/<run_name>/sanity/` 下写出:
- `bf1_latent_ablation_*_sanity.json`
- `bf3_confidence_progression_*_sanity.json`（来自 BF-1 baseline）
- `pf3_attention_distance_*_sanity.json`（来自 BF-1 baseline）
- `cf2_pf_decay_curve_*_sanity.json`
- `summary_sanity.json`

---

## 验证 / 注意

- `smoke_test.py` 覆盖：corruption 算子、curve 归约、AUC/char_severity（升降型）、
  data field_map、sanity report，以及统一 MetricResult → analysis 出全套图。**已通过 ✅**
- BF-3/PF-3 的真实 forward 需要 GPU + 权重，无法在 CPU 验证，但核心逻辑是从你的
  原脚本移植并修正为 adapter span 语义，不再把 `<|image_pad|>` 后文本当成 LVR latent。
- 内部指标走**单样本 forward**（hidden_states / attentions），不走 batch generate；
  PF-3 + eager attention 偏重，`data.max_samples` 默认 30，按需调。
- 新增模型时优先在 `pipeline/adapters/` 注册新的 adapter；metric 代码只依赖 `VLMWrapper.build_inputs()` / `generate()`。
- 每个 BF-1/CF-2 payload 会记录 `query_target_kind`、`query_span`、`image_span` 和 `adapter_notes`，用于确认审计目标不是 post-image text。
