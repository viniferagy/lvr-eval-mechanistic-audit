# LVR-Eval Mechanistic Audit

`lvr-eval-mechanistic-audit` 是一个面向 VLM/LVR 的内部机制审计框架。它把模型、数据集、审计 span、metric、sanity check 和 analysis 拆成清晰的模块，让 Qwen baseline 与 LVR 模型可以在同一套 runner 中比较。

核心原则：

- metric 都是 `pipeline/metrics/` 下的平级模块，通过 registry 调度。
- metric 不直接猜 token layout，而是通过 adapter 返回的 `AuditSpans` 定位审计对象。
- Qwen baseline 没有 latent tokens，只使用 `answer_probe_pos` 作为 control query span。
- LVR teacher-forced 审计使用 `<|lvr|>` placeholder positions。
- LVR inference-time 审计应使用 generation trace 中的 continuous latent state，例如 `lvr_mode_switch` 与 `last_position_hidden_state`。
- `<|image_pad|>` 之后的文本不是 LVR latent；旧 post-image text helper 只保留作历史结果调试。

**Architecture**

```
lvr-eval-mechanistic-audit/
├── config.yaml
├── run_all.py
├── launch_sharded.sh
├── merge_and_analyze.py
├── smoke_test.py
├── tools/
│   ├── run_and_hold.sh
│   └── hold_gpu.py
├── docs/
│   └── validation_report.md
└── pipeline/
    ├── adapters/
    │   ├── base.py
    │   ├── qwen_vl.py
    │   ├── lvr_qwen.py
    │   ├── spans.py
    │   └── registry.py
    ├── metrics/
    │   ├── bf3_confidence_progression.py
    │   ├── pf3_attention_distance.py
    │   ├── bf1_latent_ablation.py
    │   ├── bf1_layer_ablation.py
    │   ├── cf2_pf_decay_curve.py
    │   ├── lvr_generation_trace.py
    │   └── registry.py
    ├── sanity/
    ├── data.py
    ├── model_utils.py
    ├── internal_metrics.py
    ├── ablation.py
    ├── degradation.py
    ├── results.py
    └── analysis.py
```

`run_all.py` 负责读取 config、加载 probe set、加载模型、执行选中的 metric、写统一 result envelope、运行 sanity check，并调用 analysis。`launch_sharded.sh` 用于多 GPU 分片运行；`merge_and_analyze.py` 用于合并 sharded 输出并重新生成 sanity/analysis artifact。

**Adapters**

Adapters 是模型语义边界。metric 只依赖 wrapper 与 adapter，不直接解析 prompt token。

`QwenVLAdapter`：

- 适用于 `qwen2_5_vl` / `qwen3_vl` / `auto`。
- 通过 `<|image_pad|>` 定位 image token span。
- 返回 `latent_tokens=None`。
- 返回 `answer_probe_pos` 作为 baseline control query span。

`LVRQwenAdapter`：

- 适用于 `lvr_qwen2_5_vl` / `lvr` / `qwen_lvr`。
- 使用官方 `QwenWithLVR` 与 LVR monkey patch 加载路径。
- 识别 `lvr_start_id`、`lvr_id`、`lvr_latent_end_id`、`lvr_end_id`。
- teacher-forced 模式下通过 `<|lvr|>` token 定位 `lvr_placeholder_tokens`。
- `generate_with_trace()` 提供 inference-time trace 的第一版入口；当前 trace quality 标记为 approximate，后续应直接 instrument 官方 generation loop。

`AuditSpans` 位于 `pipeline/adapters/spans.py`：

- `image_tokens`
- `question_tokens`
- `lvr_placeholder_tokens`
- `latent_tokens`
- `answer_probe_pos`

metric 使用 `spans.preferred_query_span()` 选择 query side：

- LVR latent/placeholder 优先。
- Qwen fallback 到 `answer_probe_pos`。
- 无有效 query span 时直接报错。

**Metrics**

所有 metric 都是 registry 下的同级模块，既可以单独运行，也可以被 analysis 汇总。

`bf3_confidence_progression`

- 类型：internal curve readout。
- 输入：adapter query span。
- 输出：逐层 logit-lens entropy curve。
- 主要标量：`early_to_late_drop`、`final_entropy`、`mean_entropy`。
- 语义：观察 query position 的 confidence sharpening 是否随 decoder layer 推进。

`pf3_attention_distance`

- 类型：internal curve readout。
- 输入：query span 与 image token span。
- 输出：intact image 与 corrupted image 的 query-to-image attention KL curve。
- 主要标量：`mean_kl`、`mid_kl`、`peak_kl`。
- 语义：观察 query representation 对视觉证据扰动的内部注意力敏感度。

`bf1_latent_ablation`

- 类型：sweep。
- 默认行为：targeted span ablation。
- 输入：adapter `preferred_query_span()`。
- 输出：逐层 targeted ablation 后的 BF-3 readout 变化。
- 主要标量：`delta.bf3`。
- 语义：估计各 decoder layer 对审计 query span 的 causal contribution。

`bf1_layer_ablation`

- 类型：sweep。
- 默认关闭。
- 行为：legacy whole-layer ablation。
- 用途：保留旧整层消融对照；不作为默认 latent ablation 解释。

`cf2_pf_decay_curve`

- 类型：sweep。
- 输入：corruption family 与 severity list。
- 输出：severity 轴上的 readout curve。
- 默认 readout：`pf3_attention_distance`。
- 主要特征：`auc`、`char_severity`、`rel_change`。
- 语义：观察内部视觉依赖信号如何随图像退化而变化。

`lvr_generation_trace`

- 类型：trace。
- 输入：LVR adapter 的 `generate_with_trace()`。
- 输出：generated text、LVR token positions、trace quality、trace notes。
- 用途：为 inference-time LVR 审计保存生成轨迹。当前版本是第一版 trace 接口，长期版本应直接记录每一步 `lvr_mode_switch` 和 `last_position_hidden_state`。

**Config**

模型配置示例：

```yaml
models:
  qwen2_5_vl_7b:
    name: "Qwen2.5-VL-7B"
    path: "./models/Qwen/Qwen2___5-VL-7B-Instruct"
    arch: "qwen2_5_vl"
    image_pad_token: "<|image_pad|>"

  lvr_7b:
    name: "LVR-7B"
    path: "./models/LVR-7B"
    arch: "lvr_qwen2_5_vl"
    image_pad_token: "<|image_pad|>"
    lvr_start_token: "<|lvr_start|>"
    lvr_token: "<|lvr|>"
    lvr_latent_end_token: "<|lvr_latent_end|>"
    lvr_end_token: "<|lvr_end|>"
```

审计配置：

```yaml
audit:
  mode: "teacher_forced"
  query_target: "auto"
  lvr_num_tokens: 16
  allow_lvr_fallback_to_answer_probe: false
  lvr_decoding_strategy: "steps"
  lvr_steps: 16
```

LVR 官方 JSON list 数据：

```yaml
data:
  source_type: "lvr_json"
  json_path: "./data/meta_data_lvr_sft_stage1.json"
  image_root: "./data/images"
  max_samples: 50
  skip_missing_images: true
```

`lvr_json` loader 读取官方 LLaVA-style list record，并保留完整 metadata：

- `image`
- `conversations`
- assistant-side `<lvr>`
- optional `bboxes`
- `dataset`

在 `audit.mode=teacher_forced` 时，`LVRQwenAdapter` 会把 assistant 文本中的第一个 `<lvr>` 展开为：

```text
<|lvr_start|><|lvr|>...<|lvr|><|lvr_end|>
```

`audit.lvr_num_tokens` 控制 `<|lvr|>` 的数量。`allow_lvr_fallback_to_answer_probe=false` 时，如果 LVR teacher-forced 输入中没有产生 `<|lvr|>` span，run 会 fail，而不是悄悄退回 answer-probe control。

metric 开关：

```yaml
metrics:
  bf3_confidence_progression:
    enabled: true
  pf3_attention_distance:
    enabled: true
  bf1_latent_ablation:
    enabled: true
  bf1_layer_ablation:
    enabled: false
  cf2_pf_decay_curve:
    enabled: true
  lvr_generation_trace:
    enabled: false
```

`all` 只运行 enabled metric。显式 `--only bf1_layer` 或 `--only lvr_trace` 会运行对应 metric，即使它默认 disabled。

**Run**

安装依赖并先跑 smoke：

```bash
pip install -r requirements.txt
python smoke_test.py
```

常用命令：

```bash
python run_all.py --config config.yaml --models qwen2_5_vl_7b lvr_7b
python run_all.py --config config.yaml --models qwen2_5_vl_7b --only bf1
python run_all.py --config config.yaml --models lvr_7b --only cf2_pf_decay_curve
python run_all.py --config config.yaml --models lvr_7b --only lvr_trace
python run_all.py --config config.yaml --models qwen2_5_vl_7b --no-sanity
```

服务器上需要 GPU 独占时，GPU 命令统一走 wrapper：

```bash
bash tools/run_and_hold.sh 0,1,2,3 launch_sharded.sh
bash tools/run_and_hold.sh 0,1,2,3 ../.venv/bin/python merge_and_analyze.py --dir runs/<run>
```

`tools/hold_gpu.py` 会在命令结束后重新 hold 可见 GPU，并周期性自动扩张 ballast。

**Outputs**

每个 metric 会写统一 envelope：

```text
runs/<run>/metrics/<metric_id>_<model>.json
```

兼容历史 analysis 的 metric 也会写 legacy result：

```text
runs/<run>/bf1_<model>.json
runs/<run>/cf2_<model>.json
```

analysis 输出：

- `summary.json`
- `rank_correlation.json`
- `radar_4metric.png`
- `bf1_layerwise_bf3.png`
- `bf1_baseline_bf3_curve.png`
- `cf2_decay_<family>.png`
- `metric_results_summary.json`
- `metric_plots/*.png`

BF-1/CF-2 payload 会记录 span metadata：

```json
{
  "query_target_kind": "lvr_placeholder_tokens",
  "query_span": [3, 6],
  "image_span": [0, 2],
  "adapter_notes": {}
}
```

聚合结果还会记录：

```json
{
  "n_total": 50,
  "n_success": 48,
  "n_skipped": 2,
  "skip_reasons": {},
  "query_target_counts": {
    "lvr_placeholder_tokens": 48
  }
}
```

**Sanity Checks**

默认运行 sanity，并写入：

```text
runs/<run>/sanity/
```

覆盖内容：

- BF-3 curve length、finite、entropy drop、monotonicity。
- PF-3 KL non-negative、nonzero signal、mid-layer signal。
- BF-1 baseline/layer result/delta 是否存在且非零。
- CF-2 severity axis、finite curve、severity=0 clean baseline、trend。
- span metadata valid rate、Qwen `answer_probe_pos`、LVR `<|lvr|>` placeholder、query/image span 不混淆。

`validation.fail_fast: true` 时 sanity fail 会让 run 以非零退出；默认只 warning。

**Validation**

当前无模型 smoke 覆盖：

- corruption operators。
- severity=0 clean baseline。
- Qwen baseline span semantics。
- LVR teacher-forced span semantics。
- registry metric aliases。
- curve reductions。
- data field mapping。
- LVR JSON list loader。
- assistant-side `<lvr>` expansion。
- sanity reports。
- unified MetricResult 到 analysis artifacts。

运行：

```bash
python -m py_compile run_all.py smoke_test.py merge_and_analyze.py pipeline/*.py pipeline/sanity/*.py pipeline/metrics/*.py pipeline/adapters/*.py
python smoke_test.py
```

完整服务器验证记录见：

```text
docs/validation_report.md
```

**Notes**

- 真实 BF-3/PF-3/BF-1/CF-2 forward 需要 GPU 与本地权重。
- PF-3 使用 eager attention，显存压力较高。
- `data.max_samples` 建议先用小样本 smoke，再扩大。
- 新增模型时优先增加 adapter；新增审计方法时增加 metric module 与 registry entry。
- 新增数据集优先通过 `data.field_map` 配字段，不改 metric 代码。
