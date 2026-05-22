"""
pipeline/internal_metrics.py
============================
从 bf3_today.py / pf3_today.py 抽出的**核心**(去掉 CLI / 数据加载),
重构成可被 BF-1 / CF-2 复用的、作用于 VLMWrapper 的函数。

两个指标都是 **model-internal、逐层** 的:

BF-3 (Confidence Progression)
    在 adapter 声明的 query span 末端, 对每层 hidden state 做 logit-lens, 算 entropy。
    返回 curve[n_layers]。需要 output_hidden_states。

PF-3 (Corrupted-vs-Intact Attention Distance)
    intact 与 corrupted 两次 forward, 取 query_span→image 的 attention 分布,
    逐层算 KL(intact || corrupted)。返回 curve[n_layers]。
    需要 eager attention + output_attentions。

关键: 这些 forward 是普通 forward —— 若外部已用 ablation hook 包住
(见 ablation.py 的 ablate_layer 上下文), 这里算出的 curve 自动反映“消融后”的
内部状态。这正是 BF-1 与 BF-3/PF-3 联动的机制。
"""
from __future__ import annotations

import logging
import random
from dataclasses import replace
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

try:                                  # torch 仅 BF-3/PF-3 forward 路径需要
    import torch
    _no_grad = torch.no_grad
except ImportError:                   # 无 torch 时(如 smoke_test) 仍可用 corruption/reduce
    torch = None

    def _no_grad():
        def deco(fn):
            return fn
        return deco

logger = logging.getLogger("lvr_eval.internal")


# --------------------------------------------------------------------------- #
#  audit span helpers
# --------------------------------------------------------------------------- #
def get_post_image_text_span(input_ids: torch.Tensor, image_pad_id: int) -> tuple[int, int]:
    """
    Legacy helper. This is NOT a latent span.

    It returns the post-image text span and is retained only for debugging old
    results that treated image_pad[-1] + 1 : seq_end as a latent region.
    """
    img_pos = (input_ids[0] == image_pad_id).nonzero(as_tuple=True)[0]
    if len(img_pos) == 0:
        raise ValueError("输入里没有 image token —— 检查 image_pad token 是否正确")
    return img_pos[-1].item() + 1, input_ids.shape[1]


def get_image_token_indices(input_ids: torch.Tensor, image_pad_id: int) -> torch.Tensor:
    return (input_ids[0] == image_pad_id).nonzero(as_tuple=True)[0]


def _query_span_from_adapter(wrapper, inputs, outputs=None):
    spans = wrapper.adapter.get_spans(wrapper, inputs, outputs)
    query = spans.preferred_query_span()
    if query.length <= 0:
        raise ValueError(f"invalid query span from adapter: {query}")
    if spans.image_tokens.length <= 0:
        raise ValueError(f"invalid image span from adapter: {spans.image_tokens}")
    return spans, query


def _span_payload(spans, query_span) -> dict:
    from .adapters.spans import spans_to_metadata

    return spans_to_metadata(spans, query_span)


def _build_inputs(wrapper, image=None, question: str | None = None, sample=None):
    if sample is not None and hasattr(wrapper, "build_inputs_from_sample"):
        return wrapper.build_inputs_from_sample(sample)
    return wrapper.build_inputs(image, question)


# --------------------------------------------------------------------------- #
#  BF-3 : logit-lens entropy
# --------------------------------------------------------------------------- #
def logit_lens_entropy(wrapper, hidden_state: torch.Tensor) -> np.ndarray:
    """hidden_state [D] or [B,D] -> entropy(nats). 用 wrapper 缓存的 norm/head。"""
    if hidden_state.dim() == 1:
        hidden_state = hidden_state.unsqueeze(0)
    normed = wrapper.final_norm(hidden_state) if wrapper.final_norm is not None else hidden_state
    logits = wrapper.lm_head(normed)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.detach().cpu().numpy()


def _to_model_device(tensor: torch.Tensor, wrapper) -> torch.Tensor:
    try:
        return tensor.to(wrapper.lm_head.weight.device)
    except Exception:  # noqa: BLE001
        return tensor.to(wrapper.device)


@_no_grad()
def bf3_curve_from_inputs_low_memory(wrapper, inputs, query_span) -> np.ndarray:
    """
    Compute BF-3 without materializing full output_hidden_states.

    A forward hook stores only the query-position hidden vector from each decoder
    layer on CPU. This is much cheaper for LVR teacher-forced prompts, where
    output_hidden_states=True can retain many GiB of [layer, seq, dim] tensors.
    """
    last_pos = query_span.end - 1
    captured: list[torch.Tensor | None] = [None] * wrapper.n_layers
    handles = []

    def make_hook(idx: int):
        def hook(_module, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[idx] = h[0, last_pos, :].detach().cpu()
            return output
        return hook

    for idx, layer in enumerate(wrapper.layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))

    try:
        wrapper.model(
            **inputs,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    ents = []
    for idx, h_cpu in enumerate(captured):
        if h_cpu is None:
            raise RuntimeError(f"BF-3 hook did not capture layer {idx}")
        h = _to_model_device(h_cpu, wrapper)
        ents.append(float(logit_lens_entropy(wrapper, h)[0]))
        del h
    return np.asarray(ents)


@_no_grad()
def bf3_curve_from_inputs(wrapper, inputs, query_span=None, outputs=None) -> np.ndarray:
    """Compute BF-3 from prebuilt inputs and an adapter-declared query span."""
    if query_span is None:
        spans, query_span = _query_span_from_adapter(wrapper, inputs, outputs)
    if outputs is None:
        return bf3_curve_from_inputs_low_memory(wrapper, inputs, query_span)

    out = outputs
    last_pos = query_span.end - 1
    ents = []
    for li in range(1, len(out.hidden_states)):
        h = out.hidden_states[li][0, last_pos, :]
        ents.append(float(logit_lens_entropy(wrapper, h)[0]))
    return np.asarray(ents)


@_no_grad()
def bf3_curve_with_meta(wrapper, image: Image.Image, question: str) -> dict:
    """Single-sample BF-3 curve plus span metadata."""
    inputs = _build_inputs(wrapper, image=image, question=question)
    spans, query_span = _query_span_from_adapter(wrapper, inputs)
    return {
        "curve": bf3_curve_from_inputs(wrapper, inputs, query_span),
        **_span_payload(spans, query_span),
    }


@_no_grad()
def bf3_curve_with_meta_from_sample(wrapper, sample) -> dict:
    """Single-sample BF-3 using full ProbeSample metadata."""
    inputs = _build_inputs(wrapper, sample=sample)
    spans, query_span = _query_span_from_adapter(wrapper, inputs)
    return {
        "curve": bf3_curve_from_inputs(wrapper, inputs, query_span),
        **_span_payload(spans, query_span),
    }


@_no_grad()
def bf3_curve(wrapper, image: Image.Image, question: str) -> np.ndarray:
    """单样本逐层 entropy curve [n_layers]。普通 forward (会受外部 ablation hook 影响)。"""
    return bf3_curve_with_meta(wrapper, image, question)["curve"]


# --------------------------------------------------------------------------- #
#  图像 corruption (PF-3 用; CF-2 在此之上做 severity 参数化)
# --------------------------------------------------------------------------- #
def random_patch_mask(image: Image.Image, ratio=0.5, grid_size=8,
                      seed: Optional[int] = None) -> Image.Image:
    rng = random.Random(seed) if seed is not None else random
    arr = np.array(image.convert("RGB")).copy()
    h, w = arr.shape[:2]
    ph, pw = h // grid_size, w // grid_size
    n = grid_size * grid_size
    for idx in rng.sample(range(n), int(n * ratio)):
        r, c = idx // grid_size, idx % grid_size
        arr[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw] = 0
    return Image.fromarray(arr)


def patch_shuffle(image: Image.Image, grid_size=8,
                  seed: Optional[int] = None) -> Image.Image:
    rng = random.Random(seed) if seed is not None else random
    arr = np.array(image.convert("RGB")).copy()
    h, w = arr.shape[:2]
    ph, pw = h // grid_size, w // grid_size
    patches = [arr[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw].copy()
               for r in range(grid_size) for c in range(grid_size)]
    rng.shuffle(patches)
    new = arr.copy()
    for idx, p in enumerate(patches):
        r, c = idx // grid_size, idx % grid_size
        new[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw] = p
    return Image.fromarray(new)


def corrupt_image(image: Image.Image, mode: str,
                  seed: Optional[int] = None,
                  severity: Optional[float] = None) -> Image.Image:
    """
    统一 corruption 入口。
    - 离散模式(PF-3 原版): mask_50pct / mask_80pct / gaussian_blur / patch_shuffle
    - 连续模式(CF-2 用): 传 severity 覆盖强度
        mask:   severity = mask ratio (0..1)
        gaussian_blur: severity = blur radius
        patch_shuffle: severity 暂不参数化(固定 grid)
    """
    if mode.startswith("mask"):
        ratio = severity if severity is not None else (0.8 if "80" in mode else 0.5)
        if float(ratio) <= 0:
            return image.convert("RGB")
        return random_patch_mask(image, ratio=ratio, seed=seed)
    if mode == "gaussian_blur":
        radius = severity if severity is not None else 10
        if float(radius) <= 0:
            return image.convert("RGB")
        return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))
    if mode == "patch_shuffle":
        return patch_shuffle(image, grid_size=8, seed=seed)
    raise ValueError(f"Unknown corruption mode: {mode}")


# --------------------------------------------------------------------------- #
#  PF-3 : intact vs corrupted attention KL
# --------------------------------------------------------------------------- #
def _kl_safe(p: torch.Tensor, q: torch.Tensor, eps=1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


@_no_grad()
def _forward_attn(wrapper, image: Image.Image, question: str) -> dict:
    inputs = _build_inputs(wrapper, image=image, question=question)
    out = wrapper.model(**inputs, output_hidden_states=False,
                        output_attentions=True, return_dict=True)
    spans = wrapper.adapter.get_spans(wrapper, inputs, out)
    return {"attentions": out.attentions, "input_ids": inputs["input_ids"],
            "inputs": inputs, "spans": spans, "seq_len": inputs["input_ids"].shape[1]}


@_no_grad()
def _forward_attn_from_sample(wrapper, sample) -> dict:
    inputs = _build_inputs(wrapper, sample=sample)
    out = wrapper.model(**inputs, output_hidden_states=False,
                        output_attentions=True, return_dict=True)
    spans = wrapper.adapter.get_spans(wrapper, inputs, out)
    return {"attentions": out.attentions, "input_ids": inputs["input_ids"],
            "inputs": inputs, "spans": spans, "seq_len": inputs["input_ids"].shape[1]}


@_no_grad()
def pf3_curve_with_meta(wrapper, image: Image.Image, question: str,
                        corruption_mode: str = "mask_50pct",
                        num_seeds: int = 3,
                        severity: Optional[float] = None) -> dict:
    """
    单样本逐层 attention-distance curve [n_layers] + intact token 数。
    token 数不一致的 corrupted seed 会被跳过 (Qwen 动态分辨率)。
    severity 仅 CF-2 用 (连续 corruption 强度)。
    """
    intact = _forward_attn(wrapper, image, question)
    intact_T = intact["seq_len"]
    intact_attns = intact["attentions"]
    spans = intact["spans"]
    query_span = spans.preferred_query_span()
    image_span = spans.image_tokens
    image_idx = torch.arange(
        image_span.start,
        image_span.end,
        device=intact["input_ids"].device,
    )

    n_layers = len(intact_attns)
    per_seed = []
    skip_reasons: dict[str, int] = {}
    # patch_shuffle / blur 在固定 severity 下其实是确定性(blur)或仅依赖 seed(shuffle)
    for seed in range(num_seeds):
        corr_img = corrupt_image(image, corruption_mode, seed=seed, severity=severity)
        corr = _forward_attn(wrapper, corr_img, question)
        if corr["seq_len"] != intact_T:        # token 数必须一致
            skip_reasons["seq_len_mismatch"] = skip_reasons.get("seq_len_mismatch", 0) + 1
            continue
        corr_attns = corr["attentions"]
        per_layer = []
        for li in range(n_layers):
            Ai = intact_attns[li][0][:, query_span.start:query_span.end, :][:, :, image_idx]
            Ac = corr_attns[li][0][:, query_span.start:query_span.end, :][:, :, image_idx]
            Ai = Ai / (Ai.sum(dim=-1, keepdim=True) + 1e-8)
            Ac = Ac / (Ac.sum(dim=-1, keepdim=True) + 1e-8)
            per_layer.append(_kl_safe(Ai.float(), Ac.float()).mean().item())
        per_seed.append(per_layer)

    curve = np.asarray(per_seed).mean(axis=0) if per_seed else None
    return {
        "curve": curve,
        "seq_len": intact_T,
        "n_total": int(num_seeds),
        "n_success": len(per_seed),
        "n_skipped": int(num_seeds - len(per_seed)),
        "skip_reasons": skip_reasons,
        **_span_payload(spans, query_span),
    }


@_no_grad()
def pf3_curve_with_meta_from_sample(wrapper, sample,
                                    corruption_mode: str = "mask_50pct",
                                    num_seeds: int = 3,
                                    severity: Optional[float] = None) -> dict:
    """
    Sample-level PF-3 path. Required for LVR teacher-forced inputs because the
    assistant-side <lvr> metadata lives on ProbeSample.
    """
    intact = _forward_attn_from_sample(wrapper, sample)
    intact_T = intact["seq_len"]
    intact_attns = intact["attentions"]
    spans = intact["spans"]
    query_span = spans.preferred_query_span()
    image_span = spans.image_tokens
    image_idx = torch.arange(
        image_span.start,
        image_span.end,
        device=intact["input_ids"].device,
    )

    n_layers = len(intact_attns)
    per_seed = []
    skip_reasons: dict[str, int] = {}
    for seed in range(num_seeds):
        corr_img = corrupt_image(sample.image, corruption_mode, seed=seed, severity=severity)
        corr_sample = replace(sample, image=corr_img)
        corr = _forward_attn_from_sample(wrapper, corr_sample)
        if corr["seq_len"] != intact_T:
            skip_reasons["seq_len_mismatch"] = skip_reasons.get("seq_len_mismatch", 0) + 1
            continue
        corr_attns = corr["attentions"]
        per_layer = []
        for li in range(n_layers):
            Ai = intact_attns[li][0][:, query_span.start:query_span.end, :][:, :, image_idx]
            Ac = corr_attns[li][0][:, query_span.start:query_span.end, :][:, :, image_idx]
            Ai = Ai / (Ai.sum(dim=-1, keepdim=True) + 1e-8)
            Ac = Ac / (Ac.sum(dim=-1, keepdim=True) + 1e-8)
            per_layer.append(_kl_safe(Ai.float(), Ac.float()).mean().item())
        per_seed.append(per_layer)

    curve = np.asarray(per_seed).mean(axis=0) if per_seed else None
    return {
        "curve": curve,
        "seq_len": intact_T,
        "n_total": int(num_seeds),
        "n_success": len(per_seed),
        "n_skipped": int(num_seeds - len(per_seed)),
        "skip_reasons": skip_reasons,
        **_span_payload(spans, query_span),
    }


@_no_grad()
def pf3_curve(wrapper, image: Image.Image, question: str,
             corruption_mode: str = "mask_50pct",
             num_seeds: int = 3,
             severity: Optional[float] = None) -> tuple[Optional[np.ndarray], int]:
    """
    单样本逐层 attention-distance curve [n_layers] + intact token 数。
    token 数不一致的 corrupted seed 会被跳过 (Qwen 动态分辨率)。
    severity 仅 CF-2 用 (连续 corruption 强度)。
    """
    result = pf3_curve_with_meta(
        wrapper,
        image,
        question,
        corruption_mode=corruption_mode,
        num_seeds=num_seeds,
        severity=severity,
    )
    return result["curve"], result["seq_len"]


# --------------------------------------------------------------------------- #
#  curve -> scalar 归约 (供 BF-1 Δ / CF-2 decay 轴用)
# --------------------------------------------------------------------------- #
def bf3_reduce(curve: np.ndarray) -> dict[str, float]:
    """BF-3 curve 的标量归约。"""
    L = len(curve)
    early = float(curve[:4].mean())
    late = float(curve[-4:].mean())
    return {
        "early_to_late_drop": early - late,   # 主标量: confidence sharpening
        "final_entropy": late,
        "mean_entropy": float(curve.mean()),
    }


def pf3_reduce(curve: np.ndarray) -> dict[str, float]:
    """PF-3 curve 的标量归约。"""
    L = len(curve)
    mid = curve[L // 4: 3 * L // 4]
    return {
        "mean_kl": float(curve.mean()),       # 主标量: modality dependence
        "mid_kl": float(mid.mean()),
        "peak_kl": float(curve.max()),
    }
