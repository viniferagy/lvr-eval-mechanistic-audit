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


def prepare_image_for_audit(wrapper, image):
    """Adapter-aware image preprocessing plus metadata."""
    adapter = getattr(wrapper, "adapter", None)
    if adapter is not None and hasattr(adapter, "prepare_image_for_audit"):
        return adapter.prepare_image_for_audit(wrapper, image)
    if isinstance(image, Image.Image):
        img = image.convert("RGB")
        w, h = img.size
        return img, {
            "image_original_size": [int(w), int(h)],
            "image_processed_size": [int(w), int(h)],
            "image_resize_applied": False,
            "max_image_side": None,
            "max_image_pixels": None,
        }
    return image, {
        "image_original_size": None,
        "image_processed_size": None,
        "image_resize_applied": False,
        "max_image_side": None,
        "max_image_pixels": None,
    }


def _with_image_preprocess(payload: dict, image_meta: dict | None) -> dict:
    if image_meta is not None:
        payload["image_preprocess"] = image_meta
    return payload


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
        try:
            wrapper.model(
                **inputs,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
                use_cache=False,
            )
        except TypeError:
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
    image, image_meta = prepare_image_for_audit(wrapper, image)
    inputs = _build_inputs(wrapper, image=image, question=question)
    spans, query_span = _query_span_from_adapter(wrapper, inputs)
    return _with_image_preprocess({
        "curve": bf3_curve_from_inputs(wrapper, inputs, query_span),
        **_span_payload(spans, query_span),
    }, image_meta)


@_no_grad()
def bf3_curve_with_meta_from_sample(wrapper, sample) -> dict:
    """Single-sample BF-3 using full ProbeSample metadata."""
    image, image_meta = prepare_image_for_audit(wrapper, sample.image)
    sample = replace(sample, image=image)
    inputs = _build_inputs(wrapper, sample=sample)
    spans, query_span = _query_span_from_adapter(wrapper, inputs)
    return _with_image_preprocess({
        "curve": bf3_curve_from_inputs(wrapper, inputs, query_span),
        **_span_payload(spans, query_span),
    }, image_meta)


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


def _query_image_attn_kl_curve(intact: dict, corrupted: dict) -> tuple[Optional[np.ndarray], dict[str, int]]:
    skip_reasons: dict[str, int] = {}
    if corrupted["seq_len"] != intact["seq_len"]:
        skip_reasons["seq_len_mismatch"] = 1
        return None, skip_reasons

    intact_attns = intact["query_image_attn"]
    corrupted_attns = corrupted["query_image_attn"]
    if len(corrupted_attns) != len(intact_attns):
        skip_reasons["layer_count_mismatch"] = 1
        return None, skip_reasons

    per_layer = []
    for li, (Ai, Ac) in enumerate(zip(intact_attns, corrupted_attns)):
        if tuple(Ai.shape) != tuple(Ac.shape):
            skip_reasons[f"attention_shape_mismatch_layer_{li}"] = 1
            return None, skip_reasons
        Ai = Ai / (Ai.sum(dim=-1, keepdim=True) + 1e-8)
        Ac = Ac / (Ac.sum(dim=-1, keepdim=True) + 1e-8)
        per_layer.append(_kl_safe(Ai.float(), Ac.float()).mean().item())
    return np.asarray(per_layer, dtype=float), skip_reasons


@_no_grad()
def _forward_query_image_attn_from_inputs(wrapper, inputs) -> dict:
    """
    Run one eager-attention forward while retaining only query->image slices.

    Returning all layer attentions keeps O(layers * seq_len^2) tensors alive.
    A forward hook captures the small [heads, query, image] slice per layer on
    CPU, then replaces the layer's attention output with a tiny tensor so the
    model output does not accumulate full attention maps.
    """
    spans = wrapper.adapter.get_spans(wrapper, inputs, None)
    query_span = spans.preferred_query_span()
    image_span = spans.image_tokens
    captured: list[torch.Tensor | None] = [None] * wrapper.n_layers
    handles = []

    def make_hook(idx: int):
        def hook(_module, _args, output):
            if not isinstance(output, tuple) or len(output) < 2:
                return output
            attn = output[1]
            if not torch.is_tensor(attn) or attn.dim() < 4:
                return output
            captured[idx] = (
                attn[0, :, query_span.start:query_span.end, image_span.start:image_span.end]
                .detach()
                .float()
                .cpu()
            )
            out = list(output)
            out[1] = attn.new_empty((0,))
            return tuple(out)
        return hook

    for idx, layer in enumerate(wrapper.layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))

    try:
        try:
            wrapper.model(
                **inputs,
                output_hidden_states=False,
                output_attentions=True,
                return_dict=True,
                use_cache=False,
            )
        except TypeError:
            wrapper.model(
                **inputs,
                output_hidden_states=False,
                output_attentions=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    missing = [idx for idx, value in enumerate(captured) if value is None]
    if missing:
        raise RuntimeError(f"PF-3 hook did not capture attention for layers: {missing[:5]}")
    return {
        "query_image_attn": captured,
        "input_ids": inputs["input_ids"],
        "inputs": inputs,
        "spans": spans,
        "seq_len": inputs["input_ids"].shape[1],
    }


@_no_grad()
def _forward_attn(wrapper, image: Image.Image, question: str) -> dict:
    inputs = _build_inputs(wrapper, image=image, question=question)
    return _forward_query_image_attn_from_inputs(wrapper, inputs)


@_no_grad()
def _forward_attn_from_sample(wrapper, sample) -> dict:
    inputs = _build_inputs(wrapper, sample=sample)
    return _forward_query_image_attn_from_inputs(wrapper, inputs)


@_no_grad()
def query_image_attention_kl_with_meta_from_sample(
    wrapper,
    sample,
    corrupted_image: Image.Image,
) -> dict:
    """
    Compute query->image attention KL between a clean sample and one explicitly
    supplied corrupted image. This is the region-targeted counterpart to PF-3's
    random corruption loop and is used by PF-A.
    """
    image, image_meta = prepare_image_for_audit(wrapper, sample.image)
    clean_sample = replace(sample, image=image)
    intact = _forward_attn_from_sample(wrapper, clean_sample)
    spans = intact["spans"]
    query_span = spans.preferred_query_span()

    corr_image, corr_image_meta = prepare_image_for_audit(wrapper, corrupted_image)
    corr_sample = replace(clean_sample, image=corr_image)
    corrupted = _forward_attn_from_sample(wrapper, corr_sample)
    curve, skip_reasons = _query_image_attn_kl_curve(intact, corrupted)

    return _with_image_preprocess({
        "curve": curve,
        "seq_len": intact["seq_len"],
        "n_total": 1,
        "n_success": 1 if curve is not None else 0,
        "n_skipped": 0 if curve is not None else 1,
        "skip_reasons": skip_reasons,
        "corrupted_image_preprocess": corr_image_meta,
        **_span_payload(spans, query_span),
    }, image_meta)


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
    image, image_meta = prepare_image_for_audit(wrapper, image)
    intact = _forward_attn(wrapper, image, question)
    intact_T = intact["seq_len"]
    intact_attns = intact["query_image_attn"]
    spans = intact["spans"]
    query_span = spans.preferred_query_span()
    image_span = spans.image_tokens

    n_layers = len(intact_attns)
    per_seed = []
    skip_reasons: dict[str, int] = {}
    # patch_shuffle / blur 在固定 severity 下其实是确定性(blur)或仅依赖 seed(shuffle)
    for seed in range(num_seeds):
        corr_img = corrupt_image(image, corruption_mode, seed=seed, severity=severity)
        corr = _forward_attn(wrapper, corr_img, question)
        curve_values, seed_skip = _query_image_attn_kl_curve(intact, corr)
        if curve_values is None:
            for reason, count in seed_skip.items():
                skip_reasons[reason] = skip_reasons.get(reason, 0) + count
            continue
        per_seed.append(curve_values)

    curve = np.asarray(per_seed).mean(axis=0) if per_seed else None
    return _with_image_preprocess({
        "curve": curve,
        "seq_len": intact_T,
        "n_total": int(num_seeds),
        "n_success": len(per_seed),
        "n_skipped": int(num_seeds - len(per_seed)),
        "skip_reasons": skip_reasons,
        **_span_payload(spans, query_span),
    }, image_meta)


@_no_grad()
def pf3_curve_with_meta_from_sample(wrapper, sample,
                                    corruption_mode: str = "mask_50pct",
                                    num_seeds: int = 3,
                                    severity: Optional[float] = None) -> dict:
    """
    Sample-level PF-3 path. Required for LVR teacher-forced inputs because the
    assistant-side <lvr> metadata lives on ProbeSample.
    """
    image, image_meta = prepare_image_for_audit(wrapper, sample.image)
    sample = replace(sample, image=image)
    intact = _forward_attn_from_sample(wrapper, sample)
    intact_T = intact["seq_len"]
    intact_attns = intact["query_image_attn"]
    spans = intact["spans"]
    query_span = spans.preferred_query_span()
    image_span = spans.image_tokens

    n_layers = len(intact_attns)
    per_seed = []
    skip_reasons: dict[str, int] = {}
    for seed in range(num_seeds):
        corr_img = corrupt_image(image, corruption_mode, seed=seed, severity=severity)
        corr_sample = replace(sample, image=corr_img)
        corr = _forward_attn_from_sample(wrapper, corr_sample)
        curve_values, seed_skip = _query_image_attn_kl_curve(intact, corr)
        if curve_values is None:
            for reason, count in seed_skip.items():
                skip_reasons[reason] = skip_reasons.get(reason, 0) + count
            continue
        per_seed.append(curve_values)

    curve = np.asarray(per_seed).mean(axis=0) if per_seed else None
    return _with_image_preprocess({
        "curve": curve,
        "seq_len": intact_T,
        "n_total": int(num_seeds),
        "n_success": len(per_seed),
        "n_skipped": int(num_seeds - len(per_seed)),
        "skip_reasons": skip_reasons,
        **_span_payload(spans, query_span),
    }, image_meta)


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
