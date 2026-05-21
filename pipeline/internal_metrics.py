"""
pipeline/internal_metrics.py
============================
从 bf3_today.py / pf3_today.py 抽出的**核心**(去掉 CLI / 数据加载),
重构成可被 BF-1 / CF-2 复用的、作用于 VLMWrapper 的函数。

两个指标都是 **model-internal、逐层** 的:

BF-3 (Confidence Progression)
    在最后一个 latent 位置, 对每层 hidden state 做 logit-lens, 算 entropy。
    返回 curve[n_layers]。需要 output_hidden_states。

PF-3 (Corrupted-vs-Intact Attention Distance)
    intact 与 corrupted 两次 forward, 取 latent→image 的 attention 分布,
    逐层算 KL(intact || corrupted)。返回 curve[n_layers]。
    需要 eager attention + output_attentions。

关键: 这些 forward 是普通 forward —— 若外部已用 ablation hook 包住
(见 ablation.py 的 ablate_layer 上下文), 这里算出的 curve 自动反映“消融后”的
内部状态。这正是 BF-1 与 BF-3/PF-3 联动的机制。
"""
from __future__ import annotations

import logging
import random
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
#  latent span / image token 定位 (Qwen2.5-VL: <|image_pad|>)
# --------------------------------------------------------------------------- #
def get_latent_span(input_ids: torch.Tensor, image_pad_id: int) -> tuple[int, int]:
    """latent = 最后一个 image token 之后 到 input 末尾。"""
    img_pos = (input_ids[0] == image_pad_id).nonzero(as_tuple=True)[0]
    if len(img_pos) == 0:
        raise ValueError("输入里没有 image token —— 检查 image_pad token 是否正确")
    return img_pos[-1].item() + 1, input_ids.shape[1]


def get_image_token_indices(input_ids: torch.Tensor, image_pad_id: int) -> torch.Tensor:
    return (input_ids[0] == image_pad_id).nonzero(as_tuple=True)[0]


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


@_no_grad()
def bf3_curve(wrapper, image: Image.Image, question: str) -> np.ndarray:
    """单样本逐层 entropy curve [n_layers]。普通 forward (会受外部 ablation hook 影响)。"""
    inputs = wrapper.build_inputs(image, question)
    out = wrapper.model(**inputs, output_hidden_states=True,
                        output_attentions=False, return_dict=True)
    latent_start, latent_end = get_latent_span(inputs["input_ids"], wrapper.image_pad_id)
    last_pos = latent_end - 1
    ents = []
    # hidden_states: tuple(len = n_layers+1); 跳过 index0(embedding)
    for li in range(1, len(out.hidden_states)):
        h = out.hidden_states[li][0, last_pos, :]
        ents.append(float(logit_lens_entropy(wrapper, h)[0]))
    return np.asarray(ents)


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
        return random_patch_mask(image, ratio=ratio, seed=seed)
    if mode == "gaussian_blur":
        radius = severity if severity is not None else 10
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
    inputs = wrapper.build_inputs(image, question)
    out = wrapper.model(**inputs, output_hidden_states=False,
                        output_attentions=True, return_dict=True)
    return {"attentions": out.attentions, "input_ids": inputs["input_ids"],
            "seq_len": inputs["input_ids"].shape[1]}


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
    intact = _forward_attn(wrapper, image, question)
    intact_T = intact["seq_len"]
    intact_attns = intact["attentions"]
    latent_start, latent_end = get_latent_span(intact["input_ids"], wrapper.image_pad_id)
    image_idx = get_image_token_indices(intact["input_ids"], wrapper.image_pad_id)
    if len(image_idx) == 0:
        raise ValueError("No image tokens found")

    n_layers = len(intact_attns)
    per_seed = []
    # patch_shuffle / blur 在固定 severity 下其实是确定性(blur)或仅依赖 seed(shuffle)
    for seed in range(num_seeds):
        corr_img = corrupt_image(image, corruption_mode, seed=seed, severity=severity)
        corr = _forward_attn(wrapper, corr_img, question)
        if corr["seq_len"] != intact_T:        # token 数必须一致
            continue
        corr_attns = corr["attentions"]
        per_layer = []
        for li in range(n_layers):
            Ai = intact_attns[li][0][:, latent_start:latent_end, :][:, :, image_idx]
            Ac = corr_attns[li][0][:, latent_start:latent_end, :][:, :, image_idx]
            Ai = Ai / (Ai.sum(dim=-1, keepdim=True) + 1e-8)
            Ac = Ac / (Ac.sum(dim=-1, keepdim=True) + 1e-8)
            per_layer.append(_kl_safe(Ai.float(), Ac.float()).mean().item())
        per_seed.append(per_layer)

    if not per_seed:
        return None, intact_T
    return np.asarray(per_seed).mean(axis=0), intact_T


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
