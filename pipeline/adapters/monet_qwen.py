"""Monet/Qwen2.5-VL adapter.

Monet's paper-grade latent inference path is implemented through its modified
vLLM runner. This adapter loads the published Monet Transformers model for
architecture probes and for the official Transformers latent-mode path used in
Monet training (`latent_mode`, `ce_patch_pos`, `ce_patch_vec`). The latter is a
real hidden-state replacement gate, but it is still distinct from scheduler-
native vLLM generation.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

from .base import ModelBundle
from .qwen_vl import QwenVLAdapter

logger = logging.getLogger("lvr_eval.adapters.monet_qwen")


def _resolve_path(value: Any, *, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _find_monet_source(cfg_model: dict, *, repo_root: Path) -> Path:
    candidates = [
        cfg_model.get("monet_source_path"),
        os.environ.get("MONET_SOURCE_PATH"),
        repo_root.parent / "Monet",
        repo_root.parent / "monet",
        Path("/home/pengguangyue/workspace/proj/Monet"),
        Path("/data/pengguangyue/proj/Monet"),
    ]
    tried: list[Path] = []
    for candidate in candidates:
        path = _resolve_path(candidate, base=repo_root)
        if path is None:
            continue
        tried.append(path)
        if (path / "monet_qwen_model" / "modeling_qwen2_5_vl_monet.py").is_file():
            return path
    raise RuntimeError(
        "Monet source checkout missing monet_qwen_model/modeling_qwen2_5_vl_monet.py. "
        "Set models.monet_7b.monet_source_path or MONET_SOURCE_PATH. Tried: "
        + ", ".join(str(path) for path in tried)
    )


def _load_monet_model_class(source_path: Path):
    source_text = str(source_path)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("monet_qwen_model.modeling_qwen2_5_vl_monet")
    return getattr(module, "Qwen2_5_VLForConditionalGeneration")


class MonetQwenAdapter(QwenVLAdapter):
    """Adapter for the public Monet-7B Transformers checkpoint."""

    def __init__(self, arch: str = "monet_qwen2_5_vl"):
        super().__init__("qwen2_5_vl")
        self.arch = arch

    def load(self, cfg_model: dict, *, dtype: torch.dtype, device: str) -> ModelBundle:
        try:
            from transformers import AutoProcessor
        except ValueError as exc:
            if "grouped_mm_fallback" in str(exc) or "infer_schema" in str(exc):
                raise RuntimeError(
                    "transformers 与 torch 版本不兼容。请运行: "
                    "pip install -U 'transformers>=4.51,<4.56'"
                ) from exc
            raise

        repo_root = Path.cwd()
        source_path = _find_monet_source(cfg_model, repo_root=repo_root)
        model_cls = _load_monet_model_class(source_path)

        path = cfg_model["path"]
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = model_cls.from_pretrained(
            path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=cfg_model.get("attn_implementation", "eager"),
        )
        model.to(device).eval()
        self._configure_latent_tokens(model, processor, cfg_model)
        image_pad_id = self._image_pad_id(processor, cfg_model)
        logger.info("loaded Monet model via source checkout: %s", source_path)
        return ModelBundle(model=model, processor=processor, image_pad_id=image_pad_id)

    def _configure_latent_tokens(self, model, processor, cfg_model: dict) -> None:
        tokenizer = processor.tokenizer
        latent_pad_token = cfg_model.get("latent_pad_token", "<abs_vis_token_pad>")
        latent_start_token = cfg_model.get("latent_start_token", "<abs_vis_token>")
        latent_end_token = cfg_model.get("latent_end_token", "</abs_vis_token>")
        latent_pad_id = int(cfg_model.get(
            "latent_pad_id",
            tokenizer.convert_tokens_to_ids(latent_pad_token),
        ))
        latent_start_id = int(cfg_model.get(
            "latent_start_id",
            tokenizer.convert_tokens_to_ids(latent_start_token),
        ))
        latent_end_id = int(cfg_model.get(
            "latent_end_id",
            tokenizer.convert_tokens_to_ids(latent_end_token),
        ))
        if latent_pad_id < 0 or latent_start_id < 0 or latent_end_id < 0:
            raise RuntimeError(
                "Monet latent tokens are missing from the tokenizer: "
                f"pad={latent_pad_id} start={latent_start_id} end={latent_end_id}"
            )
        model.config.latent_token_id = latent_pad_id
        model.config.latent_start_token_id = latent_start_id
        model.config.latent_end_token_id = latent_end_id
        if not hasattr(model.config, "answer_start_pattern") or not model.config.answer_start_pattern:
            model.config.answer_start_pattern = tokenizer.encode(
                "<|im_start|>assistant",
                add_special_tokens=False,
            )

    def get_spans(self, wrapper, inputs, model_outputs=None):
        spans = super().get_spans(wrapper, inputs, model_outputs=model_outputs)
        spans.notes.update({
            "adapter": "monet_qwen",
            "latent_tokens": "monet_abs_vis_latent_tokens",
            "transformers_latent_mode": "supports_ce_patch_pos_ce_patch_vec",
            "vllm_scheduler_native": "requires_modified_vllm_runner",
        })
        return spans

    def latent_prefix_text(self, wrapper, question: str, *, latent_size: int) -> str:
        latent_size = int(latent_size)
        if latent_size <= 0:
            raise ValueError("latent_size must be positive")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]}]
        base = wrapper.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        local = getattr(wrapper, "cfg", {}) or {}
        model_cfg = local.get("models", {}).get("monet_7b", {}) if isinstance(local, dict) else {}
        start = model_cfg.get("latent_start_token", "<abs_vis_token>")
        pad = model_cfg.get("latent_pad_token", "<abs_vis_token_pad>")
        end = model_cfg.get("latent_end_token", "</abs_vis_token>")
        return base + start + (pad * latent_size) + end + "\n"

    def _processor_inputs(self, wrapper, image, text: str):
        image = self._prepare_image(wrapper, image)
        return wrapper.processor(text=[text], images=[image], return_tensors="pt").to(wrapper.model.device)

    @torch.no_grad()
    def capture_latent_state(
        self,
        wrapper,
        image,
        question: str,
        *,
        latent_size: int,
        output_hidden_states: bool = False,
    ) -> dict:
        """Run Monet's official Transformers latent-mode path and capture `ce_patch_vec`."""
        prefix_text = self.latent_prefix_text(wrapper, question, latent_size=latent_size)
        inputs = self._processor_inputs(wrapper, image, prefix_text)
        outputs = wrapper.model(
            **inputs,
            latent_mode=True,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            return_dict=True,
        )
        pos = getattr(outputs, "ce_patch_pos", None)
        vec = getattr(outputs, "ce_patch_vec", None)
        if not pos or not isinstance(pos, list):
            raise RuntimeError("Monet latent_mode returned no ce_patch_pos")
        if not vec or not isinstance(vec, list):
            raise RuntimeError("Monet latent_mode returned no ce_patch_vec")
        positions = [int(v) for v in (pos[0] or [])]
        vectors = vec[0]
        if not torch.is_tensor(vectors) or vectors.ndim != 2:
            raise RuntimeError(f"Monet ce_patch_vec must be [n_latent, hidden], got {type(vectors)}")
        if len(positions) != int(vectors.shape[0]):
            raise RuntimeError(
                f"Monet ce_patch_pos/vec length mismatch: {len(positions)} vs {tuple(vectors.shape)}"
            )
        return {
            "prefix_text": prefix_text,
            "input_ids": inputs["input_ids"].detach(),
            "ce_patch_pos": [positions],
            "ce_patch_vec": [vectors.detach()],
            "trace_quality": "monet_transformers_latent_mode_v0",
            "latent_mode_path": "transformers_ce_patch_vec",
            "vllm_scheduler_native": False,
            "n_monet_latent_positions": len(positions),
            "n_captured_latent_states": int(vectors.shape[0]),
            "hidden_size": int(vectors.shape[-1]) if vectors.numel() else None,
            "captured_state_shapes": [list(vectors.shape)],
            "latent_positions": positions,
        }

    @torch.no_grad()
    def score_candidates_with_latents(
        self,
        wrapper,
        image,
        question: str,
        *,
        latent_size: int,
        ce_patch_pos: list[list[int]],
        ce_patch_vec: list[torch.Tensor],
        candidates: tuple[str, ...] | list[str],
    ) -> dict:
        """Forced-score constrained answer candidates with supplied latent vectors."""
        prefix_text = self.latent_prefix_text(wrapper, question, latent_size=latent_size)
        prefix_inputs = self._processor_inputs(wrapper, image, prefix_text)
        prefix_len = int(prefix_inputs["input_ids"].shape[1])
        scores: dict[str, float] = {}
        token_ids: dict[str, list[int]] = {}
        for candidate in candidates:
            full_text = prefix_text + str(candidate)
            inputs = self._processor_inputs(wrapper, image, full_text)
            labels = inputs["input_ids"].clone()
            outputs = wrapper.model(
                **inputs,
                labels=labels,
                latent_mode=False,
                ce_patch_pos=ce_patch_pos,
                ce_patch_vec=[tensor.to(wrapper.model.device) for tensor in ce_patch_vec],
                loss_type=["ce"],
                logits_to_keep=0,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits
            if logits is None:
                raise RuntimeError("Monet scoring forward returned no logits; loss_type=['ce'] is required")
            ids = inputs["input_ids"][0]
            answer_ids = ids[prefix_len:]
            if answer_ids.numel() == 0:
                raise RuntimeError(f"candidate {candidate!r} produced no answer tokens")
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)
            total = 0.0
            for pos in range(prefix_len, int(ids.shape[0])):
                total += float(log_probs[pos - 1, int(ids[pos])].detach().cpu().item())
            scores[str(candidate)] = total
            token_ids[str(candidate)] = [int(v) for v in answer_ids.detach().cpu().tolist()]
        pred = max(scores, key=lambda key: scores[key])
        return {
            "scores": scores,
            "token_ids": token_ids,
            "predicted_answer": pred,
            "prefix_len": prefix_len,
        }

    def generate_with_trace(self, wrapper, image, question: str, **kwargs) -> dict:
        raise NotImplementedError(
            "Monet true latent generation is implemented by the official modified vLLM "
            "runner. This adapter exposes the official Transformers latent-mode "
            "ce_patch_pos/ce_patch_vec path via capture_latent_state() for W13 causal "
            "range gates, but generate_with_trace() remains reserved for scheduler-native "
            "vLLM tracing."
        )
