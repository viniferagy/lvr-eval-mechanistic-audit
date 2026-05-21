"""Adapter for official LVR Qwen models."""
from __future__ import annotations

import logging

import torch

from .base import ModelBundle
from .qwen_vl import QwenVLAdapter
from .spans import AuditSpans, TokenSpan

logger = logging.getLogger("lvr_eval.adapters.lvr_qwen")


class LVRQwenAdapter(QwenVLAdapter):
    """
    Adapter for VincentLeebang/LVR-style QwenWithLVR models.

    Teacher-forced audits use discrete ``<|lvr|>`` placeholder token positions.
    Inference-time LVR is continuous hidden-state recurrence and should be
    audited through ``generate_with_trace``.
    """

    def __init__(self, arch: str = "lvr_qwen2_5_vl"):
        super().__init__("qwen2_5_vl")
        self.arch = arch

    def load(self, cfg_model: dict, *, dtype: torch.dtype, device: str) -> ModelBundle:
        from transformers import AutoConfig, AutoProcessor

        path = cfg_model["path"]
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)

        try:
            from src.model.qwen_lvr_model import QwenWithLVR
            from src.train.monkey_patch_forward_lvr import (
                replace_qwen2_5_with_mixed_modality_forward_lvr,
            )

            replace_qwen2_5_with_mixed_modality_forward_lvr(
                inference_mode=True,
                lvr_head=getattr(config, "lvr_head", False),
            )

            model = QwenWithLVR.from_pretrained(
                path,
                config=config,
                torch_dtype=dtype,
                trust_remote_code=True,
                attn_implementation="eager",
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load LVR model via official QwenWithLVR path. "
                "Make sure VincentLeebang/lvr source is installed/importable, "
                "or vendor the required model files into this repo."
            ) from exc

        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model.to(device).eval()
        image_pad_id = self._image_pad_id(processor, cfg_model)
        self._ensure_lvr_ids(model, processor, cfg_model)
        return ModelBundle(model=model, processor=processor, image_pad_id=image_pad_id)

    def _ensure_lvr_ids(self, model, processor, cfg_model: dict):
        token_names = {
            "lvr_start_id": cfg_model.get("lvr_start_token", "<|lvr_start|>"),
            "lvr_id": cfg_model.get("lvr_token", "<|lvr|>"),
            "lvr_latent_end_id": cfg_model.get(
                "lvr_latent_end_token", "<|lvr_latent_end|>"
            ),
            "lvr_end_id": cfg_model.get("lvr_end_token", "<|lvr_end|>"),
        }

        for attr, tok in token_names.items():
            if getattr(model.config, attr, None) is None:
                tok_id = processor.tokenizer.convert_tokens_to_ids(tok)
                if tok_id is None or tok_id < 0:
                    raise ValueError(f"LVR token {tok!r} not found in tokenizer.")
                setattr(model.config, attr, int(tok_id))

        logger.info(
            "LVR ids: start=%s lvr=%s latent_end=%s end=%s",
            model.config.lvr_start_id,
            model.config.lvr_id,
            model.config.lvr_latent_end_id,
            model.config.lvr_end_id,
        )

    def _make_lvr_sequence(self, wrapper, sample, cfg: dict | None = None) -> str:
        cfg = cfg or getattr(wrapper, "cfg", {}) or {}
        audit_cfg = cfg.get("audit", {}) if isinstance(cfg, dict) else {}
        n = int(audit_cfg.get("lvr_num_tokens", 16))
        return "<|lvr_start|>" + "<|lvr|>" * n + "<|lvr_end|>"

    def _teacher_forced_assistant_text(self, wrapper, sample) -> str:
        assistant = sample.lvr_assistant or ""
        if "<lvr>" in assistant:
            return assistant.replace("<lvr>", self._make_lvr_sequence(wrapper, sample), 1)

        answer = sample.answer or ""
        return f"{self._make_lvr_sequence(wrapper, sample)}\n<answer>{answer}</answer>"

    def build_teacher_forced_inputs_from_sample(self, wrapper, sample):
        assistant_text = self._teacher_forced_assistant_text(wrapper, sample)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample.image},
                    {"type": "text", "text": sample.question},
                ],
            },
            {
                "role": "assistant",
                "content": assistant_text,
            },
        ]
        text = wrapper.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = wrapper.processor(
            text=[text],
            images=[sample.image],
            padding=True,
            return_tensors="pt",
        ).to(wrapper.model.device)
        return inputs

    def build_inputs_from_sample(self, wrapper, sample):
        cfg = getattr(wrapper, "cfg", {}) or {}
        audit_cfg = cfg.get("audit", {}) if isinstance(cfg, dict) else {}
        mode = audit_cfg.get("mode", "teacher_forced")
        if mode == "teacher_forced":
            return self.build_teacher_forced_inputs_from_sample(wrapper, sample)
        return self.build_inputs(wrapper, sample.image, sample.question)

    def get_spans(self, wrapper, inputs, model_outputs=None) -> AuditSpans:
        ids = inputs["input_ids"][0]
        image_pos = (ids == wrapper.image_pad_id).nonzero(as_tuple=True)[0]
        if len(image_pos) == 0:
            raise ValueError("No image tokens found in LVR input.")

        image_span = TokenSpan(
            start=int(image_pos[0].item()),
            end=int(image_pos[-1].item()) + 1,
            kind="image_tokens",
        )

        lvr_id = wrapper.model.config.lvr_id
        lvr_pos = (ids == lvr_id).nonzero(as_tuple=True)[0]
        lvr_span = None
        if len(lvr_pos) > 0:
            lvr_span = TokenSpan(
                start=int(lvr_pos[0].item()),
                end=int(lvr_pos[-1].item()) + 1,
                kind="lvr_placeholder_tokens",
            )

        cfg = getattr(wrapper, "cfg", {}) or {}
        audit_cfg = cfg.get("audit", {}) if isinstance(cfg, dict) else {}
        allow_fallback = bool(audit_cfg.get("allow_lvr_fallback_to_answer_probe", False))
        mode = audit_cfg.get("mode", "teacher_forced")
        if mode == "teacher_forced" and lvr_span is None and not allow_fallback:
            raise ValueError(
                "LVR teacher_forced audit expected <|lvr|> placeholder tokens, "
                "but none were found. Check source_type=lvr_json and sample.lvr_assistant."
            )

        return AuditSpans(
            image_tokens=image_span,
            question_tokens=None,
            lvr_placeholder_tokens=lvr_span,
            latent_tokens=None,
            answer_probe_pos=int(ids.shape[0] - 1),
            notes={
                "adapter": "lvr_qwen",
                "mode": "teacher_forced_spans_only",
                "lvr_start_id": int(wrapper.model.config.lvr_start_id),
                "lvr_id": int(wrapper.model.config.lvr_id),
                "lvr_latent_end_id": int(wrapper.model.config.lvr_latent_end_id),
                "lvr_end_id": int(wrapper.model.config.lvr_end_id),
            },
        )

    @torch.no_grad()
    def generate_with_trace(
        self,
        wrapper,
        image,
        question: str,
        *,
        decoding_strategy: str = "steps",
        lvr_steps: int = 16,
        max_new_tokens: int = 128,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
    ) -> dict:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        text = wrapper.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = wrapper.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(wrapper.model.device)

        spans = self.get_spans(wrapper, inputs)
        gen = wrapper.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            decoding_strategy=decoding_strategy,
            lvr_steps=[lvr_steps],
            return_dict_in_generate=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        seq = gen.sequences[0]
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = seq[prompt_len:]
        text_out = wrapper.processor.batch_decode(
            [new_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]

        lvr_start_id = int(wrapper.model.config.lvr_start_id)
        lvr_end_id = int(wrapper.model.config.lvr_end_id)
        lvr_generated_positions = []
        in_lvr = False
        for pos in range(prompt_len, int(seq.shape[0])):
            tid = int(seq[pos].item())
            if tid == lvr_start_id:
                in_lvr = True
            elif tid == lvr_end_id:
                in_lvr = False
            elif in_lvr:
                lvr_generated_positions.append(pos)

        return {
            "inputs": inputs,
            "prompt_spans": spans,
            "sequences": gen.sequences,
            "generated_text": text_out,
            "attentions": getattr(gen, "attentions", None),
            "hidden_states": getattr(gen, "hidden_states", None),
            "lvr_generated_positions": lvr_generated_positions,
            "trace_quality": "approx_from_generated_token_ids",
            "notes": {
                "decoding_strategy": decoding_strategy,
                "lvr_steps": lvr_steps,
                "needs_loop_instrumentation": True,
            },
        }
