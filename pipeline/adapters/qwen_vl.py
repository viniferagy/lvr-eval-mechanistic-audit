"""Qwen-VL family adapter."""
from __future__ import annotations

import logging

import torch

from .base import ModelBundle

logger = logging.getLogger("lvr_eval.adapters.qwen_vl")


class QwenVLAdapter:
    """Adapter for Qwen2.5-VL/Qwen3-VL style chat-template processors."""

    def __init__(self, arch: str):
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

        path = cfg_model["path"]
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        common_kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )

        model = None
        if self.arch == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, **common_kwargs)
        elif self.arch == "qwen3_vl":
            try:
                from transformers import Qwen3VLForConditionalGeneration  # type: ignore
                model = Qwen3VLForConditionalGeneration.from_pretrained(path, **common_kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Qwen3VL 专用类不可用(%s)，回退 AutoModelForVision2Seq", exc)

        if model is None:
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(path, **common_kwargs)

        model.to(device).eval()
        image_pad_id = self._image_pad_id(processor, cfg_model)
        return ModelBundle(model=model, processor=processor, image_pad_id=image_pad_id)

    def _image_pad_id(self, processor, cfg_model: dict) -> int:
        tok_text = cfg_model.get("image_pad_token", "<|image_pad|>")
        image_pad_id = processor.tokenizer.convert_tokens_to_ids(tok_text)
        if image_pad_id is None or image_pad_id < 0:
            logger.warning("image_pad token '%s' 不在词表，PF-3/BF-3 latent span 可能失效", tok_text)
        return image_pad_id

    def build_inputs(self, wrapper, image, question: str):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        text = wrapper.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = wrapper.processor(text=[text], images=[image], return_tensors="pt")
        return inputs.to(wrapper.model.device)

    @torch.no_grad()
    def generate(self, wrapper, images: list, prompts: list[str],
                 max_new_tokens: int = 64) -> list[str]:
        messages = [[{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}] for image, prompt in zip(images, prompts)]
        texts = [
            wrapper.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        inputs = wrapper.processor(
            text=texts, images=list(images), padding=True, return_tensors="pt"
        ).to(wrapper.model.device)
        generated = wrapper.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return wrapper.processor.batch_decode(trimmed, skip_special_tokens=True)

