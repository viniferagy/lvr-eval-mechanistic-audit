"""Adapter for official LVR Qwen models."""
from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path

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
        self.lvr_start_token = "<|lvr_start|>"
        self.lvr_token = "<|lvr|>"
        self.lvr_latent_end_token = "<|lvr_latent_end|>"
        self.lvr_end_token = "<|lvr_end|>"

    def load(self, cfg_model: dict, *, dtype: torch.dtype, device: str) -> ModelBundle:
        from transformers import AutoConfig, AutoProcessor

        path = cfg_model["path"]
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        self._ensure_lvr_source_importable(cfg_model)

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
            self._patch_generation_cache_position_compat(model)
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

    def _patch_generation_cache_position_compat(self, model) -> None:
        """Accept the old two-arg cache-position call used by the LVR source.

        Current Transformers expects `_get_initial_cache_position(seq_length,
        device, model_kwargs)`. Some VincentLeebang/lvr generation branches
        still call `_get_initial_cache_position(input_ids, model_kwargs)`.
        Patch only the loaded model instance so external source files remain
        untouched.
        """
        original = getattr(model, "_get_initial_cache_position", None)
        if original is None or getattr(original, "_lvr_eval_compat", False):
            return

        def compat(this, *args, **kwargs):
            if len(args) == 2 and not kwargs:
                input_ids, model_kwargs = args
                if hasattr(input_ids, "shape") and hasattr(input_ids, "device"):
                    return original(int(input_ids.shape[-1]), input_ids.device, model_kwargs)
            return original(*args, **kwargs)

        compat._lvr_eval_compat = True  # type: ignore[attr-defined]
        model._get_initial_cache_position = types.MethodType(compat, model)

    def _ensure_lvr_source_importable(self, cfg_model: dict):
        """Add the official VincentLeebang/lvr checkout to sys.path if configured."""
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            cfg_model.get("lvr_source_path"),
            os.environ.get("LVR_SOURCE_PATH"),
            repo_root.parent / "lvr",
            Path("/home/pengguangyue/workspace/proj/lvr"),
            Path("/data/pengguangyue/proj/lvr"),
        ]
        tried = []
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(os.path.expandvars(os.path.expanduser(str(candidate))))
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            tried.append(str(path))
            marker = path / "src" / "model" / "qwen_lvr_model.py"
            if marker.is_file():
                path_str = str(path)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)
                logger.info("LVR source path = %s", path_str)
                return

        raise RuntimeError(
            "Official LVR source checkout is required but was not found. "
            "Set models.lvr_7b.lvr_source_path or LVR_SOURCE_PATH to a directory "
            "containing src/model/qwen_lvr_model.py. Tried: "
            + ", ".join(tried)
        )

    def _ensure_lvr_ids(self, model, processor, cfg_model: dict):
        self.lvr_start_token = cfg_model.get("lvr_start_token", "<|lvr_start|>")
        self.lvr_token = cfg_model.get("lvr_token", "<|lvr|>")
        self.lvr_latent_end_token = cfg_model.get(
            "lvr_latent_end_token", "<|lvr_latent_end|>"
        )
        self.lvr_end_token = cfg_model.get("lvr_end_token", "<|lvr_end|>")

        token_names = {
            "lvr_start_id": self.lvr_start_token,
            "lvr_id": self.lvr_token,
            "lvr_latent_end_id": self.lvr_latent_end_token,
            "lvr_end_id": self.lvr_end_token,
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
        """
        Match official proj/lvr/src/dataset/data_utils.py::replace_lvr_tokens.

        Implemented mode:
          fixed_num_of_lvr_tokens is not None
            => <|lvr_start|> + N * <|lvr|> + <|lvr_end|>

        Important:
          Official fixed-token mode does NOT insert <|lvr_latent_end|>.
          <|lvr_latent_end|> only appears in the official dynamic token-index
          branch when fixed_num_of_lvr_tokens is None and latent_end_token is set.

        This repo currently supports the fixed teacher-forced audit path only.
        """
        cfg = cfg or getattr(wrapper, "cfg", {}) or {}
        audit_cfg = cfg.get("audit", {}) if isinstance(cfg, dict) else {}

        expansion_mode = str(audit_cfg.get("lvr_expansion_mode", "fixed")).strip().lower()
        if expansion_mode not in {"fixed", "fixed_num_tokens"}:
            raise NotImplementedError(
                "Only official fixed_num_of_lvr_tokens LVR expansion is implemented. "
                "Dynamic token-index expansion requires lvr_token_idxs_list from the "
                "official data pipeline and is not yet supported here."
            )

        include_latent_end = bool(audit_cfg.get("lvr_include_latent_end_token", False))
        if include_latent_end:
            raise ValueError(
                "Invalid LVR config: official replace_lvr_tokens() does not insert "
                "<|lvr_latent_end|> in fixed_num_of_lvr_tokens mode. Set "
                "audit.lvr_include_latent_end_token=false, or implement the official "
                "dynamic token-index branch before enabling latent_end."
            )

        n = int(audit_cfg.get("lvr_num_tokens", 16))
        if n <= 0:
            raise ValueError(f"audit.lvr_num_tokens must be positive, got {n}")

        return self.lvr_start_token + self.lvr_token * n + self.lvr_end_token

    def _teacher_forced_assistant_text(self, wrapper, sample) -> str:
        cfg = getattr(wrapper, "cfg", {}) or {}
        audit_cfg = cfg.get("audit", {}) if isinstance(cfg, dict) else {}
        assistant = sample.lvr_assistant or ""
        num_lvr = assistant.count("<lvr>")
        if num_lvr > 1 and not bool(audit_cfg.get("allow_multiple_lvr_placeholders", False)):
            raise ValueError(
                f"LVR teacher_forced audit currently supports exactly one <lvr> block, got {num_lvr}. "
                "Multiple blocks require disjoint span support."
            )
        if num_lvr > 0:
            return assistant.replace("<lvr>", self._make_lvr_sequence(wrapper, sample))

        if not bool(audit_cfg.get("allow_synthetic_lvr_assistant", False)):
            raise ValueError(
                "LVR teacher_forced mode requires sample.lvr_assistant containing <lvr>. "
                "Set audit.allow_synthetic_lvr_assistant=true only for debugging."
            )

        answer = sample.answer or ""
        return f"{self._make_lvr_sequence(wrapper, sample)}\n<answer>{answer}</answer>"

    def build_teacher_forced_inputs_from_sample(self, wrapper, sample):
        image = self._prepare_image(wrapper, sample.image)
        assistant_text = self._teacher_forced_assistant_text(wrapper, sample)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
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
            images=[image],
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
                "lvr_expansion_mode": audit_cfg.get("lvr_expansion_mode", "fixed"),
                "lvr_include_latent_end_token": bool(
                    audit_cfg.get("lvr_include_latent_end_token", False)
                ),
                "allow_multiple_lvr_placeholders": bool(
                    audit_cfg.get("allow_multiple_lvr_placeholders", False)
                ),
                "span_semantics": (
                    "continuous_span_may_include_interleaving_text"
                    if bool(audit_cfg.get("allow_multiple_lvr_placeholders", False))
                    else "single_contiguous_lvr_placeholder_block"
                ),
                "lvr_placeholder_span_excludes_latent_end": True,
            },
        )

    def _extract_lvr_positions_from_sequence(
        self,
        seq,
        prompt_len: int,
        *,
        lvr_start_id: int,
        lvr_id: int,
        lvr_latent_end_id: int | None,
        lvr_end_id: int,
    ) -> dict:
        lvr_token_positions = []
        lvr_latent_end_positions = []
        lvr_block_spans = []
        unexpected_lvr_inner_positions = []

        in_lvr = False
        current_start = None

        for pos in range(prompt_len, int(seq.shape[0])):
            tid = int(seq[pos].item())

            if tid == lvr_start_id:
                in_lvr = True
                current_start = pos
                continue

            if tid == lvr_end_id:
                if in_lvr and current_start is not None:
                    lvr_block_spans.append([current_start, pos + 1])
                in_lvr = False
                current_start = None
                continue

            if not in_lvr:
                continue

            if lvr_latent_end_id is not None and tid == lvr_latent_end_id:
                lvr_latent_end_positions.append(pos)
            elif tid == lvr_id:
                lvr_token_positions.append(pos)
            else:
                unexpected_lvr_inner_positions.append({
                    "position": pos,
                    "token_id": tid,
                })

        return {
            "lvr_generated_positions": lvr_token_positions,
            "lvr_token_positions": lvr_token_positions,
            "lvr_latent_end_positions": lvr_latent_end_positions,
            "lvr_block_spans": lvr_block_spans,
            "unexpected_lvr_inner_positions": unexpected_lvr_inner_positions,
        }

    @torch.no_grad()
    def generate(self, wrapper, images: list, prompts: list[str],
                 max_new_tokens: int = 64) -> list[str]:
        cfg = getattr(wrapper, "cfg", {}) or {}
        audit_cfg = cfg.get("audit", {}) if isinstance(cfg, dict) else {}
        decoding_strategy = audit_cfg.get("lvr_decoding_strategy", "steps")
        lvr_steps = int(audit_cfg.get("lvr_steps", 16))

        images = [self._prepare_image(wrapper, image) for image in images]
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
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            decoding_strategy=decoding_strategy,
            lvr_steps=[lvr_steps] * len(images),
        )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return wrapper.processor.batch_decode(trimmed, skip_special_tokens=True)

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
        output_scores: bool = False,
    ) -> dict:
        image = self._prepare_image(wrapper, image)
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

        # Generation prompts do not contain teacher-forced <|lvr|> placeholders.
        # Use baseline prompt spans here; generated LVR state is traced below.
        spans = super().get_spans(wrapper, inputs)
        gen = wrapper.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            decoding_strategy=decoding_strategy,
            lvr_steps=[lvr_steps],
            return_dict_in_generate=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
        )

        seq = gen.sequences[0]
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = seq[prompt_len:]
        text_out = wrapper.processor.batch_decode(
            [new_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        generated_ids = [int(token_id) for token_id in new_ids.detach().cpu().tolist()]
        decoded_generated_tokens = []
        for token_id in generated_ids:
            try:
                token_text = wrapper.processor.tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                token_text = wrapper.processor.tokenizer.decode([int(token_id)])
            decoded_generated_tokens.append(token_text)
        scores = getattr(gen, "scores", None)
        score_token_alignment = []
        if scores is not None:
            for idx in range(len(scores)):
                token_id = generated_ids[idx] if idx < len(generated_ids) else None
                score_token_alignment.append({
                    "score_index": int(idx),
                    "generated_token_index": int(idx) if token_id is not None else None,
                    "generated_token_id": token_id,
                    "generated_token_text": (
                        decoded_generated_tokens[idx]
                        if idx < len(decoded_generated_tokens)
                        else None
                    ),
                })

        lvr_start_id = int(wrapper.model.config.lvr_start_id)
        lvr_id = int(wrapper.model.config.lvr_id)
        lvr_latent_end_id = getattr(wrapper.model.config, "lvr_latent_end_id", None)
        if lvr_latent_end_id is not None:
            lvr_latent_end_id = int(lvr_latent_end_id)
        lvr_end_id = int(wrapper.model.config.lvr_end_id)
        lvr_pos_info = self._extract_lvr_positions_from_sequence(
            seq,
            prompt_len,
            lvr_start_id=lvr_start_id,
            lvr_id=lvr_id,
            lvr_latent_end_id=lvr_latent_end_id,
            lvr_end_id=lvr_end_id,
        )

        return {
            "inputs": inputs,
            "prompt_spans": spans,
            "sequences": gen.sequences,
            "prompt_len": int(prompt_len),
            "generated_ids": generated_ids,
            "decoded_generated_tokens": decoded_generated_tokens,
            "score_token_alignment": score_token_alignment,
            "generated_text": text_out,
            "attentions": getattr(gen, "attentions", None),
            "hidden_states": getattr(gen, "hidden_states", None),
            "scores": scores,
            **lvr_pos_info,
            "trace_quality": "approx_from_generated_token_ids",
            "notes": {
                "decoding_strategy": decoding_strategy,
                "lvr_steps": lvr_steps,
                "needs_loop_instrumentation": True,
            },
        }
