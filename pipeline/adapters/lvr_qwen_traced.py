"""Sparse hook instrumentation for LVR generation traces."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from .lvr_qwen import LVRQwenAdapter

logger = logging.getLogger("lvr_eval.adapters.lvr_qwen_traced")


def _module_at_path(root: Any, path: str):
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _first_module(root: Any, paths: list[str]):
    for path in paths:
        module = _module_at_path(root, path)
        if module is not None:
            return path, module
    return None, None


def _tensor_bool_list(value) -> list[bool] | None:
    if not torch.is_tensor(value):
        return None
    flat = value.detach().bool().flatten().cpu().tolist()
    return [bool(item) for item in flat]


def sampled_layer_indices(n_layers: int) -> list[int]:
    if n_layers <= 0:
        return []
    raw = [0, n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1]
    out = []
    for idx in raw:
        idx = max(0, min(n_layers - 1, int(idx)))
        if idx not in out:
            out.append(idx)
    return out


def _shape(value) -> list[int] | None:
    if torch.is_tensor(value):
        return [int(dim) for dim in value.shape]
    return None


def _last_hidden_shape(output) -> list[int] | None:
    tensor = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(tensor):
        return None
    if tensor.dim() >= 3:
        return [int(tensor.shape[0]), 1, int(tensor.shape[-1])]
    return _shape(tensor)


@dataclass
class TraceRecorder:
    """Context manager that records sparse generation-time hook metadata."""

    wrapper: Any
    capture_tensors: bool = False
    max_captured_tensors: int = 16
    capture_device_policy: str = "cpu_float32"
    patch_states: list[Any] | None = None
    patch_steps: set[int] | None = None
    patch_shape_policy: str = "strict"
    handles: list[Any] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    captured_states: list[dict[str, Any]] = field(default_factory=list)
    n_patch_applied: int = 0
    sampled_layers: list[int] = field(default_factory=list)
    module_paths: dict[str, str] = field(default_factory=dict)
    missing_modules: list[str] = field(default_factory=list)

    def __enter__(self):
        self.sampled_layers = sampled_layer_indices(getattr(self.wrapper, "n_layers", 0))
        self._register()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _append(self, event: dict[str, Any]):
        event["event_index"] = len(self.events)
        self.events.append(event)

    def _should_capture_tensor(self, lvr_mode, tensor) -> bool:
        if not self.capture_tensors or tensor is None or not torch.is_tensor(tensor):
            return False
        if len(self.captured_states) >= int(self.max_captured_tensors):
            return False
        mode_values = _tensor_bool_list(lvr_mode)
        return bool(mode_values and any(mode_values))

    def _store_tensor(self, *, kind: str, step_index: int, tensor, lvr_mode=None):
        if not self._should_capture_tensor(lvr_mode, tensor):
            return
        stored = tensor.detach()
        if self.capture_device_policy == "cpu_float32":
            stored = stored.float().cpu()
        else:
            stored = stored.clone()
        self.captured_states.append({
            "kind": kind,
            "step_index": int(step_index),
            "shape": _shape(tensor),
            "lvr_mode_switch": _tensor_bool_list(lvr_mode),
            "tensor": stored,
        })

    def _patch_tensor_for_step(self, step_index: int, current):
        if not self.patch_states or current is None or not torch.is_tensor(current):
            return current
        patch_steps = self.patch_steps
        if patch_steps is not None and step_index not in patch_steps:
            return current
        patch_idx = min(self.n_patch_applied, len(self.patch_states) - 1)
        replacement = self.patch_states[patch_idx]
        if replacement is None or not torch.is_tensor(replacement):
            return current
        patched = current.clone()
        repl = replacement.to(device=current.device, dtype=current.dtype)
        if repl.dim() == current.dim() - 1:
            repl = repl.unsqueeze(0)
        if repl.shape != patched.shape:
            if self.patch_shape_policy != "slice":
                raise RuntimeError(
                    "latent patch shape mismatch under strict policy: "
                    f"current={tuple(patched.shape)} replacement={tuple(repl.shape)}"
                )
            slices = tuple(slice(0, min(a, b)) for a, b in zip(patched.shape, repl.shape))
            patched[slices] = repl[slices]
        else:
            patched = repl.clone()
        self.n_patch_applied += 1
        return patched

    def _register(self):
        model = self.wrapper.model
        self._register_model_forward(model)
        self._register_embed(model)
        self._register_norm(model)
        self._register_visual_merger(model)
        self._register_layers()
        self._register_lm_head(model)

    def _register_model_forward(self, model):
        self.module_paths["model_forward"] = type(model).__name__

        def hook(_module, args, kwargs=None):
            kwargs = kwargs or {}
            lvr_mode = kwargs.get("lvr_mode_switch")
            last_hidden = kwargs.get("last_position_hidden_state")
            step_index = len([
                event for event in self.events
                if event.get("hook") == "model_forward_pre"
            ])
            patch_applied_before = int(self.n_patch_applied)
            if last_hidden is not None and self.patch_states:
                patched = self._patch_tensor_for_step(step_index, last_hidden)
                if patched is not last_hidden:
                    kwargs["last_position_hidden_state"] = patched
                    last_hidden = patched
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            self._store_tensor(
                kind="incoming_last_position_hidden_state",
                step_index=step_index,
                tensor=last_hidden,
                lvr_mode=lvr_mode,
            )
            self._append({
                "hook": "model_forward_pre",
                "step_index": step_index,
                "lvr_mode_switch": _tensor_bool_list(lvr_mode),
                "last_position_hidden_state_shape": _shape(last_hidden),
                "input_ids_shape": _shape(input_ids),
                "inputs_embeds_shape": _shape(kwargs.get("inputs_embeds")),
                "patched_last_position_hidden_state": int(self.n_patch_applied) > patch_applied_before,
            })
            return args, kwargs

        self.handles.append(model.register_forward_pre_hook(hook, with_kwargs=True))

        def output_hook(_module, _args, output):
            step_index = len([
                event for event in self.events
                if event.get("hook") == "model_forward_output"
            ])
            tensor = getattr(output, "last_position_hidden_state", None)
            lvr_mode = None
            pre_events = [
                event for event in self.events
                if event.get("hook") == "model_forward_pre"
            ]
            if step_index < len(pre_events):
                lvr_mode = pre_events[step_index].get("lvr_mode_switch")
                if lvr_mode is not None:
                    lvr_mode = torch.tensor(lvr_mode, dtype=torch.bool)
            self._store_tensor(
                kind="output_last_position_hidden_state",
                step_index=step_index,
                tensor=tensor,
                lvr_mode=lvr_mode,
            )
            self._append({
                "hook": "model_forward_output",
                "step_index": step_index,
                "last_position_hidden_state_shape": _shape(tensor),
            })
            return output

        self.handles.append(model.register_forward_hook(output_hook))

    def _register_embed(self, model):
        path, module = _first_module(model, [
            "model.model.embed_tokens",
            "model.language_model.model.embed_tokens",
            "model.language_model.embed_tokens",
            "language_model.model.embed_tokens",
            "language_model.embed_tokens",
            "model.embed_tokens",
        ])
        if module is None and hasattr(model, "get_input_embeddings"):
            module = model.get_input_embeddings()
            path = "get_input_embeddings()"
        if module is None:
            self.missing_modules.append("embed_tokens")
            return
        self.module_paths["embed_tokens"] = path

        def hook(_module, args, kwargs=None):
            kwargs = kwargs or {}
            input_ids = kwargs.get("input_ids")
            inputs_embeds = kwargs.get("inputs_embeds")
            if input_ids is None and args:
                input_ids = args[0]
            mode = "latent" if inputs_embeds is not None else "token"
            self._append({
                "hook": "embed_tokens_pre",
                "mode": mode,
                "input_ids_shape": _shape(input_ids),
                "inputs_embeds_shape": _shape(inputs_embeds),
            })

        self.handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

    def _register_norm(self, model):
        path, module = _first_module(model, [
            "model.model.norm",
            "model.language_model.norm",
            "language_model.model.norm",
            "model.norm",
        ])
        if module is None:
            self.missing_modules.append("model.norm")
            return
        self.module_paths["model_norm"] = path

        def hook(_module, _args, output):
            self._append({
                "hook": "model_norm",
                "last_hidden_state_shape": _last_hidden_shape(output),
            })
            return output

        self.handles.append(module.register_forward_hook(hook))

    def _register_visual_merger(self, model):
        path, module = _first_module(model, [
            "model.visual.merger",
            "visual.merger",
            "model.vision_tower.merger",
        ])
        if module is None:
            self.missing_modules.append("visual.merger")
            return
        self.module_paths["visual_merger"] = path

        def hook(_module, _args, output):
            self._append({"hook": "visual_merger", "output_shape": _shape(output)})
            return output

        self.handles.append(module.register_forward_hook(hook))

    def _register_layers(self):
        layers = getattr(self.wrapper, "layers", [])

        def make_hook(layer_idx: int):
            def hook(_module, _args, output):
                self._append({
                    "hook": "decoder_layer",
                    "layer": int(layer_idx),
                    "resid_sample_shape": _last_hidden_shape(output),
                })
                return output

            return hook

        for idx in self.sampled_layers:
            if idx >= len(layers):
                continue
            self.handles.append(layers[idx].register_forward_hook(make_hook(idx)))

    def _register_lm_head(self, model):
        path, module = _first_module(model, [
            "lm_head",
            "language_model.lm_head",
            "model.lm_head",
        ])
        if module is None:
            self.missing_modules.append("lm_head")
            return
        self.module_paths["lm_head"] = path

        def hook(_module, args, kwargs=None):
            hidden = kwargs.get("input") if kwargs else None
            if hidden is None and args:
                hidden = args[0]
            self._append({"hook": "lm_head_pre", "hidden_shape": _shape(hidden)})

        self.handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

    def summary(self) -> dict[str, Any]:
        lm_head_calls = [event for event in self.events if event.get("hook") == "lm_head_pre"]
        norm_events = [event for event in self.events if event.get("hook") == "model_norm"]
        forward_events = [
            event for event in self.events
            if event.get("hook") == "model_forward_pre"
            and event.get("lvr_mode_switch") is not None
        ]
        modes = [
            any(event.get("lvr_mode_switch") or [])
            for event in forward_events
        ]
        if not modes:
            modes = [
                event.get("mode") == "latent"
                for event in self.events
                if event.get("hook") == "embed_tokens_pre"
            ]
        hidden_feedback_steps = [
            event for event in forward_events
            if event.get("last_position_hidden_state_shape") is not None
        ]
        embed_events = [
            event for event in self.events
            if event.get("hook") == "embed_tokens_pre"
        ]
        return {
            "trace_quality": "instrumented_sparse_v0",
            "sampled_layers": self.sampled_layers,
            "module_paths": self.module_paths,
            "missing_modules": self.missing_modules,
            "n_events": len(self.events),
            "mode": modes,
            "n_lvr_mode_steps": int(sum(bool(item) for item in modes)),
            "n_hidden_feedback_steps": len(hidden_feedback_steps),
            "n_captured_latent_states": len(self.captured_states),
            "captured_steps": [int(item["step_index"]) for item in self.captured_states],
            "hidden_size": (
                int(self.captured_states[-1]["shape"][-1])
                if self.captured_states and self.captured_states[-1].get("shape")
                else None
            ),
            "capture_device_policy": self.capture_device_policy,
            "n_patch_applied": int(self.n_patch_applied),
            "n_embed_calls": len(embed_events),
            "lm_head_called": bool(lm_head_calls),
            "lm_head_call_count": len(lm_head_calls),
            "last_hidden_state_shape": (
                norm_events[-1].get("last_hidden_state_shape") if norm_events else None
            ),
            "events": self.events,
        }


class TracedLVRQwenAdapter(LVRQwenAdapter):
    """LVR adapter variant that instruments generation with sparse hooks."""

    def generate_with_trace(self, wrapper, image, question: str, **kwargs) -> dict:
        trace_capture = kwargs.pop("trace_capture", {}) or {}
        capture_tensors = bool(trace_capture.get("capture_tensors", False))
        max_captured = int(trace_capture.get("max_captured_tensors", 16))
        patch_states = trace_capture.get("patch_states")
        patch_steps = trace_capture.get("patch_steps")
        patch_shape_policy = str(trace_capture.get("patch_shape_policy", "strict"))
        trace_v2_cfg = (getattr(wrapper, "cfg", {}) or {}).get("trace_v2", {}) or {}
        require_trace = bool(trace_v2_cfg.get("required", False))
        forbid_fallback = bool(trace_v2_cfg.get("forbid_fallback", require_trace))
        if patch_steps is not None:
            patch_steps = {int(step) for step in patch_steps}
        try:
            with TraceRecorder(
                wrapper,
                capture_tensors=capture_tensors,
                max_captured_tensors=max_captured,
                patch_states=patch_states,
                patch_steps=patch_steps,
                patch_shape_policy=patch_shape_policy,
            ) as recorder:
                trace = super().generate_with_trace(wrapper, image, question, **kwargs)
            sparse = recorder.summary()
            tensor_bank = [
                {k: v for k, v in item.items() if k != "tensor"}
                for item in recorder.captured_states
            ]
            trace.update({
                **sparse,
                "captured_state_metadata": tensor_bank,
                "_captured_states": recorder.captured_states,
                "legacy_trace_quality": trace.get("trace_quality"),
                "trace_quality": sparse["trace_quality"],
                "notes": {
                    **dict(trace.get("notes") or {}),
                    "instrumentation": "sparse_hooks_v0",
                    "fallback_available": True,
                },
            })
            return trace
        except Exception as exc:  # noqa: BLE001
            if patch_states or require_trace or forbid_fallback:
                raise
            logger.warning("trace v2 failed; falling back to legacy trace: %s", exc)
            trace = super().generate_with_trace(wrapper, image, question, **kwargs)
            trace["trace_quality"] = trace.get("trace_quality") or "approx_from_generated_token_ids"
            trace["trace_v2_error"] = repr(exc)
            trace["notes"] = {
                **dict(trace.get("notes") or {}),
                "instrumentation": "sparse_hooks_v0_failed",
                "fallback_available": True,
            }
            return trace
