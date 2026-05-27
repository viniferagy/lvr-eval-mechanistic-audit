"""Optional DINO image-backbone sensitivity backend for PF-B."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


BACKEND_VERSION = "dino_pf_b_v0"
DEFAULT_DINO_MODEL = "facebook/dinov2-small"
DEFAULT_CACHE_DIR = ".cache/dino_pf_b"
_BACKENDS: dict[tuple[Any, ...], "BaseDinoBackend"] = {}


class DinoBackendError(RuntimeError):
    """Raised when a requested real DINO backend cannot be loaded or run."""


class BaseDinoBackend:
    backend_name = "base"

    def embed(self, image: Image.Image) -> tuple[np.ndarray, bool]:
        raise NotImplementedError


def _image_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0:
        return arr
    return arr / norm


def _cosine_alignment(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize(a)
    b = _normalize(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    cosine = float(np.dot(a, b) / denom)
    cosine = max(-1.0, min(1.0, cosine))
    return float((cosine + 1.0) / 2.0)


class MockDinoBackend(BaseDinoBackend):
    """Deterministic dependency-free backend used only by smoke tests."""

    backend_name = "mock"

    def embed(self, image: Image.Image) -> tuple[np.ndarray, bool]:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        q25 = np.quantile(arr, 0.25, axis=(0, 1))
        q75 = np.quantile(arr, 0.75, axis=(0, 1))
        dark = np.asarray([(arr.mean(axis=2) < 0.1).mean()], dtype=np.float32)
        bright = np.asarray([(arr.mean(axis=2) > 0.9).mean()], dtype=np.float32)
        return _normalize(np.concatenate([mean, std, q25, q75, dark, bright])), False


class TransformersDinoBackend(BaseDinoBackend):
    backend_name = "transformers_dinov2"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_DINO_MODEL,
        device: str = "cpu",
        cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
        local_files_only: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.local_files_only = local_files_only
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except Exception as exc:  # noqa: BLE001
            raise DinoBackendError(
                "DINO backend requires torch and transformers; install dependencies or use pf_b.dino_backend=mock"
            ) from exc

        self.torch = torch
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            ).eval()
        except Exception as exc:  # noqa: BLE001
            raise DinoBackendError(
                f"could not load DINO model {model_name!r} "
                f"(local_files_only={local_files_only}); cache or download the weights first"
            ) from exc

        self.model.to(device)

    def _cache_key(self, image: Image.Image) -> str:
        h = hashlib.sha256()
        h.update(BACKEND_VERSION.encode("utf-8"))
        h.update(self.backend_name.encode("utf-8"))
        h.update(self.model_name.encode("utf-8"))
        h.update(_image_bytes(image))
        return h.hexdigest()

    def _cache_path(self, image: Image.Image) -> Path:
        return self.cache_dir / f"{self._cache_key(image)}.npy"

    def embed(self, image: Image.Image) -> tuple[np.ndarray, bool]:
        path = self._cache_path(image)
        if path.is_file():
            return np.load(path).astype(np.float32), True

        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            output = self.model(**inputs)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        emb = pooled.detach().float().cpu().numpy()[0].astype(np.float32)
        emb = _normalize(emb)

        tmp_path = path.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(tmp_path, emb)
        os.replace(tmp_path, path)
        return emb, False


def _auto_device(cfg: dict, local: dict) -> str:
    requested = str(local.get("dino_device") or "auto")
    if requested != "auto":
        return requested
    inference_device = str((cfg.get("inference") or {}).get("device") or "cpu")
    try:
        import torch
        if inference_device.startswith("cuda") and torch.cuda.is_available():
            return inference_device
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def get_backend(cfg: dict, local: dict) -> BaseDinoBackend:
    backend_name = str(local.get("dino_backend") or "transformers")
    if backend_name == "mock":
        key = ("mock",)
        if key not in _BACKENDS:
            _BACKENDS[key] = MockDinoBackend()
        return _BACKENDS[key]

    if backend_name not in {"transformers", "transformers_dinov2"}:
        raise DinoBackendError(f"unknown pf_b.dino_backend={backend_name!r}")

    model_name = str(local.get("dino_model") or DEFAULT_DINO_MODEL)
    device = _auto_device(cfg, local)
    cache_dir = str(local.get("dino_cache_dir") or DEFAULT_CACHE_DIR)
    local_files_only = bool(local.get("dino_local_files_only", True))
    key = ("transformers", model_name, device, cache_dir, local_files_only)
    if key not in _BACKENDS:
        _BACKENDS[key] = TransformersDinoBackend(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    return _BACKENDS[key]


def dino_alignment_for_masks(
    *,
    clean_image: Image.Image,
    relevant_image: Image.Image,
    irrelevant_image: Image.Image,
    random_image: Image.Image,
    cfg: dict,
    local: dict,
) -> dict:
    backend = get_backend(cfg, local)
    clean, clean_hit = backend.embed(clean_image)
    relevant, relevant_hit = backend.embed(relevant_image)
    irrelevant, irrelevant_hit = backend.embed(irrelevant_image)
    random, random_hit = backend.embed(random_image)

    rel = _cosine_alignment(clean, relevant)
    irr = _cosine_alignment(clean, irrelevant)
    rnd = _cosine_alignment(clean, random)
    rel_delta = 1.0 - rel
    irr_delta = 1.0 - irr
    rnd_delta = 1.0 - rnd
    selectivity = rel_delta - float(np.mean([irr_delta, rnd_delta]))
    return {
        "requested": True,
        "available": True,
        "backend": backend.backend_name,
        "model": str(local.get("dino_model") or DEFAULT_DINO_MODEL) if backend.backend_name != "mock" else "mock",
        "cache_hits": {
            "clean": bool(clean_hit),
            "relevant": bool(relevant_hit),
            "irrelevant": bool(irrelevant_hit),
            "random": bool(random_hit),
        },
        "dino_alignment": rel,
        "dino_relevant_alignment": rel,
        "dino_irrelevant_alignment": irr,
        "dino_random_alignment": rnd,
        "dino_relevant_delta": rel_delta,
        "dino_irrelevant_delta": irr_delta,
        "dino_random_delta": rnd_delta,
        "dino_region_selectivity": float(selectivity),
    }
