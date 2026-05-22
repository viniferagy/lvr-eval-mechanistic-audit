"""Static model go/no-go probes for Week 2 adapter planning."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AdapterProbe:
    model_id: str
    display_name: str
    public_weights: str
    hookability: str
    main_pool_status: str
    rationale: str
    next_action: str


PROBES: tuple[AdapterProbe, ...] = (
    AdapterProbe(
        model_id="monet",
        display_name="Monet",
        public_weights="unverified",
        hookability="unknown_adapter_surface",
        main_pool_status="no_go_week2",
        rationale="Do not enter main pool until local public weights and decoder-layer hooks are verified.",
        next_action="Record exact checkpoint path and run a no-forward architecture probe before GPU audit.",
    ),
    AdapterProbe(
        model_id="latent_sketchpad",
        display_name="Latent Sketchpad",
        public_weights="unverified",
        hookability="likely_custom_loop",
        main_pool_status="no_go_week2",
        rationale="Latent interface may not expose Qwen-style placeholder spans or standard decoder layers.",
        next_action="Inspect model source for latent-token lifecycle and map span semantics before integration.",
    ),
    AdapterProbe(
        model_id="crystal",
        display_name="CrystaL",
        public_weights="unverified",
        hookability="unknown_adapter_surface",
        main_pool_status="no_go_week2",
        rationale="Needs public-weight availability and hook path confirmation before causal metrics are meaningful.",
        next_action="Add candidate config only after weights, processor, final norm, lm_head, and layers are discoverable.",
    ),
)


def list_adapter_probes() -> list[dict]:
    return [asdict(probe) for probe in PROBES]


def validate_adapter_probes(probes: list[dict] | None = None) -> dict:
    rows = list_adapter_probes() if probes is None else probes
    required = {
        "model_id",
        "display_name",
        "public_weights",
        "hookability",
        "main_pool_status",
        "rationale",
        "next_action",
    }
    missing = {
        str(row.get("model_id", f"row_{idx}")): sorted(required - set(row))
        for idx, row in enumerate(rows)
        if required - set(row)
    }
    return {
        "n_models": len(rows),
        "model_ids": [row["model_id"] for row in rows if "model_id" in row],
        "missing_fields": missing,
        "all_complete": not missing,
        "main_pool_ready": [
            row["model_id"]
            for row in rows
            if row.get("main_pool_status") in {"go", "go_week2"}
        ],
    }
