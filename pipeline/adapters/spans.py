"""Audit-relevant token span types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    kind: str

    def as_slice(self) -> slice:
        return slice(self.start, self.end)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class AuditSpans:
    image_tokens: TokenSpan
    question_tokens: Optional[TokenSpan] = None
    lvr_placeholder_tokens: Optional[TokenSpan] = None
    latent_tokens: Optional[TokenSpan] = None
    answer_probe_pos: Optional[int] = None
    notes: Optional[dict] = None

    def preferred_query_span(self) -> TokenSpan:
        """Preferred query span for internal attention/readout metrics."""
        if self.latent_tokens is not None:
            return self.latent_tokens
        if self.lvr_placeholder_tokens is not None:
            return self.lvr_placeholder_tokens
        if self.answer_probe_pos is not None:
            return TokenSpan(
                self.answer_probe_pos,
                self.answer_probe_pos + 1,
                "answer_probe_pos",
            )
        raise ValueError("No valid query span: latent/lvr/answer_probe_pos missing.")


def span_to_dict(span: TokenSpan | None) -> dict | None:
    if span is None:
        return None
    return {"start": int(span.start), "end": int(span.end), "kind": span.kind}


def spans_to_metadata(spans: AuditSpans, query_span: TokenSpan | None = None) -> dict:
    query = query_span or spans.preferred_query_span()
    return {
        "query_target_kind": query.kind,
        "query_span": [int(query.start), int(query.end)],
        "image_span": [int(spans.image_tokens.start), int(spans.image_tokens.end)],
        "spans": {
            "image_tokens": span_to_dict(spans.image_tokens),
            "question_tokens": span_to_dict(spans.question_tokens),
            "lvr_placeholder_tokens": span_to_dict(spans.lvr_placeholder_tokens),
            "latent_tokens": span_to_dict(spans.latent_tokens),
            "answer_probe_pos": spans.answer_probe_pos,
        },
        "adapter_notes": dict(spans.notes or {}),
    }
