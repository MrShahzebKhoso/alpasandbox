"""(De)serialization of model outputs, shared by the HTTP API and the stdio worker."""
from __future__ import annotations

import numpy as np

from ..core import PolicyOutput, QueryResult


def output_to_dict(out: PolicyOutput) -> dict:
    return {
        "waypoints": np.asarray(out.waypoints, float).round(4).tolist(), "dt": out.dt,
        "reasoning": out.reasoning, "latency_ms": out.latency_ms, "meta": out.meta,
        "candidates": [np.asarray(c, float).round(4).tolist() for c in out.candidates],
        "candidate_reasoning": list(out.candidate_reasoning),
    }


def output_from_dict(d: dict, latency_ms: float | None = None) -> PolicyOutput:
    meta = dict(d.get("meta", {}))
    if latency_ms is not None:
        meta.setdefault("model_latency_ms", d.get("latency_ms"))
    return PolicyOutput(
        np.asarray(d["waypoints"], float), float(d.get("dt", 0.1)), d.get("reasoning", ""),
        float(latency_ms if latency_ms is not None else d.get("latency_ms", 0.0)), meta,
        [np.asarray(c, float) for c in d.get("candidates", [])], list(d.get("candidate_reasoning", [])))


def query_to_dict(q: QueryResult) -> dict:
    return {"answer": q.answer, "latency_ms": q.latency_ms, "meta": q.meta}


def query_from_dict(d: dict, latency_ms: float | None = None) -> QueryResult:
    meta = dict(d.get("meta", {}))
    if latency_ms is not None:
        meta.setdefault("model_latency_ms", d.get("latency_ms"))
    return QueryResult(d.get("answer", ""), float(latency_ms if latency_ms is not None else d.get("latency_ms", 0)),
                       meta)
