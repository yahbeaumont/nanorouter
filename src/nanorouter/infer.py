"""Inferência pontual com validação e reparo opcional."""

from __future__ import annotations

from typing import Any

from .schema import parse_decision, validate_against_spec


def route(backend, spec: dict[str, Any], utterance: str) -> dict[str, Any]:
    text, latency = backend.route(spec, utterance)
    decision, error = parse_decision(text)

    result: dict[str, Any] = {
        "utterance": utterance,
        "backend": backend.name,
        "latency_ms": round(latency, 1),
        "raw": text,
    }

    if decision is None:
        result.update({"ok": False, "error": error})
        return result

    violations = validate_against_spec(decision, spec)
    result.update(
        {
            "decision": decision.model_dump(),
            "ok": not violations,
            "violations": violations,
        }
    )
    return result


def batch(backend, spec: dict[str, Any], utterances: list[str]) -> list[dict[str, Any]]:
    return [route(backend, spec, u) for u in utterances]
