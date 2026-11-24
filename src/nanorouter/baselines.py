"""Baselines de prompt para comparar contra o modelo fine-tunado.

O baseline few-shot é um `Backend` que envolve outro `Backend`: assim ele
herda a medição de latência e não precisa mexer nos métodos do objeto interno.
"""

from __future__ import annotations

import json
from typing import Any

from .backends import Backend
from .prompts import build_messages, gold_completion

__all__ = ["FewShotBackend", "constrained_prompt", "schema_only_prompt", "build_messages"]


class FewShotBackend(Backend):
    """Injeta k exemplos rotulados antes do enunciado do usuário."""

    def __init__(self, inner: Backend, spec: dict[str, Any], examples: list[dict[str, Any]]):
        self.inner = inner
        self.name = f"fewshot-{len(examples)}:{inner.name}"
        shots = "\n\n".join(
            f"EXEMPLO {i + 1}\nEntrada: {ex['utterance']}\nSaída: {gold_completion(ex)}"
            for i, ex in enumerate(examples)
        )
        self.prefix = f"Exemplos de referência:\n\n{shots}\n\n"

    def complete(self, spec: dict[str, Any], utterance: str) -> str:
        return self.inner.complete(spec, f"{self.prefix}Entrada: {utterance}")


def constrained_prompt(spec: dict[str, Any]) -> dict[str, Any]:
    """Spec com JSON Schema embutido, para constrained decoding."""
    from .schema import RouteDecision

    enriched = dict(spec)
    enriched["json_schema"] = RouteDecision.model_json_schema()
    return enriched


def schema_only_prompt(spec: dict[str, Any]) -> str:
    return json.dumps(spec.get("json_schema", {}), ensure_ascii=False)[:2000]
