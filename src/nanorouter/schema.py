"""Contrato de saída, parsing tolerante e validação contra o tool spec."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

NO_TOOL = "no_tool"

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


class RouteDecision(BaseModel):
    """Decisão de roteamento. `extra=forbid` impede o modelo de inventar campos."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False, sort_keys=True)


def parse_json_loose(text: str) -> dict[str, Any] | None:
    """Extrai o primeiro objeto JSON do output bruto. Retorna None se não houver."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJ.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def parse_decision(text: str) -> tuple[RouteDecision | None, str | None]:
    """Parseia o output do modelo. Retorna (decisão, erro)."""
    raw = parse_json_loose(text)
    if raw is None:
        return None, "no_json_object"
    try:
        return RouteDecision(**raw), None
    except ValidationError as exc:
        return None, f"schema:{exc.error_count()}_violations"


def validate_against_spec(decision: RouteDecision, spec: dict[str, Any]) -> list[str]:
    """Valida a decisão contra o tool spec. Lista vazia = válida."""
    errors: list[str] = []
    tools = {t["name"]: t for t in spec["tools"]}

    if decision.tool == NO_TOOL:
        if decision.args:
            errors.append("no_tool_with_args")
        return errors

    if decision.tool not in tools:
        return [f"unknown_tool:{decision.tool}"]

    tool = tools[decision.tool]
    slots = tool.get("slots", {})

    for name, slot in slots.items():
        if slot.get("required") and name not in decision.args:
            errors.append(f"missing_required:{name}")

    for name, value in decision.args.items():
        if name not in slots:
            errors.append(f"unknown_slot:{name}")
            continue
        errors.extend(_check_slot_value(name, value, slots[name]))

    return errors


def _check_slot_value(name: str, value: Any, slot: dict[str, Any]) -> list[str]:
    kind = slot.get("type")
    if kind == "int" and not isinstance(value, int):
        return [f"bad_type:{name}"]
    if kind == "float" and not isinstance(value, int | float):
        return [f"bad_type:{name}"]
    if kind == "enum" and value not in slot.get("values", []):
        return [f"bad_enum:{name}"]
    if kind == "iata" and not (isinstance(value, str) and re.fullmatch(r"[A-Z]{3}", value)):
        return [f"bad_iata:{name}"]
    if kind == "date" and not (
        isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
    ):
        return [f"bad_date:{name}"]
    if kind == "currency" and not (isinstance(value, str) and re.fullmatch(r"[A-Z]{3}", value)):
        return [f"bad_currency:{name}"]
    if kind == "int" and "min" in slot and isinstance(value, int) and value < slot["min"]:
        return [f"below_min:{name}"]
    if kind == "int" and "max" in slot and isinstance(value, int) and value > slot["max"]:
        return [f"above_max:{name}"]
    return []
