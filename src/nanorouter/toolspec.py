"""Carregamento e serialização do tool spec."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_spec(path: str | Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _validate_spec(spec)
    return spec


def _validate_spec(spec: dict[str, Any]) -> None:
    if "tools" not in spec or not spec["tools"]:
        raise ValueError("spec sem tools")
    names = [t["name"] for t in spec["tools"]]
    if len(names) != len(set(names)):
        raise ValueError("nomes de tool duplicados")
    for tool in spec["tools"]:
        if "name" not in tool or "slots" not in tool:
            raise ValueError(f"tool malformada: {tool}")


def spec_fingerprint(spec: dict[str, Any]) -> str:
    """Hash estável do spec. Amarra checkpoints ao contrato que foram treinados."""
    blob = json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def compact_tool_list(spec: dict[str, Any]) -> str:
    """Lista compacta de tools para o prompt de sistema."""
    lines = []
    for tool in spec["tools"]:
        slots = tool.get("slots", {})
        sig = ", ".join(
            f"{k}:{v['type']}" + ("!" if v.get("required") else "") for k, v in slots.items()
        )
        lines.append(f"- {tool['name']}({sig}) — {tool['description']}")
    return "\n".join(lines)
