"""Construção de prompts. Formato idêntico em treino, eval e produção."""

from __future__ import annotations

import json
from typing import Any

from .schema import NO_TOOL
from .toolspec import compact_tool_list

SYSTEM_TEMPLATE = """Você é um roteador de intenções. Escolha exatamente uma tool do catálogo
ou `{no_tool}` quando nenhuma se aplicar.

CATÁLOGO:
{tools}

REGRAS:
1. Responda APENAS com um objeto JSON, sem texto antes ou depois.
2. Formato: {{"tool": "<nome>", "args": {{...}}, "confidence": <0.0-1.0>}}
3. Preencha todo slot obrigatório (marcado com `!`).
4. Não invente slots que não existem na tool.
5. Datas em YYYY-MM-DD, aeroportos e moedas em ISO de 3 letras maiúsculas.
6. Se a intenção for ambígua mas houver uma tool dominante, escolha-a com confidence baixa.
7. Conversa social, pergunta fora de escopo ou pedido sem informação suficiente: `{no_tool}`.
"""


def build_system_prompt(spec: dict[str, Any]) -> str:
    return SYSTEM_TEMPLATE.format(tools=compact_tool_list(spec), no_tool=NO_TOOL)


def build_messages(spec: dict[str, Any], utterance: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt(spec)},
        {"role": "user", "content": utterance},
    ]


def gold_completion(record: dict[str, Any]) -> str:
    """Serialização canônica do alvo — a mesma em treino e em eval."""
    payload = {
        "tool": record["tool"],
        "args": record.get("args", {}),
        "confidence": record.get("confidence", 0.9),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
