"""Geração de dataset sintético.

Três ingredientes fazem o dataset valer algo:
1. paráfrases multilíngues por template,
2. fuzzing de slots com valores realistas,
3. hard negatives — enunciados que citam palavras-chave de uma tool mas
   pertencem a outra, e enunciados que não pertencem a nenhuma.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any

from .schema import NO_TOOL

CITIES = [
    ("São Paulo", "GRU"),
    ("Rio de Janeiro", "GIG"),
    ("Lisboa", "LIS"),
    ("Porto", "OPO"),
    ("Madri", "MAD"),
    ("Buenos Aires", "EZE"),
    ("Santiago", "SCL"),
    ("Bogotá", "BOG"),
    ("Miami", "MIA"),
    ("Nova York", "JFK"),
    ("Paris", "CDG"),
    ("Tóquio", "HND"),
]
CURRENCIES = ["BRL", "USD", "EUR", "GBP", "JPY", "ARS"]
PEOPLE = ["Ana Souza", "Bruno Lima", "Carla Dias", "Diego Alves", "Elisa Rocha"]
BOOKING_IDS = [f"BK{1000 + i}" for i in range(400)]
FLIGHT_IDS = [f"FL{2000 + i}" for i in range(400)]

TEMPLATES: dict[str, list[str]] = {
    "search_flights": [
        "quero um voo de {origin_city} para {destination_city} em {date}",
        "voos {origin_iata}-{destination_iata} dia {date} para {passengers} pessoa(s)",
        "preciso ir de {origin_city} a {destination_city} no dia {date}",
        "busca passagem aérea {origin_iata} para {destination_iata} em {date}, classe {cabin}",
        "i need a flight from {origin_iata} to {destination_iata} on {date}",
        "necesito un vuelo de {origin_city} a {destination_city} el {date}",
    ],
    "search_hotels": [
        "hotéis em {city} de {date} a {date2} para {guests} hóspedes",
        "quero me hospedar em {city} entre {date} e {date2}",
        "hospedagem em {city} com diária até {max_price} reais",
        "hotel em {city} para {guests} pessoas, check-in {date}, check-out {date2}",
        "find a hotel in {city} from {date} to {date2}",
    ],
    "book_flight": [
        "reserva o voo {flight_id} para {person}",
        "quero comprar o voo {flight_id}, passageiro {person}",
        "confirma o {flight_id} no nome de {person}, assento {seat}",
        "book flight {flight_id} for {person}",
    ],
    "cancel_booking": [
        "cancela a reserva {booking_id}",
        "quero cancelar o {booking_id}, motivo: {reason}",
        "desfaz a reserva {booking_id}",
        "cancel booking {booking_id}",
    ],
    "booking_status": [
        "qual o status da reserva {booking_id}?",
        "como está o {booking_id}",
        "me diz a situação da reserva {booking_id}",
        "status of booking {booking_id}",
    ],
    "get_weather": [
        "como vai estar o tempo em {city} em {date}?",
        "previsão do tempo para {city} no dia {date}",
        "vai chover em {city} em {date}?",
        "weather forecast for {city} on {date}",
    ],
    "convert_currency": [
        "converte {amount} {from_currency} para {to_currency}",
        "quanto dá {amount} {from_currency} em {to_currency}?",
        "cotação de {from_currency} para {to_currency}, valor {amount}",
        "convert {amount} {from_currency} to {to_currency}",
    ],
    "search_poi": [
        "pontos turísticos de categoria {category} em {city}",
        "museus e {category} perto de {city}, raio {radius_km} km",
        "o que visitar em {city} na categoria {category}",
        "find {category} near {city}",
    ],
}

NO_TOOL_UTTERANCES = [
    # social
    "oi, tudo bem?",
    "bom dia",
    "boa tarde",
    "boa noite",
    "tchau",
    "obrigado, era só isso",
    "valeu, ajudou bastante",
    "tudo certo por aí?",
    "você é um robô?",
    "conta uma piada",
    "bom fim de semana",
    "hi, how are you?",
    "thanks, that's all",
    "hola, ¿qué tal?",
    # fora de escopo
    "qual a capital da Austrália?",
    "me explica o que é inflação",
    "quem escreveu Dom Casmurro?",
    "quanto é 17 vezes 43?",
    "como faço um bolo de cenoura?",
    "qual a previsão do dólar para 2030?",
    "me indica um filme bom",
    "traduz 'bom dia' para o japonês",
    "what's the weather on Mars?",
    "explain quantum entanglement",
    # incompleto ou ambíguo
    "preciso de ajuda",
    "e agora?",
    "quero viajar",
    "me ajuda com isso",
    "pode ser amanhã",
    "não sei ainda",
    "depois eu vejo",
    "isso aí",
    "como assim?",
    "não entendi",
]

# Cada enunciado cita a superfície lexical de uma tool vizinha (voo, tempo,
# reserva, câmbio) mas pertence a outra. São templates, não frases fixas: com
# dez frases fixas o dataset viraria dez exemplos repetidos cem vezes, e a
# métrica de generalização passaria a medir memorização.
HARD_NEGATIVES: list[tuple[str, str]] = [
    ("quanto custa um voo para {destination_city}?", "search_flights"),
    ("tem voo direto de {origin_city} para {destination_city}?", "search_flights"),
    ("como está o tempo em {city}?", "get_weather"),
    ("vai fazer sol em {city} no dia {date}?", "get_weather"),
    ("quero cancelar, meu código é {booking_id}", "cancel_booking"),
    ("o {booking_id} já foi emitido?", "booking_status"),
    ("quanto é {amount} {from_currency} em {to_currency}?", "convert_currency"),
    ("tem museu bom em {city}?", "search_poi"),
    ("hotel barato em {city} para {guests} pessoas", "search_hotels"),
    ("reserva o {flight_id} para {person}", "book_flight"),
]

DISTRACTORS = [
    "e o voo?",
    "e o hotel?",
    "quanto fica?",
    "para duas pessoas",
    "mesma data de antes",
    "pode ser mais barato",
    "e o tempo lá?",
]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _slot_values(rng: random.Random, spec_tool: dict[str, Any]) -> dict[str, Any]:
    """Fuzzing de slots com valores plausíveis por tipo."""
    out: dict[str, Any] = {}
    base = date(2026, 12, 1)
    for name, slot in spec_tool.get("slots", {}).items():
        kind = slot["type"]
        if kind == "iata":
            out[name] = rng.choice(CITIES)[1]
        elif kind == "city":
            out[name] = rng.choice(CITIES)[0]
        elif kind == "date":
            out[name] = (base + timedelta(days=rng.randint(0, 120))).isoformat()
        elif kind == "currency":
            out[name] = rng.choice(CURRENCIES)
        elif kind == "enum":
            out[name] = rng.choice(slot["values"])
        elif kind == "int":
            out[name] = rng.randint(slot.get("min", 1), slot.get("max", 9))
        elif kind == "float":
            out[name] = round(rng.uniform(50, 900), 2)
        elif kind == "person":
            out[name] = rng.choice(PEOPLE)
        elif kind == "string" and name == "booking_id":
            out[name] = rng.choice(BOOKING_IDS)
        elif kind == "string" and name == "flight_id":
            out[name] = rng.choice(FLIGHT_IDS)
        elif kind == "string":
            out[name] = rng.choice(["remarcação", "imprevisto", "mudança de planos"])
    return out


def _render(rng: random.Random, template: str, slots: dict[str, Any]) -> str:
    """Preenche o template com o contexto do registro.

    Cada template usa um subconjunto dos slots, e ferramentas diferentes usam
    nomes diferentes para a mesma ideia (`date` vs. `checkin`/`checkout`).
    Por isso o contexto é montado com fallbacks explícitos: sem eles, um
    template que pede `{date}` numa tool de hospedagem estoura KeyError.
    """
    ctx = dict(slots)
    ctx["origin_city"] = next((c for c, i in CITIES if i == slots.get("origin")), "São Paulo")
    ctx["destination_city"] = next(
        (c for c, i in CITIES if i == slots.get("destination")), "Lisboa"
    )
    ctx["origin_iata"] = slots.get("origin", "GRU")
    ctx["destination_iata"] = slots.get("destination", "LIS")

    base = date(2026, 12, 1) + timedelta(days=rng.randint(0, 120))
    checkin = slots.get("date") or slots.get("checkin") or base.isoformat()
    checkout = (
        slots.get("checkout")
        or (date.fromisoformat(checkin) + timedelta(days=rng.randint(1, 10))).isoformat()
    )
    ctx["date"] = checkin
    ctx["date2"] = checkout

    ctx["max_price"] = slots.get("max_price", 600)
    ctx["radius_km"] = slots.get("radius_km", 5)
    ctx["seat"] = slots.get("seat_pref", "window")
    ctx["reason"] = slots.get("reason", "imprevisto")
    ctx["passengers"] = slots.get("passengers", 1)
    ctx["guests"] = slots.get("guests", 1)
    ctx["cabin"] = slots.get("cabin", "economy")
    ctx["amount"] = slots.get("amount", 100)
    ctx["from_currency"] = slots.get("from_currency", "BRL")
    ctx["to_currency"] = slots.get("to_currency", "USD")
    ctx["city"] = slots.get("city", rng.choice(CITIES)[0])
    ctx.setdefault("person", rng.choice(PEOPLE))
    return template.format(**ctx)


def _required_args(spec_tool: dict[str, Any], slots: dict[str, Any]) -> dict[str, Any]:
    required = {k for k, v in spec_tool.get("slots", {}).items() if v.get("required")}
    return {k: v for k, v in slots.items() if k in required}


def generate(spec: dict[str, Any], n: int, seed: int) -> Iterator[dict[str, Any]]:
    """Gera `n` registros. Distribuição: 70% tool, 12% no_tool, 18% hard negative."""
    rng = _rng(seed)
    tools = spec["tools"]

    for i in range(n):
        roll = rng.random()

        if roll < 0.12:
            yield {
                "id": f"syn-{i:06d}",
                "utterance": rng.choice(NO_TOOL_UTTERANCES),
                "tool": NO_TOOL,
                "args": {},
                "confidence": round(rng.uniform(0.55, 0.95), 2),
                "kind": "no_tool",
            }
            continue

        if roll < 0.30:
            template, tool_name = rng.choice(HARD_NEGATIVES)
            spec_tool = next(t for t in tools if t["name"] == tool_name)
            slots = _slot_values(rng, spec_tool)
            utterance = _render(rng, template, slots)
            yield {
                "id": f"syn-{i:06d}",
                "utterance": utterance,
                "tool": tool_name,
                "args": _required_args(spec_tool, slots),
                "confidence": round(rng.uniform(0.45, 0.75), 2),
                "kind": "hard_negative",
            }
            continue

        spec_tool = rng.choice(tools)
        slots = _slot_values(rng, spec_tool)
        template = rng.choice(TEMPLATES[spec_tool["name"]])
        utterance = _render(rng, template, slots)

        if rng.random() < 0.10:
            utterance = f"{utterance}. {rng.choice(DISTRACTORS)}"

        yield {
            "id": f"syn-{i:06d}",
            "utterance": utterance,
            "tool": spec_tool["name"],
            "args": _required_args(spec_tool, slots),
            "confidence": round(rng.uniform(0.7, 0.99), 2),
            "kind": "tool",
        }


def split(records: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Split estratificado por `kind`, 80/10/10."""
    rng = _rng(seed)
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for kind in ("tool", "no_tool", "hard_negative"):
        subset = [r for r in records if r["kind"] == kind]
        rng.shuffle(subset)
        n = len(subset)
        cut_val, cut_test = int(n * 0.8), int(n * 0.9)
        buckets["train"].extend(subset[:cut_val])
        buckets["val"].extend(subset[cut_val:cut_test])
        buckets["test"].extend(subset[cut_test:])
    for name in buckets:
        rng.shuffle(buckets[name])
    return buckets
