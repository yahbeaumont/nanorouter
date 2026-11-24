"""Métricas. Tudo determinístico, sem dependência de LLM-as-judge."""

from __future__ import annotations

import random
from typing import Any

from .schema import RouteDecision, validate_against_spec


def exact_match(pred: RouteDecision, gold: dict[str, Any], tol: float = 0.25) -> bool:
    if pred.tool != gold["tool"]:
        return False
    if pred.args != gold.get("args", {}):
        return False
    return abs(pred.confidence - gold.get("confidence", 0.9)) <= tol


def arg_counts(pred: RouteDecision, gold: dict[str, Any]) -> tuple[int, int, int]:
    """(acertos, preditos, ouros) para F1 de slots."""
    g = gold.get("args", {})
    p = pred.args if pred.tool == gold["tool"] else {}
    tp = sum(1 for k, v in p.items() if g.get(k) == v)
    return tp, len(p), len(g)


def confusion(pairs: list[tuple[str, str]]) -> tuple[list[str], list[list[int]]]:
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for pred, gold in pairs:
        matrix[index[gold]][index[pred]] += 1
    return labels, matrix


def bootstrap_ci(
    values: list[float], iters: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(iters))
    return means[int(iters * alpha / 2)], means[int(iters * (1 - alpha / 2))]


def compute(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    """Agrega todas as métricas de uma rodada de avaliação."""
    n = len(rows)
    tp = pred_args = gold_args = 0
    exact = tool_ok = schema_valid = spec_valid = 0
    latencies: list[float] = []
    pairs: list[tuple[str, str]] = []

    for row in rows:
        gold = row["gold"]
        pred: RouteDecision | None = row["pred"]
        latencies.append(row["latency_ms"])
        pairs.append((pred.tool if pred else "<parse_error>", gold["tool"]))

        if pred is None:
            continue
        schema_valid += 1
        if not validate_against_spec(pred, spec):
            spec_valid += 1
        if pred.tool == gold["tool"]:
            tool_ok += 1
        if exact_match(pred, gold):
            exact += 1
        a, b, c = arg_counts(pred, gold)
        tp, pred_args, gold_args = tp + a, pred_args + b, gold_args + c

    precision = tp / pred_args if pred_args else 0.0
    recall = tp / gold_args if gold_args else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    exact_flags = [1.0 if (r["pred"] and exact_match(r["pred"], r["gold"])) else 0.0 for r in rows]
    lo, hi = bootstrap_ci(exact_flags)

    latencies.sort()
    return {
        "n": n,
        "parse_error_rate": round(1 - schema_valid / n, 4) if n else 0.0,
        "schema_validity": round(schema_valid / n, 4) if n else 0.0,
        "spec_validity": round(spec_valid / n, 4) if n else 0.0,
        "tool_accuracy": round(tool_ok / n, 4) if n else 0.0,
        "exact_match": round(exact / n, 4) if n else 0.0,
        "exact_match_ci95": [round(lo, 4), round(hi, 4)],
        "arg_f1": round(f1, 4),
        "latency_p50_ms": round(latencies[n // 2], 1) if n else 0.0,
        "latency_p95_ms": round(latencies[int(n * 0.95)], 1) if n else 0.0,
    }
