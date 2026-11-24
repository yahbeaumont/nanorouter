"""Harness de avaliação. Roda N backends contra o mesmo suite, salva JSONL + markdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .metrics import compute
from .schema import parse_decision

console = Console()


def run_suite(
    backend, spec: dict[str, Any], suite: list[dict[str, Any]], limit: int | None = None
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    items = suite[:limit] if limit else suite
    for item in items:
        text, latency = backend.route(spec, item["utterance"])
        pred, error = parse_decision(text)
        rows.append(
            {
                "id": item["id"],
                "utterance": item["utterance"],
                "raw": text,
                "pred": pred,
                "gold": item,
                "parse_error": error,
                "latency_ms": latency,
            }
        )
    return {"backend": backend.name, "rows": rows, "metrics": compute(rows, spec)}


def compare(results: list[dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    table = Table(title="Comparação de backends", show_lines=False)
    cols = ["backend", "exact", "tool_acc", "spec_valid", "arg_f1", "p95_ms"]
    for c in cols:
        table.add_column(c, justify="right" if c != "backend" else "left")

    lines = [
        "# Resultados\n",
        "Gerado por `nanorouter eval`. Reproduza com `make eval`.\n",
        "| backend | exact_match | tool_acc | spec_valid | arg_f1 | p95_ms |",
        "|---|---|---|---|---|---|",
    ]

    for r in results:
        m = r["metrics"]
        table.add_row(
            r["backend"],
            f"{m['exact_match']:.3f}",
            f"{m['tool_accuracy']:.3f}",
            f"{m['spec_validity']:.3f}",
            f"{m['arg_f1']:.3f}",
            f"{m['latency_p95_ms']:.0f}",
        )
        lines.append(
            f"| {r['backend']} | {m['exact_match']:.3f} | {m['tool_accuracy']:.3f} | "
            f"{m['spec_validity']:.3f} | {m['arg_f1']:.3f} | {m['latency_p95_ms']:.0f} |"
        )

    console.print(table)

    for r in results:
        slug = r["backend"].replace(":", "_").replace("/", "_").replace("+", "_")
        path = out_dir / f"{slug}.jsonl"
        with path.open("w") as fh:
            for row in r["rows"]:
                pred = row["pred"]
                fh.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "utterance": row["utterance"],
                            "gold": row["gold"]["tool"],
                            "pred": pred.tool if pred else None,
                            "parse_error": row["parse_error"],
                            "latency_ms": round(row["latency_ms"], 1),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    md = out_dir / "RESULTS.md"
    md.write_text("\n".join(lines) + "\n")
    return md
