from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanorouter.hub import FALLBACK_TABLE, build_card, results_table, stage

CARD = """---
base_model: $base_model
---

# card

repo: $repo
date: $date

$table
"""

RESULTS = """# Resultados

| backend | exact_match | tool_acc | spec_valid | arg_f1 | p95_ms |
|---|---|---|---|---|---|
| base-0.5b | 0.412 | 0.688 | 0.790 | 0.604 | 812 |
| lora-0.5b | 0.901 | 0.947 | 0.998 | 0.918 | 36 |

Texto depois da tabela.
"""


def test_results_table_without_eval_returns_fallback(tmp_path: Path) -> None:
    assert results_table(tmp_path) == FALLBACK_TABLE


def test_results_table_without_backend_table_returns_fallback(tmp_path: Path) -> None:
    (tmp_path / "RESULTS.md").write_text("# Resultados\n\nnada aqui\n", encoding="utf-8")
    assert results_table(tmp_path) == FALLBACK_TABLE


def test_results_table_extracts_only_backend_rows(tmp_path: Path) -> None:
    (tmp_path / "RESULTS.md").write_text(RESULTS, encoding="utf-8")
    table = results_table(tmp_path)

    assert table.startswith("| backend | exact_match")
    assert "| lora-0.5b | 0.901 |" in table
    assert "Texto depois" not in table
    assert len(table.splitlines()) == 4


def test_build_card_substitutes_every_placeholder(tmp_path: Path) -> None:
    card = tmp_path / "card.md"
    card.write_text(CARD, encoding="utf-8")

    out = build_card(
        card,
        tmp_path / "eval",
        "yahbeaumont/nanorouter-lora",
        "Qwen/Qwen2.5-0.5B-Instruct",
        date="2026-11-24",
    )

    assert "$" not in out
    assert "yahbeaumont/nanorouter-lora" in out
    assert "2026-11-24" in out
    assert FALLBACK_TABLE in out


def test_build_card_uses_eval_table_when_present(tmp_path: Path) -> None:
    card = tmp_path / "card.md"
    card.write_text(CARD, encoding="utf-8")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "RESULTS.md").write_text(RESULTS, encoding="utf-8")

    out = build_card(card, eval_dir, "user/repo", "base", date="2026-01-01")

    assert "| lora-0.5b | 0.901 |" in out
    assert FALLBACK_TABLE not in out


def test_stage_requires_adapter_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        stage(tmp_path / "vazio", tmp_path / "staging", "card", {})


def test_stage_copies_adapter_and_writes_card(tmp_path: Path) -> None:
    adapter = tmp_path / "final"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00")
    (adapter / "spec_fingerprint.json").write_text("{}", encoding="utf-8")
    (adapter / "training_args.bin").write_bytes(b"\x00")
    (adapter / "tokenizer.json").write_text("{}", encoding="utf-8")

    staging = stage(
        adapter,
        tmp_path / "staging",
        "# card",
        {"base_model": "base", "repo": "user/repo"},
    )

    names = sorted(p.name for p in staging.iterdir())
    assert names == [
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "publish_meta.json",
        "spec_fingerprint.json",
        "tokenizer.json",
    ]
    assert (staging / "README.md").read_text(encoding="utf-8") == "# card"
    meta = json.loads((staging / "publish_meta.json").read_text(encoding="utf-8"))
    assert meta["repo"] == "user/repo"
