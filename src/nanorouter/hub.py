"""Montagem da model card e do staging de publicação do adapter no Hugging Face Hub.

A lógica fica aqui, sem depender de `huggingface_hub`, para ser testável sem
credenciais e sem rede. `scripts/publish_adapter.py` faz o upload de fato.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_CARD = Path("docs/adapter_card.md")
# Sufixos que compõem um repo PEFT mínimo: pesos, config do adapter, tokenizer
# e a própria card. Checkpoints intermediários do trainer ficam de fora.
KEEP_SUFFIXES = frozenset({".json", ".safetensors", ".md", ".txt", ".model"})
FALLBACK_TABLE = "| backend | exact_match |\n|---|---|\n| sem run | rode `make eval` |"


def results_table(eval_dir: Path) -> str:
    """Extrai a tabela de backends de `outputs/eval/RESULTS.md`, se ela existir.

    Sem a run de avaliação a tabela não existe, e a card precisa dizer isso em
    vez de exibir número de outro lugar.
    """
    results = Path(eval_dir) / "RESULTS.md"
    if not results.exists():
        return FALLBACK_TABLE

    lines = results.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("| backend")), None)
    if start is None:
        return FALLBACK_TABLE

    table = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.startswith("|"):
            break
        table.append(line)
    return "\n".join(table) if len(table) > 2 else FALLBACK_TABLE


def build_card(
    card_path: Path = DEFAULT_CARD,
    eval_dir: Path = Path("outputs/eval"),
    repo: str = "",
    base_model: str = BASE_MODEL,
    date: str | None = None,
) -> str:
    """Preenche a model card com modelo base, repo, data e a tabela de resultados."""
    template = Template(Path(card_path).read_text(encoding="utf-8"))
    return template.substitute(
        base_model=base_model,
        repo=repo,
        date=date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        table=results_table(eval_dir),
    )


def stage(
    adapter_dir: Path,
    staging_dir: Path,
    card_text: str,
    meta: dict[str, Any],
) -> Path:
    """Copia o adapter para um diretório limpo com card e metadados.

    Só os arquivos que compõem um repo PEFT são copiados: checkpoints
    intermediários do trainer ficam fora, e o staging é descartável.
    """
    adapter = Path(adapter_dir)
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"adapter não encontrado em {adapter}. Rode `make train` antes de publicar."
        )

    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    for item in sorted(adapter.iterdir()):
        if item.is_file() and item.suffix in KEEP_SUFFIXES:
            shutil.copy2(item, staging / item.name)

    (staging / "README.md").write_text(card_text, encoding="utf-8")
    (staging / "publish_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return staging
