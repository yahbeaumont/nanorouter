"""Gráficos da avaliação: matriz de confusão por backend.

Fica no pacote (e não em `scripts/`) porque o `nanorouter eval --plot` precisa
importá-lo; importar de um diretório fora do pacote quebra quando a CLI é
instalada via pip.
"""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import confusion


def plot(eval_dir: str | Path) -> list[Path]:
    """Gera uma matriz de confusão por arquivo `.jsonl` em `eval_dir`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    eval_dir = Path(eval_dir)
    written: list[Path] = []

    for path in sorted(eval_dir.glob("*.jsonl")):
        pairs = [
            (row["pred"] or "<parse_error>", row["gold"])
            for row in (json.loads(line) for line in path.read_text().splitlines() if line)
        ]
        if not pairs:
            continue
        labels, matrix = confusion(pairs)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(np.array(matrix), cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predito")
        ax.set_ylabel("ouro")
        ax.set_title(path.stem)
        fig.tight_layout()

        out = eval_dir / f"{path.stem}-confusion.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        written.append(out)

    return written
