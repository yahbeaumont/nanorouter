#!/usr/bin/env python3
"""Wrapper de linha de comando para os gráficos de avaliação.

A implementação vive em `nanorouter.plotting` para que `nanorouter eval --plot`
funcione também com o pacote instalado.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanorouter.plotting import plot  # noqa: E402

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/eval")
    for written in plot(target):
        print(written)
