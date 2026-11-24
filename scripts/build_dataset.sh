#!/usr/bin/env bash
set -euo pipefail
N="${1:-6000}"
python -m nanorouter.cli data --out data --n "$N" --seed 1337
wc -l data/*.jsonl
