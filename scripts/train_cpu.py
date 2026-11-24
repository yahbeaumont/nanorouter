"""Runner de treino para CPU: subset explícito, config derivada e log de progresso.

Não substitui `make train`; existe para registrar exatamente o que foi rodado na
medição publicada (hardware sem GPU, subset e épocas reduzidas).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

N_RECORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
EPOCHS = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 2
ACCUM = int(sys.argv[4]) if len(sys.argv) > 4 else 4
MAX_LEN = int(sys.argv[5]) if len(sys.argv) > 5 else 640
OUT = sys.argv[6] if len(sys.argv) > 6 else "checkpoints/lora_cpu"

from nanorouter.train import run  # noqa: E402

records = [
    line for line in Path("data/train.jsonl").read_text(encoding="utf-8").splitlines() if line
][:N_RECORDS]
Path("data/train_cpu.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")

cfg = yaml.safe_load(Path("configs/lora_sft.yaml").read_text(encoding="utf-8"))
cfg["train"].update(
    {
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "grad_accum": ACCUM,
        "max_seq_len": MAX_LEN,
        "save_steps": 10**9,
        "logging_steps": 5,
        "completion_only_loss": True,
    }
)
Path("configs/lora_cpu.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

start = time.time()
out = run("configs/lora_cpu.yaml", "data/train_cpu.jsonl", "configs/tools.yaml", OUT)
meta = {
    "n_records": N_RECORDS,
    "epochs": EPOCHS,
    "batch_size": BATCH,
    "grad_accum": ACCUM,
    "max_seq_len": MAX_LEN,
    "out": out,
    "seconds": round(time.time() - start),
}
Path(f"{OUT}/run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print("treino concluído:", json.dumps(meta), flush=True)
