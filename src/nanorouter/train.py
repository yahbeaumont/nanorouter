"""Treino LoRA com TRL. Determinístico, com completion-only loss."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .prompts import build_messages, gold_completion
from .toolspec import spec_fingerprint


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def run(
    config_path: str | Path,
    data_path: str | Path,
    spec_path: str | Path,
    out_dir: str | Path = "checkpoints/lora",
) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from .backends import load_model

    cfg = load_config(config_path)
    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    records = [json.loads(line) for line in Path(data_path).read_text().splitlines() if line]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def render(rec: dict[str, Any]) -> dict[str, str]:
        prompt = tokenizer.apply_chat_template(
            build_messages(spec, rec["utterance"]), tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "completion": gold_completion(rec) + tokenizer.eos_token}

    dataset = Dataset.from_list([render(r) for r in records])

    # O config do Qwen2.5 declara bf16; em CPU sem suporte a bf16 o transformers
    # cai num caminho emulado e o treino fica várias vezes mais lento. Carregar
    # o modelo explicitamente em fp32 evita isso (e o aviso do mkldnn).
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = load_model(
        AutoModelForCausalLM, cfg["model"], torch.bfloat16 if use_bf16 else torch.float32
    )

    lora = cfg["lora"]
    peft_cfg = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    t = cfg["train"]
    # `warmup_ratio` saiu do TrainingArguments no transformers 5.x; `warmup_steps`
    # existe nas duas gerações. A proporção continua sendo a fonte de verdade no
    # YAML, e o número de passos é derivado do tamanho do dataset.
    steps_per_epoch = max(1, len(dataset) // (t["batch_size"] * t["grad_accum"]))
    warmup_steps = int(steps_per_epoch * t["epochs"] * t["warmup_ratio"])

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=t["weight_decay"],
        max_length=t["max_seq_len"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=2,
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        completion_only_loss=t["completion_only_loss"],
        packing=False,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
        peft_config=peft_cfg,
        processing_class=tokenizer,
    )
    trainer.train()

    out = Path(out_dir) / "final"
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    (out / "spec_fingerprint.json").write_text(
        json.dumps(
            {
                "model": cfg["model"],
                "n_records": len(records),
                "seed": cfg["seed"],
                "spec_sha256_12": spec_fingerprint(spec),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(out)
