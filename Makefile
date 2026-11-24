PY ?= python
MODEL ?= Qwen/Qwen2.5-0.5B-Instruct
N ?= 6000
HF_REPO ?= yahbeaumont/nanorouter-lora
ADAPTER_URL ?= https://github.com/yahbeaumont/nanorouter/releases/download/v0.1.0/nanorouter-lora-cpu-400.zip

.PHONY: help setup data train eval export serve test lint fmt publish publish-dry fetch-adapter clean all

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## instala deps de dev
	$(PY) -m pip install -e ".[train,eval,export,serve,dev]"

data: ## gera dataset sintético (train/val/test)
	$(PY) -m nanorouter.cli data --out data --n $(N) --seed 1337

train: ## fine-tuna com LoRA
	$(PY) -m nanorouter.cli train --config configs/lora_sft.yaml --data data/train.jsonl

eval: ## avalia checkpoints vs. baselines
	$(PY) -m nanorouter.cli eval --suite data/test.jsonl --out outputs/eval --plot

export: ## merge + GGUF + ONNX
	$(PY) -m nanorouter.cli export --out outputs/export

serve: ## sobe API local
	$(PY) -m nanorouter.cli serve --port 8000

publish: ## publica o adapter no Hugging Face Hub
	$(PY) -m pip install -q "huggingface_hub>=0.24"
	$(PY) scripts/publish_adapter.py --repo $(HF_REPO) --eval-dir outputs/eval

publish-dry: ## monta o staging da model card sem enviar nada
	$(PY) -m pip install -q "huggingface_hub>=0.24"
	$(PY) scripts/publish_adapter.py --repo $(HF_REPO) --eval-dir outputs/eval --dry-run

fetch-adapter: ## baixa o adapter de referência da release
	curl -fsSL -o /tmp/nanorouter-lora.zip "$(ADAPTER_URL)"
	mkdir -p checkpoints/lora
	unzip -oq /tmp/nanorouter-lora.zip -d checkpoints/lora
	@echo "adapter em checkpoints/lora/final"

test: ## roda testes
	$(PY) -m pytest

lint: ## ruff + mypy
	$(PY) -m ruff check src tests scripts && $(PY) -m mypy src

fmt: ## formata
	$(PY) -m ruff format src tests scripts && $(PY) -m ruff check --fix src tests scripts

all: data train eval ## pipeline completo

clean: ## remove artefatos
	rm -rf data outputs checkpoints runs .pytest_cache .ruff_cache .mypy_cache
