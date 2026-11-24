"""API FastAPI. Mesmo prompt do treino, validação no retorno.

Configuração por variável de ambiente, para o container subir em qualquer
máquina sem editar código:

    NANOROUTER_SPEC      caminho do tool spec (default: configs/tools.yaml)
    NANOROUTER_MODEL     modelo base HF (default: Qwen/Qwen2.5-0.5B-Instruct)
    NANOROUTER_ADAPTER   caminho/repo do adapter LoRA (opcional)
    NANOROUTER_GGUF      caminho de um .gguf; se presente, usa llama.cpp
    NANOROUTER_API_URL   endpoint OpenAI-compatible; se presente, usa HTTP

Sem adapter, o servidor sobe com o modelo base: serve para fumaça e para medir
o baseline, não para produção.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .infer import route
from .toolspec import load_spec, spec_fingerprint

STATE: dict[str, Any] = {}


def _adapter_exists(adapter: str) -> bool:
    """True se for diretório local existente ou um repo do Hugging Face."""
    local = Path(adapter)
    if local.exists():
        return True
    return "/" in adapter and not adapter.startswith((".", "/"))


def _build_backend() -> tuple[Any, str]:
    from . import backends

    gguf = os.getenv("NANOROUTER_GGUF")
    if gguf:
        return backends.LlamaCppBackend(gguf), "gguf"

    api_url = os.getenv("NANOROUTER_API_URL")
    if api_url:
        return (
            backends.OpenAICompatBackend(
                model=os.getenv("NANOROUTER_API_MODEL", "local"),
                base_url=api_url,
                api_key=os.getenv("NANOROUTER_API_KEY", "local"),
            ),
            "api",
        )

    model = os.getenv("NANOROUTER_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    adapter = os.getenv("NANOROUTER_ADAPTER") or None
    if adapter and not _adapter_exists(adapter):
        adapter = None
    return backends.HFBackend(model, adapter=adapter), "hf"


@asynccontextmanager
async def lifespan(app: FastAPI):
    spec_path = os.getenv("NANOROUTER_SPEC", "configs/tools.yaml")
    STATE["spec"] = load_spec(spec_path)
    STATE["backend"], STATE["kind"] = _build_backend()
    yield
    STATE.clear()


app = FastAPI(title="nanorouter", version="0.1.0", lifespan=lifespan)


class RouteRequest(BaseModel):
    utterance: str


@app.get("/health")
def health() -> dict[str, Any]:
    spec = STATE.get("spec")
    return {
        "status": "ok" if spec else "loading",
        "backend": getattr(STATE.get("backend"), "name", None),
        "kind": STATE.get("kind"),
        "spec_fingerprint": spec_fingerprint(spec) if spec else None,
    }


@app.post("/route")
def do_route(req: RouteRequest) -> dict[str, Any]:
    if not req.utterance.strip():
        raise HTTPException(422, "utterance vazia")
    if "backend" not in STATE:
        raise HTTPException(503, "backend ainda carregando")
    return route(STATE["backend"], STATE["spec"], req.utterance)
