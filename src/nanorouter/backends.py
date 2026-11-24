"""Backends de inferência: transformers, llama.cpp (GGUF) e OpenAI-compatible."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any

from .prompts import build_messages


def resolve_device(device: str) -> str:
    """Traduz `auto` para um device concreto: `torch.Tensor.to("auto")` não existe."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(cls: Any, model: str, dtype: Any) -> Any:
    """Carrega o modelo aceitando as duas grafias do argumento de precisão.

    `torch_dtype` é o nome histórico; `dtype` passou a ser aceito nas versões
    novas do transformers. O projeto declara `transformers>=4.44`, então as duas
    precisam funcionar.
    """
    try:
        return cls.from_pretrained(model, dtype=dtype)
    except TypeError:
        return cls.from_pretrained(model, torch_dtype=dtype)


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def complete(self, spec: dict[str, Any], utterance: str) -> str: ...

    def route(self, spec: dict[str, Any], utterance: str) -> tuple[str, float]:
        start = time.perf_counter()
        text = self.complete(spec, utterance)
        return text, (time.perf_counter() - start) * 1000


class HFBackend(Backend):
    def __init__(self, model: str, adapter: str | None = None, device: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf:{model}" + (f"+{adapter}" if adapter else "")
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        import torch

        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = load_model(AutoModelForCausalLM, model, dtype)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model = self.model.to(self.device).eval()

    def complete(self, spec: dict[str, Any], utterance: str) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            build_messages(spec, utterance), tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # `decode` é tipado como `str | list[str]` nas versões novas do transformers.
        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return text if isinstance(text, str) else "".join(text)


class LlamaCppBackend(Backend):
    """GGUF via llama-cpp-python. Requer `pip install llama-cpp-python`."""

    def __init__(self, gguf_path: str, n_ctx: int = 2048, threads: int = 4):
        from llama_cpp import Llama

        self.name = f"gguf:{os.path.basename(gguf_path)}"
        self.llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=threads, verbose=False)

    def complete(self, spec: dict[str, Any], utterance: str) -> str:
        from typing import cast

        out = self.llm.create_chat_completion(
            messages=cast(Any, build_messages(spec, utterance)),
            max_tokens=192,
            temperature=0.0,
        )
        if not isinstance(out, dict):
            raise RuntimeError("llama.cpp devolveu stream inesperado")
        return out["choices"][0]["message"]["content"] or ""


class OpenAICompatBackend(Backend):
    """Qualquer endpoint /v1/chat/completions (vLLM, Ollama, TGI, OpenAI)."""

    def __init__(self, model: str, base_url: str, api_key: str = "local"):
        from openai import OpenAI

        self.name = f"api:{model}"
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, spec: dict[str, Any], utterance: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=build_messages(spec, utterance),
            temperature=0.0,
            max_tokens=192,
        )
        return resp.choices[0].message.content or ""
