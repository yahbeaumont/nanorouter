"""Backends de inferência: transformers, llama.cpp (GGUF) e OpenAI-compatible."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any

from .prompts import build_messages


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
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf:{model}" + (f"+{adapter}" if adapter else "")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.bfloat16 if device != "cpu" else torch.float32
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model = self.model.to(device).eval()

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
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )


class LlamaCppBackend(Backend):
    """GGUF via llama-cpp-python. Requer `pip install llama-cpp-python`."""

    def __init__(self, gguf_path: str, n_ctx: int = 2048, threads: int = 4):
        from llama_cpp import Llama

        self.name = f"gguf:{os.path.basename(gguf_path)}"
        self.llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=threads, verbose=False)

    def complete(self, spec: dict[str, Any], utterance: str) -> str:
        out = self.llm.create_chat_completion(
            messages=build_messages(spec, utterance),
            max_tokens=192,
            temperature=0.0,
        )
        return out["choices"][0]["message"]["content"]


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
