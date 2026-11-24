"""Testes das decisões de carregamento que não exigem torch instalado."""

from __future__ import annotations

from typing import Any

import pytest

from nanorouter.backends import load_model, resolve_device


class ModelWithDtype:
    @classmethod
    def from_pretrained(cls, model: str, dtype: Any = None) -> tuple[str, Any]:
        return (model, dtype)


class ModelWithTorchDtype:
    @classmethod
    def from_pretrained(cls, model: str, torch_dtype: Any = None) -> tuple[str, Any]:
        return (model, torch_dtype)


class ModelThatRejectsBoth:
    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> tuple[str, Any]:
        raise TypeError("nenhum argumento de precisão aceito")


def test_load_model_uses_dtype_when_supported() -> None:
    assert load_model(ModelWithDtype, "base", "fp32") == ("base", "fp32")


def test_load_model_falls_back_to_torch_dtype() -> None:
    assert load_model(ModelWithTorchDtype, "base", "fp32") == ("base", "fp32")


def test_load_model_propagates_unrelated_type_errors() -> None:
    with pytest.raises(TypeError):
        load_model(ModelThatRejectsBoth, "base", "fp32")


def test_resolve_device_keeps_explicit_device() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


def test_resolve_device_maps_auto_to_concrete_device() -> None:
    assert resolve_device("auto") in {"cpu", "cuda"}
