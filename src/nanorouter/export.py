"""Merge do adapter, quantização GGUF e export ONNX int8."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def merge_adapter(base_model: str, adapter: str, out_dir: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = out_dir / "merged"
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(merged_dir)
    return merged_dir


def to_gguf(merged_dir: Path, out_dir: Path, quant: str = "q4_k_m") -> Path:
    """Requer llama.cpp clonado e `convert_hf_to_gguf.py` disponível."""
    convert = shutil.which("convert_hf_to_gguf.py")
    if not convert:
        raise RuntimeError(
            "convert_hf_to_gguf.py não encontrado. Clone llama.cpp e adicione ao PATH."
        )
    f16 = out_dir / "model-f16.gguf"
    subprocess.run(
        [convert, str(merged_dir), "--outfile", str(f16), "--outtype", "f16"], check=True
    )
    quantize = shutil.which("llama-quantize")
    if not quantize:
        return f16
    final = out_dir / f"model-{quant}.gguf"
    subprocess.run([quantize, str(f16), str(final), quant.upper()], check=True)
    f16.unlink(missing_ok=True)
    return final


def to_onnx(merged_dir: Path, out_dir: Path) -> Path:
    from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    onnx_dir = out_dir / "onnx"
    model = ORTModelForCausalLM.from_pretrained(merged_dir, export=True)
    model.save_pretrained(onnx_dir)
    AutoTokenizer.from_pretrained(merged_dir).save_pretrained(onnx_dir)

    quantizer = ORTQuantizer.from_pretrained(onnx_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
    quantizer.quantize(save_dir=out_dir / "onnx-int8", quantization_config=qconfig)
    return out_dir / "onnx-int8"
