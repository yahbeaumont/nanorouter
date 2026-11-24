"""CLI."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print_json

from . import synth
from .toolspec import load_spec

app = typer.Typer(add_completion=False, help="nanorouter — roteamento com JSON estrito.")


@app.command()
def data(
    out: Path = typer.Option("data", help="diretório de saída"),
    n: int = typer.Option(6000, help="registros a gerar"),
    seed: int = typer.Option(1337),
    spec_path: str = typer.Option("configs/tools.yaml"),
) -> None:
    """Gera dataset sintético e split estratificado."""
    spec = load_spec(spec_path)
    records = list(synth.generate(spec, n, seed))
    buckets = synth.split(records, seed)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in buckets.items():
        path = out / f"{name}.jsonl"
        with path.open("w") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        typer.echo(f"{name}: {len(items)} -> {path}")


@app.command()
def train(
    config: str = typer.Option("configs/lora_sft.yaml"),
    data_path: str = typer.Option("data/train.jsonl"),
    spec_path: str = typer.Option("configs/tools.yaml"),
    out: str = typer.Option("checkpoints/lora"),
) -> None:
    """Fine-tuna com LoRA."""
    from .train import run

    typer.echo(run(config, data_path, spec_path, out))


@app.command("eval")
def eval_cmd(
    suite: str = typer.Option("data/test.jsonl"),
    out: Path = typer.Option("outputs/eval"),
    spec_path: str = typer.Option("configs/tools.yaml"),
    model: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct"),
    adapter: str | None = typer.Option(None),
    gguf: str | None = typer.Option(None),
    skip_baselines: bool = typer.Option(False),
    threads: int = typer.Option(4),
    limit: int | None = typer.Option(None),
    plot: bool = typer.Option(False),
) -> None:
    """Avalia base, few-shot, GGUF e (se houver) fine-tunado na mesma suite."""
    from .backends import HFBackend, LlamaCppBackend
    from .baselines import FewShotBackend
    from .evaluate import compare, run_suite

    spec = load_spec(spec_path)
    items = [json.loads(line) for line in Path(suite).read_text().splitlines() if line]
    val = [json.loads(line) for line in Path("data/val.jsonl").read_text().splitlines()[:4]]

    backends = []
    if not skip_baselines:
        base = HFBackend(model)
        backends += [base, FewShotBackend(base, spec, val)]
    if gguf:
        backends.append(LlamaCppBackend(gguf, threads=threads))
    if adapter:
        backends.append(HFBackend(model, adapter=adapter))
    if not backends:
        raise typer.BadParameter(
            "nada para avaliar: use --adapter, --gguf ou remova --skip-baselines"
        )

    results = [run_suite(b, spec, items, limit) for b in backends]
    md = compare(results, out)
    typer.echo(f"resultados: {md}")
    if plot:
        from .plotting import plot as plot_confusion

        plot_confusion(out)


@app.command()
def route(
    utterance: str,
    adapter: str | None = typer.Option(None),
    gguf: str | None = typer.Option(None),
) -> None:
    """Roteia um enunciado único."""
    from .backends import HFBackend, LlamaCppBackend
    from .infer import route as do_route

    spec = load_spec("configs/tools.yaml")
    backend = (
        LlamaCppBackend(gguf) if gguf else HFBackend("Qwen/Qwen2.5-0.5B-Instruct", adapter=adapter)
    )
    print_json(json.dumps(do_route(backend, spec, utterance), ensure_ascii=False))


@app.command()
def export(out: Path = typer.Option("outputs/export")) -> None:
    """Merge + GGUF + ONNX."""
    from .export import merge_adapter, to_gguf, to_onnx

    out.mkdir(parents=True, exist_ok=True)
    merged = merge_adapter("Qwen/Qwen2.5-0.5B-Instruct", "checkpoints/lora/final", out)
    typer.echo(f"gguf: {to_gguf(merged, out)}")
    typer.echo(f"onnx: {to_onnx(merged, out)}")


@app.command()
def serve(port: int = typer.Option(8000)) -> None:
    """Sobe a API."""
    import uvicorn

    uvicorn.run("nanorouter.server:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    app()
