"""Publica o adapter LoRA no Hugging Face Hub com a model card preenchida.

Requer `pip install huggingface_hub` e `hf auth login` (ou `HF_TOKEN`).

    python scripts/publish_adapter.py --repo <usuario>/nanorouter-lora
    python scripts/publish_adapter.py --repo <usuario>/nanorouter-lora --dry-run

O `--dry-run` monta o staging em `outputs/_publish` sem enviar nada: serve para
inspecionar a card antes de publicar.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from nanorouter.hub import BASE_MODEL, DEFAULT_CARD, build_card, stage

DEFAULT_ADAPTER = Path("checkpoints/lora/final")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="<usuario>/<nome> no Hub")
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--eval-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--staging", type=Path, default=Path("outputs/_publish"))
    parser.add_argument("--private", action="store_true", help="cria o repo privado no Hub")
    parser.add_argument("--dry-run", action="store_true", help="monta o staging sem enviar")
    args = parser.parse_args()

    card = build_card(args.card, args.eval_dir, args.repo, args.base_model)
    staging = stage(
        args.adapter_dir,
        args.staging,
        card,
        {"base_model": args.base_model, "repo": args.repo},
    )
    print(f"staging pronto: {staging}")

    if args.dry_run:
        print("dry-run: nada foi enviado")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(staging),
        repo_id=args.repo,
        repo_type="model",
        commit_message="publish LoRA adapter",
    )
    shutil.rmtree(staging)
    print(f"publicado: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
