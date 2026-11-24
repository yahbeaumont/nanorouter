---
base_model: $base_model
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - routing
  - json-schema
  - function-calling
  - tool-use
language:
  - pt
  - en
  - es
license: mit
---

# nanorouter-lora

LoRA adapter for **$base_model**, fine-tuned to emit routing JSON that validates against a strict
tool schema. Trained on a synthetic dataset with hard negatives. Part of the
[nanorouter](https://github.com/yahbeaumont/nanorouter) project, whose documentation is in
Brazilian Portuguese.

Published on $date. Hub repo: `$repo`.

## What it does

Given an utterance and a tool catalog in the prompt, the model returns one of three outcomes: a tool
call with typed arguments, a refusal to call anything (`no_tool`), or an abstention when the request
is ambiguous. The schema is enforced after generation, so an invalid answer is a measured failure
rather than something that reaches the caller.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("$base_model")
model = PeftModel.from_pretrained(base, "$repo")
tok = AutoTokenizer.from_pretrained("$repo")
```

Or through the project CLI, which brings the prompt template and the spec validation:

```bash
nanorouter route "cancela a reserva BK1042" --adapter $repo
```

## Results

Same suite as the repository, greedy decoding, fixed seed:

$table

The full method and the exact reproduction commands live in `docs/RESULTS.md` of the repository.
If the table above says `sem run`, no evaluation was executed when this card was built: the numbers
are produced by `make eval`, not copied from anywhere.

## Limitations

Synthetic dataset, so it inherits template bias. Eight tools per spec; a catalog of hundreds will not
fit the prompt. `confidence` is not a calibrated probability. One tool per call: utterances with two
legitimate intents return a single tool. Languages: pt-BR, en, es. Use constrained decoding in
production instead of trusting the model to close the JSON by itself.

## Training

LoRA r=16, alpha=32 on all linear projections, completion-only loss, 3 epochs, lr 2e-4, cosine,
bf16. Hyperparameters in `configs/lora_sft.yaml`. The checkpoint records a `spec_fingerprint.json`
with the hash of the tool spec, the seed and the record count, so an adapter can be traced back to
the contract it was trained against.

## License

MIT. The base model is Apache-2.0.
