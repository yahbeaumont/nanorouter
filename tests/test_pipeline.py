"""Testes do harness sem GPU e sem torch: backends falsos, mesmo contrato."""

from __future__ import annotations

import json

import pytest

from nanorouter import prompts
from nanorouter.backends import Backend
from nanorouter.baselines import FewShotBackend
from nanorouter.evaluate import compare, run_suite
from nanorouter.infer import batch, route
from nanorouter.metrics import confusion
from nanorouter.plotting import plot
from nanorouter.schema import RouteDecision
from nanorouter.toolspec import compact_tool_list, load_spec, spec_fingerprint

GOOD = (
    '{"tool": "get_weather", "args": {"city": "Recife", "date": "2026-12-01"}, "confidence": 0.9}'
)
BAD = "não sei responder isso"


class FakeBackend(Backend):
    """Devolve um payload fixo. Substitui HF/GGUF/API nos testes."""

    def __init__(self, payload: str, name: str = "fake"):
        self.payload = payload
        self.name = name
        self.calls: list[str] = []

    def complete(self, spec, utterance):  # noqa: ARG002
        self.calls.append(utterance)
        return self.payload


@pytest.fixture()
def suite():
    return [
        {
            "id": "t1",
            "utterance": "como está o tempo em Recife?",
            "tool": "get_weather",
            "args": {"city": "Recife", "date": "2026-12-01"},
            "confidence": 0.9,
        },
        {
            "id": "t2",
            "utterance": "oi, tudo bem?",
            "tool": "no_tool",
            "args": {},
            "confidence": 0.7,
        },
    ]


def test_prompts_contain_catalogue(spec):
    system = prompts.build_system_prompt(spec)
    assert "get_weather" in system
    assert "no_tool" in system
    msgs = prompts.build_messages(spec, "teste")
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_gold_completion_is_canonical():
    out = prompts.gold_completion({"tool": "no_tool", "args": {}})
    assert out == '{"args": {}, "confidence": 0.9, "tool": "no_tool"}'


def test_toolspec_helpers(spec):
    assert spec_fingerprint(spec) == spec_fingerprint(spec)
    assert len(spec_fingerprint(spec)) == 12
    listing = compact_tool_list(spec)
    assert listing.count("\n") == len(spec["tools"]) - 1


def test_toolspec_rejects_duplicates(tmp_path):
    bad = tmp_path / "dup.yaml"
    bad.write_text("tools:\n  - {name: a, slots: {}}\n  - {name: a, slots: {}}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_spec(bad)


def test_run_suite_and_compare(tmp_path, spec, suite):
    good = run_suite(FakeBackend(GOOD), spec, suite)
    assert good["metrics"]["n"] == 2
    assert good["metrics"]["tool_accuracy"] == 0.5
    assert good["metrics"]["spec_validity"] == 1.0

    bad = run_suite(FakeBackend(BAD, name="bad"), spec, suite)
    assert bad["metrics"]["parse_error_rate"] == 1.0
    assert bad["metrics"]["exact_match"] == 0.0

    out = tmp_path / "eval"
    md = compare([good, bad], out)
    text = md.read_text(encoding="utf-8")
    assert "| backend |" in text
    assert (out / "fake.jsonl").exists()
    rows = [json.loads(line) for line in (out / "fake.jsonl").read_text().splitlines()]
    assert rows[0]["gold"] == "get_weather"


def test_plot_writes_confusion(tmp_path, spec, suite):
    out = tmp_path / "eval"
    compare([run_suite(FakeBackend(GOOD), spec, suite)], out)
    written = plot(out)
    assert written and written[0].exists() and written[0].stat().st_size > 0


def test_infer_ok_and_error(spec):
    ok = route(FakeBackend(GOOD), spec, "tempo em Recife?")
    assert ok["ok"] is True and ok["violations"] == []
    err = route(FakeBackend(BAD), spec, "tempo em Recife?")
    assert err["ok"] is False and err["error"] == "no_json_object"

    invalid = route(
        FakeBackend('{"tool": "get_weather", "args": {"city": "Recife"}, "confidence": 0.9}'),
        spec,
        "tempo?",
    )
    assert invalid["ok"] is False
    assert "missing_required:date" in invalid["violations"]


def test_batch(spec):
    out = batch(FakeBackend(GOOD), spec, ["a", "b"])
    assert len(out) == 2


def test_fewshot_injects_examples(spec, suite):
    inner = FakeBackend(GOOD)
    shot = FewShotBackend(inner, spec, suite)
    assert shot.name.startswith("fewshot-2:")
    shot.route(spec, "tempo em Recife?")
    assert "EXEMPLO 1" in shot.prefix
    assert inner.calls  # o backend interno foi chamado
    assert inner.calls[0].startswith("Exemplos de referência:")


def test_confusion_matrix(spec):
    labels, matrix = confusion([("get_weather", "get_weather"), ("no_tool", "get_weather")])
    assert labels == ["get_weather", "no_tool"]
    assert sum(map(sum, matrix)) == 2


def test_decision_roundtrip():
    d = RouteDecision(tool="no_tool", args={}, confidence=0.5)
    assert json.loads(d.to_json())["tool"] == "no_tool"
