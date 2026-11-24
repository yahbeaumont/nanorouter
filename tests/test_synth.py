from nanorouter import synth
from nanorouter.schema import RouteDecision, validate_against_spec


def test_generation_is_deterministic(spec):
    a = list(synth.generate(spec, 200, 42))
    b = list(synth.generate(spec, 200, 42))
    assert a == b


def test_all_records_valid(spec):
    for rec in synth.generate(spec, 500, 7):
        d = RouteDecision(tool=rec["tool"], args=rec["args"], confidence=rec["confidence"])
        assert validate_against_spec(d, spec) == []


def test_split_disjoint(spec):
    buckets = synth.split(list(synth.generate(spec, 900, 3)), 3)
    ids = [r["id"] for r in buckets["train"] + buckets["val"] + buckets["test"]]
    assert len(ids) == len(set(ids))
    assert all(buckets.values())


def test_has_all_kinds(spec):
    kinds = {r["kind"] for r in synth.generate(spec, 400, 11)}
    assert kinds == {"tool", "no_tool", "hard_negative"}


def test_every_template_renders(spec):
    """Regressão: templates que pedem {date} em tools com checkin/checkout."""
    import random

    for tool in spec["tools"]:
        slots = synth._slot_values(random.Random(0), tool)
        for template in synth.TEMPLATES[tool["name"]]:
            out = synth._render(random.Random(0), template, slots)
            assert "{" not in out, (tool["name"], template, out)


def test_hard_negatives_are_templated(spec):
    """Hard negative não pode ser frase fixa: 10 frases viram 10 exemplos repetidos."""
    for template, tool_name in synth.HARD_NEGATIVES:
        assert "{" in template
        assert tool_name in {t["name"] for t in spec["tools"]}


def test_hard_negative_utterances_vary(spec):
    texts = {r["utterance"] for r in synth.generate(spec, 600, 5) if r["kind"] == "hard_negative"}
    assert len(texts) > 50
