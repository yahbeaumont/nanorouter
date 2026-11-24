from nanorouter.schema import (
    NO_TOOL,
    RouteDecision,
    parse_decision,
    parse_json_loose,
    validate_against_spec,
)


def test_parse_plain():
    d, err = parse_decision(
        '{"tool": "get_weather", "args": {"city": "Recife", "date": "2026-12-01"}, "confidence": 0.8}'
    )
    assert err is None and d.tool == "get_weather"


def test_parse_fenced():
    d, _ = parse_decision('```json\n{"tool": "no_tool", "args": {}, "confidence": 0.5}\n```')
    assert d.tool == NO_TOOL


def test_parse_with_prose():
    d, _ = parse_decision(
        'Claro! {"tool": "booking_status", "args": {"booking_id": "BK1000"}, "confidence": 0.9} Espero ter ajudado.'
    )
    assert d.tool == "booking_status"


def test_parse_no_json():
    assert parse_json_loose("não sei") is None
    d, err = parse_decision("não sei")
    assert d is None and err == "no_json_object"


def test_extra_field_forbidden():
    d, err = parse_decision('{"tool": "no_tool", "args": {}, "confidence": 0.5, "x": 1}')
    assert d is None and err.startswith("schema:")


def test_confidence_bounds():
    assert parse_decision('{"tool": "no_tool", "args": {}, "confidence": 1.4}')[0] is None


def test_validate_ok(spec):
    d = RouteDecision(
        tool="get_weather", args={"city": "Recife", "date": "2026-12-01"}, confidence=0.9
    )
    assert validate_against_spec(d, spec) == []


def test_validate_missing_required(spec):
    d = RouteDecision(tool="get_weather", args={"city": "Recife"}, confidence=0.9)
    assert "missing_required:date" in validate_against_spec(d, spec)


def test_validate_unknown_slot(spec):
    d = RouteDecision(tool="booking_status", args={"booking_id": "BK1", "foo": 1}, confidence=0.9)
    assert "unknown_slot:foo" in validate_against_spec(d, spec)


def test_validate_bad_enum(spec):
    d = RouteDecision(tool="search_poi", args={"city": "Recife", "category": "zoo"}, confidence=0.9)
    assert "bad_enum:category" in validate_against_spec(d, spec)


def test_no_tool_with_args(spec):
    d = RouteDecision(tool=NO_TOOL, args={"a": 1}, confidence=0.5)
    assert "no_tool_with_args" in validate_against_spec(d, spec)
