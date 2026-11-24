from nanorouter.metrics import arg_counts, bootstrap_ci, compute, exact_match
from nanorouter.schema import RouteDecision


def _d(tool="get_weather", args=None, c=0.9):
    return RouteDecision(
        tool=tool, args=args or {"city": "Recife", "date": "2026-12-01"}, confidence=c
    )


def test_exact_match_true():
    assert exact_match(
        _d(),
        {
            "tool": "get_weather",
            "args": {"city": "Recife", "date": "2026-12-01"},
            "confidence": 0.9,
        },
    )


def test_exact_match_confidence_tolerance():
    gold = {
        "tool": "get_weather",
        "args": {"city": "Recife", "date": "2026-12-01"},
        "confidence": 0.9,
    }
    assert not exact_match(_d(c=0.3), gold)


def test_arg_counts_wrong_tool_zeroes():
    tp, p, g = arg_counts(_d(tool="no_tool"), {"tool": "get_weather", "args": {"city": "X"}})
    assert (tp, p, g) == (0, 0, 1)


def test_bootstrap_bounds():
    lo, hi = bootstrap_ci([1.0, 1.0, 1.0])
    assert lo <= hi


def test_compute_parse_errors_counted(spec):
    rows = [{"gold": {"tool": "no_tool", "args": {}}, "pred": None, "latency_ms": 10.0}]
    m = compute(rows, spec)
    assert m["parse_error_rate"] == 1.0 and m["exact_match"] == 0.0
