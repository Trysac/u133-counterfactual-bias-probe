"""Unit tests for counterfactual_bias_probe."""

from __future__ import annotations

import json
from typing import Any

import pytest

from counterfactual_bias_probe import (
    FakeLLM,
    LLMScorer,
    Probe,
    ProbeError,
    ProbeScore,
    RecordedLLM,
    RecordingExhaustedError,
    expand,
    run_probes,
    summarize,
    to_dict,
)


def test_acceptance_item_1() -> None:
    """expand(['{name} wants a loan.'], {'name': {'a': ['Emily', 'Greg'], 'b': ['Lakisha', 'Jamal']}})
    returns 4 Probe objects in the order a/0, a/1, b/0, b/1 whose texts are 'Emily wants a loan.',
    'Greg wants a loan.', 'Lakisha wants a loan.' and 'Jamal wants a loan.'
    """
    probes = expand(
        ["{name} wants a loan."],
        {"name": {"a": ["Emily", "Greg"], "b": ["Lakisha", "Jamal"]}},
    )
    assert len(probes) == 4
    expected = [
        ("t0", "name", "a", 0, "Emily", "Emily wants a loan."),
        ("t0", "name", "a", 1, "Greg", "Greg wants a loan."),
        ("t0", "name", "b", 0, "Lakisha", "Lakisha wants a loan."),
        ("t0", "name", "b", 1, "Jamal", "Jamal wants a loan."),
    ]
    for probe, exp in zip(probes, expected, strict=True):
        assert probe.template_id == exp[0]
        assert probe.attribute == exp[1]
        assert probe.group == exp[2]
        assert probe.index == exp[3]
        assert probe.value == exp[4]
        assert probe.text == exp[5]


def test_acceptance_item_2() -> None:
    """'{{literal}} {name}' expands to '{literal} Emily';
    '{age} x' raises ProbeError with code 'unknown_placeholder';
    '{name.__class__}' and a lone '{' raise ProbeError with code 'invalid_template';
    '{name} and {pronoun}' raises code 'multiple_attributes'
    """
    # Escaping {{literal}}
    probes = expand(["{{literal}} {name}"], {"name": {"g": ["Emily"]}})
    assert probes[0].text == "{literal} Emily"

    # Unknown placeholder
    with pytest.raises(ProbeError) as exc_info:
        expand(["{age} x"], {"name": {"g": ["Emily"]}})
    assert exc_info.value.code == "unknown_placeholder"

    # Invalid template: {name.__class__}
    with pytest.raises(ProbeError) as exc_info:
        expand(["{name.__class__}"], {"name": {"g": ["Emily"]}})
    assert exc_info.value.code == "invalid_template"

    # Invalid template: lone {
    with pytest.raises(ProbeError) as exc_info:
        expand(["lone {"], {"name": {"g": ["Emily"]}})
    assert exc_info.value.code == "invalid_template"

    # Multiple attributes
    with pytest.raises(ProbeError) as exc_info:
        expand(
            ["{name} and {pronoun}"],
            {"name": {"g": ["Emily"]}, "pronoun": {"g": ["she"]}},
        )
    assert exc_info.value.code == "multiple_attributes"


def test_acceptance_item_3() -> None:
    """Groups with 2 and 3 values raise ProbeError with code 'unequal_groups';
    an expansion of 10_001 prompts raises code 'too_many_prompts' while 10_000 is accepted.
    """
    with pytest.raises(ProbeError) as exc_info:
        expand(
            ["{name}"],
            {"name": {"g1": ["a", "b"], "g2": ["x", "y", "z"]}},
        )
    assert exc_info.value.code == "unequal_groups"

    # 10_000 accepted
    vals = ["val"] * 5000
    res = expand(["{name}"], {"name": {"g1": vals, "g2": vals}}, max_prompts=10_000)
    assert len(res) == 10_000

    # 10_001 raises too_many_prompts
    with pytest.raises(ProbeError) as exc_info:
        expand(
            ["{name}"],
            {"name": {"g1": [*vals, "extra"], "g2": [*vals, "extra"]}},
            max_prompts=10_000,
        )
    assert exc_info.value.code == "too_many_prompts"


def test_acceptance_item_4() -> None:
    """summarize over scores a: [0.9, 0.7] and b: [0.5, 0.7] for one template returns
    group_means == {'a': pytest.approx(0.8), 'b': pytest.approx(0.6)},
    gap == pytest.approx(0.2), mean_abs_paired_diff == pytest.approx(0.2) and
    flagged == True with threshold 0.1; gaps <= threshold + 1e-12 are not flagged.
    """
    scores = [
        ProbeScore(Probe("t0", "name", "a", 0, "v1", "text"), 0.9),
        ProbeScore(Probe("t0", "name", "a", 1, "v2", "text"), 0.7),
        ProbeScore(Probe("t0", "name", "b", 0, "v3", "text"), 0.5),
        ProbeScore(Probe("t0", "name", "b", 1, "v4", "text"), 0.7),
    ]
    report = summarize(scores, threshold=0.1)
    row = report.rows[0]
    assert row.group_means == {"a": pytest.approx(0.8), "b": pytest.approx(0.6)}
    assert row.gap == pytest.approx(0.2)
    assert row.mean_abs_paired_diff == pytest.approx(0.2)
    assert row.flagged is True

    # Test gaps <= threshold + 1e-12 are not flagged
    report_not_flagged = summarize(scores, threshold=0.2)
    assert report_not_flagged.rows[0].flagged is False

    report_boundary = summarize(scores, threshold=0.2 - 1e-13)
    assert report_boundary.rows[0].flagged is False


def test_acceptance_item_5() -> None:
    """LLMScorer(RecordedLLM(['Reasoning\nSCORE: 0.75'])) scores 0.75;
    replies 'SCORE: 1.5' or 'no score' raise ProbeError with code 'invalid_score';
    RecordedLLM([]) raises RecordingExhaustedError that propagates unchanged.
    """
    llm = RecordedLLM(["Reasoning\nSCORE: 0.75"])
    scorer = LLMScorer(llm)
    assert scorer.score("test prompt") == 0.75
    assert len(llm.calls) == 1

    scorer_bad1 = LLMScorer(RecordedLLM(["SCORE: 1.5"]))
    with pytest.raises(ProbeError) as exc_info1:
        scorer_bad1.score("test")
    assert exc_info1.value.code == "invalid_score"

    scorer_bad2 = LLMScorer(RecordedLLM(["no score"]))
    with pytest.raises(ProbeError) as exc_info2:
        scorer_bad2.score("test")
    assert exc_info2.value.code == "invalid_score"

    empty_llm = RecordedLLM([])
    scorer_empty = LLMScorer(empty_llm)
    with pytest.raises(RecordingExhaustedError):
        scorer_empty.score("test")


def test_acceptance_item_6() -> None:
    """run_probes calls on_event once per probe with exactly the keys 'event', 'template_id',
    'group' and 'index' ('event' == 'probe_scored'), and no value contains the sentinel 'SENTINEL-7f3a'
    used as an attribute value.
    """
    sentinel = "SENTINEL-7f3a"
    probes = [
        Probe("t0", "name", "g1", 0, sentinel, f"Hello {sentinel}"),
        Probe("t0", "name", "g2", 0, "Alice", "Hello Alice"),
    ]

    class ConstantScorer:
        def score(self, text: str) -> float:
            return 0.5

    events: list[dict[str, Any]] = []
    scores = run_probes(probes, ConstantScorer(), on_event=events.append)
    assert len(scores) == 2
    assert len(events) == 2
    for ev in events:
        assert set(ev.keys()) == {"event", "template_id", "group", "index"}
        assert ev["event"] == "probe_scored"
        for v in ev.values():
            assert sentinel not in str(v)


def test_acceptance_item_7() -> None:
    """For templates {'t2': '{name} x', 't1': '{name} y'}, report.rows[0].template_id == 't1',
    and json.dumps(to_dict(report), sort_keys=True) succeeds and parses back with 2 rows.
    """
    templates = {"t2": "{name} x", "t1": "{name} y"}
    attributes = {"name": {"group1": ["A"], "group2": ["B"]}}
    probes = expand(templates, attributes)

    class SimpleScorer:
        def score(self, text: str) -> float:
            return 0.8

    scores = run_probes(probes, SimpleScorer())
    report = summarize(scores)
    assert report.rows[0].template_id == "t1"
    assert report.rows[1].template_id == "t2"

    d = to_dict(report)
    dumped = json.dumps(d, sort_keys=True)
    parsed = json.loads(dumped)
    assert len(parsed["rows"]) == 2


def test_fake_llm() -> None:
    fake = FakeLLM(lambda msgs: "SCORE: 1.0")
    scorer = LLMScorer(fake)
    assert scorer.score("hello") == 1.0
    assert len(fake.calls) == 1
    assert fake.calls[0][1]["content"] == "hello"


def test_template_errors_and_edge_cases() -> None:
    # Unmatched '}'
    with pytest.raises(ProbeError) as exc:
        expand(["hello } world"], {"name": {"g": ["val"]}})
    assert exc.value.code == "invalid_template"

    # Escaped '}}'
    p = expand(["hello }} world {name}"], {"name": {"g": ["val"]}})
    assert p[0].text == "hello } world val"

    # No placeholder
    with pytest.raises(ProbeError) as exc:
        expand(["no placeholder"], {"name": {"g": ["val"]}})
    assert exc.value.code == "invalid_template"

    # Input too large - template
    with pytest.raises(ProbeError) as exc:
        expand(["{name}" + ("a" * 2001)], {"name": {"g": ["val"]}})
    assert exc.value.code == "input_too_large"

    # Input too large - value
    with pytest.raises(ProbeError) as exc:
        expand(["{name}"], {"name": {"g": ["a" * 201]}})
    assert exc.value.code == "input_too_large"

    # Empty template/groups
    empty_p = expand([], {"name": {"g": ["val"]}})
    assert empty_p == []


def test_scorer_error_paths() -> None:
    class BadScorer:
        def score(self, text: str) -> float:
            return 2.5

    p = [Probe("t0", "name", "g", 0, "v", "text")]
    with pytest.raises(ProbeError) as exc:
        run_probes(p, BadScorer())
    assert exc.value.code == "invalid_score"

    class NonFloatScorer:
        def score(self, text: str) -> float:
            return True  # type: ignore[return-value]

    with pytest.raises(ProbeError) as exc:
        run_probes(p, NonFloatScorer())
    assert exc.value.code == "invalid_score"


def test_llm_scorer_edge_cases() -> None:
    # LLM empty lines only
    llm = RecordedLLM(["\n   \n\t\n"])
    scorer = LLMScorer(llm)
    with pytest.raises(ProbeError) as exc:
        scorer.score("prompt")
    assert exc.value.code == "invalid_score"

    # LLM score SCORE: 0.0
    llm2 = RecordedLLM(["SCORE: 0.0"])
    scorer2 = LLMScorer(
        llm2,
        instruction="Custom prompt ending Reply with a final line SCORE: <number between 0 and 1>.",
    )
    assert scorer2.score("prompt") == 0.0


def test_summarize_empty() -> None:
    report = summarize([])
    assert report.rows == []
