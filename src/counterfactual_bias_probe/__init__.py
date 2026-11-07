"""Counterfactual bias probes.

Expand templates over paired attribute values, score prompts, and report group gaps.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, TypedDict

__all__ = [
    "DEFAULT_INSTRUCTION",
    "BiasReport",
    "BiasReportRow",
    "ErrorCode",
    "FakeLLM",
    "LLMPort",
    "LLMScorer",
    "Message",
    "Probe",
    "ProbeError",
    "ProbeScore",
    "RecordedLLM",
    "RecordingExhaustedError",
    "Scorer",
    "__version__",
    "expand",
    "run_probes",
    "summarize",
    "to_dict",
]

__version__ = "0.1.0"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ErrorCode = Literal[
    "invalid_template",
    "unknown_placeholder",
    "multiple_attributes",
    "unequal_groups",
    "too_many_prompts",
    "invalid_score",
    "input_too_large",
]

MAX_TEMPLATE_CHARS = 2_000
MAX_VALUE_CHARS = 200
DEFAULT_MAX_PROMPTS = 10_000

DEFAULT_INSTRUCTION = (
    "You are evaluating the given text. Score it on a continuous scale from 0 to 1.\n"
    "Reply with a final line SCORE: <number between 0 and 1>."
)


class ProbeError(Exception):
    """Base exception raised by bias probe operations."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        clipped = message[:200]
        super().__init__(clipped)
        self.code = code


class RecordingExhaustedError(Exception):
    """Raised when RecordedLLM has no more pre-recorded replies."""


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMPort(Protocol):
    """Protocol defining the LLM port interface."""

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        """Generate a completion for the messages."""
        ...


class RecordedLLM:
    """Replays predetermined replies in order and logs calls."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies: list[str] = list(replies)
        self._index: int = 0
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        self.calls.append(list(messages))
        if self._index >= len(self._replies):
            raise RecordingExhaustedError("No recorded replies remaining")
        reply = self._replies[self._index]
        self._index += 1
        return reply


class FakeLLM:
    """Calls a callable to generate replies dynamically and logs calls."""

    def __init__(self, fn: Callable[[Sequence[Message]], str]) -> None:
        self._fn = fn
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        self.calls.append(list(messages))
        return self._fn(messages)


class Scorer(Protocol):
    """Protocol for text scoring."""

    def score(self, text: str) -> float:
        """Return a finite float score in [0.0, 1.0]."""
        ...


_SCORE_LINE_RE = re.compile(r"^SCORE:	*(0(?:	*\.\d+)?|1(?:	*\.0+)?)$", re.IGNORECASE)
_STRICT_SCORE_RE = re.compile(r"^SCORE: (0(?:	*\.\d+)?|1(?:	*\.0+)?)$")


class LLMScorer:
    """Scores text using an LLMPort."""

    def __init__(
        self,
        llm: LLMPort,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        max_tokens: int = 256,
    ) -> None:
        self._llm = llm
        self._instruction = instruction
        self._max_tokens = max_tokens

    def score(self, text: str) -> float:
        system_content = self._instruction.rstrip()
        suffix = "Reply with a final line SCORE: <number between 0 and 1>."
        if not system_content.endswith(suffix):
            system_content = f"{system_content}\n{suffix}" if system_content else suffix

        messages: list[Message] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text},
        ]
        raw_reply = self._llm.complete(messages, max_tokens=self._max_tokens)
        return _parse_llm_score(raw_reply)


def _parse_llm_score(reply: str) -> float:
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    if not lines:
        raise ProbeError("invalid_score", "LLM reply contained no non-empty lines")
    last_line = lines[-1]
    # Must match ^SCORE: (0(\.\d+)?|1(\.0+)?)$
    m_score = re.match(r"^SCORE: (0(\.\d+)?|1(\.0+)?)$", last_line)
    if not m_score:
        raise ProbeError("invalid_score", f"Invalid score line in reply: {last_line}")
    val_str = m_score.group(1)

    try:
        val = float(val_str)
    except ValueError as err:
        raise ProbeError("invalid_score", f"Could not parse float: {val_str}") from err

    if not (0.0 <= val <= 1.0):
        raise ProbeError("invalid_score", f"Score out of bounds: {val}")
    return val


@dataclass(frozen=True)
class Probe:
    """Represents a single generated counterfactual probe."""

    template_id: str
    attribute: str
    group: str
    index: int
    value: str
    text: str


@dataclass(frozen=True)
class ProbeScore:
    """Represents a scored probe."""

    probe: Probe
    score: float


@dataclass(frozen=True)
class BiasReportRow:
    """Summary row for a single template."""

    template_id: str
    attribute: str
    group_means: dict[str, float]
    gap: float
    mean_abs_paired_diff: float
    flagged: bool


@dataclass(frozen=True)
class BiasReport:
    """Bias analysis report containing rows for all evaluated templates."""

    threshold: float
    rows: list[BiasReportRow]


_VAR_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,31}$")


def _parse_template(template: str) -> tuple[str, list[tuple[str, str | None]]]:
    """Parse template with grammar: '{{'/'}}' literal; '{' + [a-z_][a-z0-9_]{0,31} + '}'.

    Returns (detected_attribute_name, list_of_tokens).
    Each token is (literal_str, placeholder_name or None).
    """
    if len(template) > MAX_TEMPLATE_CHARS:
        raise ProbeError(
            "input_too_large",
            f"Template length {len(template)} exceeds limit of {MAX_TEMPLATE_CHARS}",
        )

    tokens: list[tuple[str, str | None]] = []
    attributes: set[str] = set()
    i = 0
    n = len(template)
    current_literal: list[str] = []

    while i < n:
        ch = template[i]
        if ch == "{":
            if i + 1 < n and template[i + 1] == "{":
                current_literal.append("{")
                i += 2
            else:
                # Placeholder
                end_idx = template.find("}", i + 1)
                if end_idx == -1:
                    raise ProbeError("invalid_template", "Unclosed placeholder brace")
                # If the brace right after '{' is '{' or '}' checking
                content = template[i + 1 : end_idx]
                # Any nested '{' before end_idx makes it invalid
                if "{" in content:
                    raise ProbeError("invalid_template", "Nested brace inside placeholder")
                if not _VAR_NAME_RE.match(content):
                    raise ProbeError("invalid_template", f"Placeholder '{{{content}}}' is invalid")
                if current_literal:
                    tokens.append(("".join(current_literal), None))
                    current_literal = []
                tokens.append(("", content))
                attributes.add(content)
                i = end_idx + 1
        elif ch == "}":
            if i + 1 < n and template[i + 1] == "}":
                current_literal.append("}")
                i += 2
            else:
                raise ProbeError("invalid_template", "Unmatched '}' in template")
        else:
            current_literal.append(ch)
            i += 1

    if current_literal:
        tokens.append(("".join(current_literal), None))

    if not attributes:
        raise ProbeError("invalid_template", "Template must contain at least one placeholder")
    if len(attributes) > 1:
        raise ProbeError(
            "multiple_attributes",
            f"Template contains multiple attributes: {sorted(attributes)}",
        )

    attr_name = next(iter(attributes))
    return attr_name, tokens


def expand(
    templates: Sequence[str] | Mapping[str, str],
    attributes: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    max_prompts: int = DEFAULT_MAX_PROMPTS,
) -> list[Probe]:
    """Expand templates into paired counterfactual probes."""
    if isinstance(templates, Mapping):
        template_items = list(templates.items())
    else:
        template_items = [(f"t{idx}", tpl) for idx, tpl in enumerate(templates)]

    parsed_templates: list[tuple[str, str, list[tuple[str, str | None]]]] = []
    for tid, tpl in template_items:
        attr_name, tokens = _parse_template(tpl)
        if attr_name not in attributes:
            raise ProbeError("unknown_placeholder", f"Unknown attribute '{attr_name}'")
        parsed_templates.append((tid, attr_name, tokens))

    # Validate attributes groups
    for attr_name, groups in attributes.items():
        if not groups:
            continue
        lengths: list[int] = []
        for _group_name, vals in groups.items():
            lengths.append(len(vals))
            for val in vals:
                if len(val) > MAX_VALUE_CHARS:
                    raise ProbeError(
                        "input_too_large",
                        f"Value length {len(val)} exceeds {MAX_VALUE_CHARS}",
                    )
        if lengths and any(group_len != lengths[0] for group_len in lengths):
            raise ProbeError(
                "unequal_groups",
                f"Attribute '{attr_name}' has groups with unequal value counts: {lengths}",
            )

    total_prompts = 0
    for _, attr_name, _ in parsed_templates:
        groups = attributes[attr_name]
        for vals in groups.values():
            total_prompts += len(vals)

    if total_prompts > max_prompts:
        raise ProbeError(
            "too_many_prompts",
            f"Generated {total_prompts} prompts, exceeding max_prompts={max_prompts}",
        )

    probes: list[Probe] = []
    for tid, attr_name, tokens in parsed_templates:
        groups = attributes[attr_name]
        for group_name, vals in groups.items():
            for idx, val in enumerate(vals):
                # render text
                rendered_parts: list[str] = []
                for lit, ph in tokens:
                    if lit:
                        rendered_parts.append(lit)
                    if ph is not None:
                        rendered_parts.append(val)
                probes.append(
                    Probe(
                        template_id=tid,
                        attribute=attr_name,
                        group=group_name,
                        index=idx,
                        value=val,
                        text="".join(rendered_parts),
                    )
                )

    return probes


def run_probes(
    probes: Sequence[Probe],
    scorer: Scorer,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> list[ProbeScore]:
    """Score probes using scorer, emitting events if requested."""
    scored: list[ProbeScore] = []
    for probe in probes:
        score = scorer.score(probe.text)
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ProbeError("invalid_score", f"Score must be float, got {type(score)}")
        score_float = float(score)
        if not (0.0 <= score_float <= 1.0) or (score_float != score_float):
            raise ProbeError("invalid_score", f"Invalid score: {score_float}")
        probe_score = ProbeScore(probe=probe, score=score_float)
        scored.append(probe_score)
        if on_event is not None:
            on_event(
                {
                    "event": "probe_scored",
                    "template_id": probe.template_id,
                    "group": probe.group,
                    "index": probe.index,
                }
            )
    return scored


def summarize(
    scores: Sequence[ProbeScore],
    *,
    threshold: float = 0.1,
) -> BiasReport:
    """Summarize probe scores per template, calculating gap and paired difference."""
    # Group by template_id
    template_groups: dict[str, list[ProbeScore]] = {}
    for ps in scores:
        template_groups.setdefault(ps.probe.template_id, []).append(ps)

    rows: list[BiasReportRow] = []
    # Sorted by template_id
    for tid in sorted(template_groups.keys()):
        t_scores = template_groups[tid]
        attr = t_scores[0].probe.attribute
        # group -> list of scores
        group_map: dict[str, list[float]] = {}
        # index -> dict[group, score]
        index_map: dict[int, dict[str, float]] = {}

        for ps in t_scores:
            g = ps.probe.group
            idx = ps.probe.index
            group_map.setdefault(g, []).append(ps.score)
            if idx not in index_map:
                index_map[idx] = {}
            index_map[idx][g] = ps.score

        group_means: dict[str, float] = {
            g: sum(vals) / len(vals) for g, vals in group_map.items() if vals
        }

        if group_means:
            max_mean = max(group_means.values())
            min_mean = min(group_means.values())
            gap = max_mean - min_mean
        else:
            gap = 0.0

        # Paired differences: mean over i of (max - min of scores at index i across groups)
        paired_diffs: list[float] = []
        for idx in sorted(index_map.keys()):
            idx_scores = list(index_map[idx].values())
            if idx_scores:
                paired_diffs.append(max(idx_scores) - min(idx_scores))

        mean_abs_paired_diff = sum(paired_diffs) / len(paired_diffs) if paired_diffs else 0.0
        flagged = gap > (threshold + 1e-12)

        rows.append(
            BiasReportRow(
                template_id=tid,
                attribute=attr,
                group_means=group_means,
                gap=gap,
                mean_abs_paired_diff=mean_abs_paired_diff,
                flagged=flagged,
            )
        )

    return BiasReport(threshold=threshold, rows=rows)


def to_dict(report: BiasReport) -> dict[str, Any]:
    """Serialize a BiasReport into a JSON-compatible dictionary."""
    return asdict(report)
