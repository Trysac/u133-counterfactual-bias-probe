# counterfactual-bias-probe

Problem: Teams struggle to measure AI bias manually with str.format and notebooks; probe sets become brittle, and unpaired comparisons hide the effect of counterfactual swaps. This library provides a strict, typed system to expand templates into paired prompts, score them, and analyze disparities.

## Usage

```python
from counterfactual_bias_probe import FakeLLM, LLMScorer, expand, run_probes, summarize

templates = ["{name} applied for a loan."]
attrs = {"name": {"group_a": ["Emily"], "group_b": ["Jamal"]}}
probes = expand(templates, attrs)
scorer = LLMScorer(FakeLLM(lambda msgs: "SCORE: 0.8"))
scores = run_probes(probes, scorer)
report = summarize(scores, threshold=0.1)
print(f"Gap: {report.rows[0].gap}, Flagged: {report.rows[0].flagged}")
```

## API Summary

- Functions: `expand`, `run_probes`, `summarize`, `to_dict`.
- Exceptions: `ProbeError` (codes: invalid_template, unknown_placeholder, multiple_attributes, unequal_groups, too_many_prompts, invalid_score, input_too_large), `RecordingExhaustedError`.
- Limits: Template characters <= 2,000; Value characters <= 200; Max prompts default 10,000.

## Why not ad-hoc notebooks?

Notebooks lack strict template grammars and automated validation. This library ensures exactly one attribute per template, enforces group size parity, and computes both aggregate group gaps and precise paired differences to isolate specific counterfactual effects.

## Security

See [docs/SECURITY.md](docs/SECURITY.md).