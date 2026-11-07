# Testing

## Tests

Existing tests in `tests/test_counterfactual_bias_probe.py` cover:

- Template parsing: brace handling, placeholder validation, multi-attribute rejection.
- Expansion: paired generation logic, group parity enforcement, prompt limit handling.
- Scoring: LLMScorer score extraction (regex parsing, boundary checks), error handling for invalid replies.
- Analysis: `summarize` logic for group means, gap calculation, and threshold-based flagging.