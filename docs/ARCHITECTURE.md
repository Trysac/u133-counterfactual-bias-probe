# Architecture

`counterfactual_bias_probe` is a deterministic, dependency-free reference library designed to perform counterfactual fairness probe expansions, evaluations, and disparity analysis.

## Components

1. **Template Parsing & Expansion (`expand`)**:
   - Tokenizes templates with custom brace handling (`{{` and `}}` are literals; `{name}` is a placeholder).
   - Strictly rejects invalid templates, ambiguous syntax, or multiple attributes per template.
   - Generates paired `Probe` objects across demographic or counterfactual groups.

2. **Scoring Protocol & LLM Scorer (`LLMPort`, `Scorer`, `LLMScorer`, `RecordedLLM`, `FakeLLM`)**:
   - Adapts to any scoring engine conforming to `Scorer`.
   - `LLMScorer` formats prompts with a standardized system instruction and extracts continuous scores in `[0.0, 1.0]`.

3. **Evaluation Pipeline (`run_probes`)**:
   - Evaluates probes through a scorer and validates returned scores.
   - Emits structured telemetry events without leaking attribute values or probe texts.

4. **Disparity Analysis (`summarize`, `to_dict`)**:
   - Computes group means, maximal group disparity (`gap`), and mean absolute paired differences across matching indices.
   - Flags disparities exceeding a configured threshold with float precision tolerance.