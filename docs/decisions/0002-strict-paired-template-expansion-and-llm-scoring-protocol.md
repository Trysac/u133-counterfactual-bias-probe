# Strict paired template expansion and LLM scoring protocol

Status: accepted (generated, pending human review)

## Context

Notebook-based bias probing often relies on ad-hoc string formatting, unpaired prompt sets, and loose regex parsing, causing silent bugs or unaligned group comparisons.

## Decision

Implement deterministic custom template parsing (without str.format), strict pairing validation across groups, and an explicit LLMPort/Scorer protocol parsing standardized score outputs.

## Consequences

Guarantees reproducible counterfactual evaluation without runtime dependencies while preventing template injection and misaligned paired comparisons.
