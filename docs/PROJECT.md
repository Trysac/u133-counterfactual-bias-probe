# Project record

## Objective

Expand templates over paired attribute values, score prompts, and report group gaps.

## Scope

- Strict template grammar parsing with explicit error reporting.
- Paired prompt expansion using provided attribute mapping.
- Scorer injection (LLM or custom callable) via `LLMPort` protocol.
- Telemetry via `on_event` callbacks.
- Disparity calculation (group means, gap, mean absolute paired difference).

## Out of Scope

- Statistical significance tests.
- Intersectional attribute analysis.
- Automated template or prompt generation.
- Async processing.

## Factory run u133 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | gemini-3.8-flash | yes | 4353 | 9709 | 0 | 0 | 0.0198 |
| fix-1 | gemini-3.8-flash | no | 45016 | 2059 | 0 | 172469 | 0.0544 |
| fix-2 | gemini-3.8-flash | no | 35814 | 404 | 0 | 89413 | 0.0351 |
| audit | gemini-3.1-flash-lite | no | 39288 | 614 | 0 | 81413 | 0.0128 |
| docs | gemini-3.1-flash-lite | no | 174570 | 1724 | 0 | 0 | 0.0462 |

Total: $0.1683. Prices: Anthropic list prices.
