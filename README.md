# counterfactual-bias-probe

[![PyPI version](https://img.shields.io/pypi/v/trysac-counterfactual-bias-probe)](https://pypi.org/project/trysac-counterfactual-bias-probe/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-counterfactual-bias-probe)](LICENSE)
[![CI](https://github.com/Trysac/u133-counterfactual-bias-probe/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u133-counterfactual-bias-probe/actions/workflows/ci.yml)

Counterfactual bias probes: expand templates over paired attribute values, score each prompt through an injected scorer and report per-group gaps

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-counterfactual-bias-probe
# or
uv add trysac-counterfactual-bias-probe
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import counterfactual_bias_probe

# normalize() trims surrounding whitespace and lower-cases the text.
print(counterfactual_bias_probe.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    counterfactual_bias_probe.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
