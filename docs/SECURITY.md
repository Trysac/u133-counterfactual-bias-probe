# Security notes

## Threat model

- Defended: Supply chain attacks via runtime dependency elimination; untrusted input via `ProbeError` validation; credential leaks via git-ignored local configuration.
- Not defended: The library does not sanitize model output beyond score parsing (i.e., downstream prompt injection inside the LLM reply is the responsibility of the system using the library). It does not sanitize LLM inputs against injection beyond template length limits.