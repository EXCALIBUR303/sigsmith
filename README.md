# SIGSMITH

**Schema-conditioned Sigma detection-rule synthesis for air-gapped SOCs.**

A 1.7B Apache-2.0 model (Qwen3-1.7B), QLoRA-tuned and shipped as a pinned GGUF, reads a deployment's own
telemetry schema card and compiles an English threat-behaviour description into a Sigma detection rule —
scored by *executing* the rule against labelled malicious events and mutation-generated benign twins,
never by text similarity.

## Status

**Design phase.** The full technical blueprint is complete and adversarially reviewed; the implementation
has not started yet.

- [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md) — the complete 7-section technical blueprint: dataset engineering,
  QLoRA fine-tuning methodology, evaluation harness, quantization & profiling, serving architecture,
  repository structure & CI/CD, and README/resume presentation strategy. Includes an executive summary,
  build order (M0–M5), and risk register.
- [`docs/CONTRACT.json`](docs/CONTRACT.json) — the locked project contract the blueprint is built against:
  model choice and rationale, task I/O schema, dataset composition, canonical file paths, metric symbol
  definitions, hardware profiles, and naming conventions.

## Why offline

In an accredited enclave (SCIF, CUI boundary, ITAR programme, an OT segment behind a data diode) there is
no API egress at any price tier — and the telemetry schema card itself is disclosure-sensitive, since a
precise list of what a deployment collects is equivalently a map of what it does not. See the blueprint's
Section 1 for the full argument against the two runner-up problem domains that were scoped and rejected.
