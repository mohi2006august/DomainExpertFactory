# CLAUDE.md: Domain Expert Factory

Guidance for Claude Code (and other AI assistants) working in this project.

## Project Summary

An end-to-end factory that turns domain documents into a small, fast, quantised expert model: teacher-generated data, LoRA fine-tuning, overfitting guards and edge-ready quantisation. The factory ingests domain docs and a task spec. A frontier teacher LLM generates diverse Q&A/instruction data with rationales (distillation). A small open model is LoRA/QLoRA fine-tuned with continuous overfitting monitoring (held-out real data, canaries, dedupe), and the best checkpoint is merged, quantised (AWQ/GGUF) and evaluated again. Output: a deployable expert model with a report card.

## Read First

- `PRD.md`: what we are building and why; requirement IDs (FR-xx) are referenced in commits.
- `SYSTEM_ARCHITECTURE.md`: components, data flow and failure modes.
- `TECHNICAL.md`: stack, data models, APIs and algorithms.
- `DESIGN.md`: UX flows, screens and visual tokens.
- `BRAIN.md`: concept primers, experiments and the decision log.

## Commands

| Command | What it does |
|---|---|
| `uv sync` | Install |
| `dagster dev` | Pipeline UI |
| `factory run --spec examples/maintenance.yaml` | Example run |

## Code Conventions

- Each Dagster asset writes a `report.json` consumed by the report card.
- Keep functions small and typed; no silent `except`. Log with structured fields.
- Every new feature references its FR-ID in the PR description and ships with tests.
- Secrets only come from environment variables. Never hard-code keys or commit `.env`.

## Gotchas

- Merge LoRA into fp16 base before quantising, not into a 4-bit base.

## When Unsure

- Prefer the simplest design that satisfies the PRD; record bigger trade-offs in `BRAIN.md` → Decision Log.
- Run the eval / benchmark suite before claiming a quality or performance improvement.
- Ask before adding a new paid dependency, external service or data store.
