# Domain Expert Factory

> An end-to-end factory that turns domain documents into a small, fast, quantised expert model: teacher-generated data, LoRA fine-tuning, overfitting guards and edge-ready quantisation.

**Concepts:** `Fine Tuning`, `LoRA`, `Overfitting`, `Knowledge Distillation`, `Quantization`, `LLMs`

The factory ingests domain docs and a task spec. A frontier teacher LLM generates diverse Q&A/instruction data with rationales (distillation). A small open model is LoRA/QLoRA fine-tuned with continuous overfitting monitoring (held-out real data, canaries, dedupe), and the best checkpoint is merged, quantised (AWQ/GGUF) and evaluated again. Output: a deployable expert model with a report card.

## Docs

| File | Contents |
|---|---|
| [BRAIN.md](BRAIN.md) | Concept primers, research, experiments, decision log |
| [CLAUDE.md](CLAUDE.md) | Instructions for AI coding assistants |
| [DESIGN.md](DESIGN.md) | UX flows, screens, wireframes, visual language |
| [PRD.md](PRD.md) | Problem, users, requirements, metrics, roadmap |
| [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) | Components, data flow, deployment, failure modes |
| [TECHNICAL.md](TECHNICAL.md) | Stack, data models, APIs, algorithms, setup |
