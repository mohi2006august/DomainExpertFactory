# Domain Expert Factory: Product Requirements Document

> An end-to-end factory that turns domain documents into a small, fast, quantised expert model: teacher-generated data, LoRA fine-tuning, overfitting guards and edge-ready quantisation.
>
> **Concepts:** `Fine Tuning`, `LoRA`, `Overfitting`, `Knowledge Distillation`, `Quantization`, `LLMs`

## 1. Overview

The factory ingests domain docs and a task spec. A frontier teacher LLM generates diverse Q&A/instruction data with rationales (distillation). A small open model is LoRA/QLoRA fine-tuned with continuous overfitting monitoring (held-out real data, canaries, dedupe), and the best checkpoint is merged, quantised (AWQ/GGUF) and evaluated again. Output: a deployable expert model with a report card.

## 2. Problem Statement

Companies want a private, cheap domain model (medical coding, legal clause review, industrial maintenance) but lack labelled data and ML expertise. The pipeline (synthetic data → fine-tune → evaluate → compress → deploy) has many failure points, especially overfitting on small synthetic datasets.

## 3. Target Users & Personas

| Persona | Needs |
|---|---|
| Domain team lead | Wants a private expert model without an ML team. |
| ML engineer | Wants a reliable pipeline with guardrails. |
| Compliance | Needs model report card and data lineage. |

## 4. Goals

- Expert student ≥ 90% of teacher on domain evals at ≤ 5% of cost.
- Automated overfitting detection and checkpoint selection.
- Model under 6GB running on a single consumer GPU or CPU.

## 5. Non-Goals

- General chat assistants.
- Replacing RAG for fast-changing facts.

## 6. User Stories

- As a lead, I upload 2,000 maintenance manuals and 100 real Q&A examples, and receive a 3B model scoring 88% of the teacher.
- As ML engineer, I see the factory stopped at epoch 2 because the real-data val loss diverged.
- As compliance, I download a report card: data sources, synthetic ratio, eval results, memorisation tests.

## 7. Functional Requirements

| ID | Priority | Feature | Description |
|---|---|---|---|
| FR-01 | P0 | Task spec & ingestion | Docs + task definition + seed examples; chunking and topic coverage map. |
| FR-02 | P0 | Teacher data generation | Diverse instructions via topic × difficulty × format grid; rationales; self-consistency filters. |
| FR-03 | P0 | LoRA/QLoRA training | Presets per base size; rank sweep. |
| FR-04 | P0 | Overfitting guard | Real held-out set, synthetic/real gap tracking, near-dup leakage checks, early stopping. |
| FR-05 | P1 | Quantisation stage | Merge LoRA → AWQ int4 and GGUF Q4/Q5; re-evaluate after quantisation. |
| FR-06 | P1 | Report card | Quality, cost, latency, memorisation, bias checks, lineage. |
| FR-07 | P2 | Continuous improvement | Production feedback → hard examples → next iteration. |

## 8. Non-Functional Requirements

- Full lineage (data, config, seeds).
- Pipeline resumable per stage.
- Budget cap on teacher spend.

## 9. Success Metrics

| Metric | Target |
|---|---|
| Student/teacher domain score | ≥ 0.90 |
| Cost per 1k answers vs teacher | ≤ 5% |
| Overfitting incidents reaching release | 0 |

## 10. Release Plan

| Phase | Scope |
|---|---|
| M1 | Teacher data gen + LoRA SFT + eval. |
| M2 | Overfitting guard + checkpoint selection. |
| M3 | Merge + quantise + post-quant eval. |
| M4 | Report card + feedback loop. |

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Synthetic data lacks diversity → overfit to teacher style | Diversity grid, dedupe, real held-out val, style-varied prompts. |
| Quantisation erodes domain skill | Post-quant eval gate; fall back to Q5/Q8. |
| Teacher mistakes | Self-consistency voting, citation to source chunk required. |

## 12. Open Questions

- Include DPO stage using teacher-as-judge preferences?
