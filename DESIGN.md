# Domain Expert Factory: Design Document

> An end-to-end factory that turns domain documents into a small, fast, quantised expert model: teacher-generated data, LoRA fine-tuning, overfitting guards and edge-ready quantisation.
>
> **Concepts:** `Fine Tuning`, `LoRA`, `Overfitting`, `Knowledge Distillation`, `Quantization`, `LLMs`

## 1. Design Principles

- Every stage has a gate and a human-readable report.
- Real data is the judge; synthetic data is the teacher.

## 2. Key User Flows

### Build an expert

1. Create spec.
2. Upload docs + seeds.
3. Approve generation budget.
4. Watch pipeline.
5. Review report card.
6. Download model.

## 3. Screens & Surfaces

| Screen | What it shows / does |
|---|---|
| Pipeline view | DAG with stage status and gates. |
| Data explorer | Synthetic examples by topic/difficulty; flagged items. |
| Training monitor | Real vs synthetic val curves with guard markers. |
| Report card | Scores, costs, memorisation, lineage. |

## 4. Wireframe (Primary Screen)

```text
Ingest ✔ → Generate ✔ ($212/$300) → Filter ✔ (18.4k kept) → Train ⚠ stopped @ep2 (guard)
→ Select ✔ ckpt-1400 → Merge ✔ → Quantize ✔ AWQ-4bit (−0.8 pts) → Eval ✔ → Package ✔
Report: domain 86.1 (teacher 94.0 · 91.6%) · 3.1GB · 42 tok/s RTX4060 · canaries 0/50
```

## 5. Visual Language

| Token | Value |
|---|---|
| Primary accent | `#7C3AED` (violet-700) |
| Neutral surface | `#0F1117` dark / `#FAFAF9` light |
| Success / Warning / Danger | `#22C55E` / `#F59E0B` / `#EF4444` |
| Type | Inter for UI, JetBrains Mono for code, tokens and numbers |
| Spacing scale | 4 / 8 / 12 / 16 / 24 / 32 px |
| Radius | 8px cards, 6px inputs, full for pills |

## 6. States

| State | Treatment |
|---|---|
| Guard stop | Amber stage with explanation and recommendations. |

## 7. Accessibility

- WCAG 2.2 AA contrast in both themes; never encode meaning with colour alone (pair with icon or label).
- Full keyboard navigation with visible focus rings; `/` focuses search, `?` opens shortcuts.
- Charts ship with a data-table fallback and `aria-describedby` summaries.
- Respect `prefers-reduced-motion`; streaming text is announced politely via `aria-live`.
