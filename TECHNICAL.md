# Domain Expert Factory: Technical Specification

> An end-to-end factory that turns domain documents into a small, fast, quantised expert model: teacher-generated data, LoRA fine-tuning, overfitting guards and edge-ready quantisation.
>
> **Concepts:** `Fine Tuning`, `LoRA`, `Overfitting`, `Knowledge Distillation`, `Quantization`, `LLMs`

## 1. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | Dagster | Assets + lineage. |
| Training | TRL, PEFT, bitsandbytes | LoRA/QLoRA. |
| Quant | AutoAWQ, llama.cpp | GPU + CPU targets. |
| Eval | lm-eval-harness + custom judge | Standard + domain. |
| Teacher | Frontier LLM API | Quality data. |

## 2. Repository Layout

```text
expert-factory/
├── pipeline/assets/   # ingest.py, generate.py, filter.py, train.py, guard.py, quantize.py, evaluate.py, package.py
├── prompts/           # generation grids, judge rubrics
├── configs/           # base models, lora presets, budgets
└── reports/templates/ # report card
```

## 3. Data Models

```python
class TaskSpec(BaseModel):
    name: str
    description: str
    output_format: Literal["answer", "json", "classification"]
    base_model: str             # e.g. Qwen2.5-3B-Instruct
    teacher_model: str
    teacher_budget_usd: float
    target_device: str          # "rtx4060-8gb" | "cpu-16gb"

class GenExample(BaseModel):
    instruction: str
    response: str
    rationale: str
    source_chunk_ids: list[str]
    topic: str
    difficulty: Literal["easy", "medium", "hard"]
    votes_agree: int
```

## 4. API Surface

| Method | Endpoint / Tool | Purpose |
|---|---|---|
| cli | factory init --spec spec.yaml | Create project. |
| cli | factory run --until package | Run pipeline. |
| cli | factory report | Report card. |
| POST | /v1/projects/{id}/runs | API trigger. |

## 5. Core Algorithms

### Diversity grid

Sample (topic cluster × difficulty × format × persona) cells uniformly; cap examples per source chunk to avoid over-representation.

### Guarded early stopping

Primary signal: loss on real held-out val. Stop when it increases for 2 consecutive evals or synthetic/real gap > δ. Restore best checkpoint.

### Post-quant gate

Accept quant q if `score(q) ≥ score(fp16) − ε` (ε=2 pts). Otherwise step up precision (Q4→Q5→Q8).

## 6. Configuration

| Variable | Purpose |
|---|---|
| TEACHER_API_KEY | Teacher |
| HF_TOKEN | Base models |
| FACTORY_BUDGET_USD | Spend cap |

## 7. Testing Strategy

- Stage contract tests (schemas between assets).
- Tiny end-to-end with 0.5B model in CI.

## 8. Performance Budgets

- 3B QLoRA on 20k examples < 2h on 1×A100.
- Quantised output ≥ 30 tok/s on RTX 4060.

## 9. Local Setup

```bash
uv sync
dagster dev
factory run --spec examples/maintenance.yaml
```
