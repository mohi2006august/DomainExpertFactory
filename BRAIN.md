# Domain Expert Factory: Brain

> Second brain for this project: concepts, mental models, research, ideas and decisions. Keep it messy and alive.

## 1. Why This Project

It's the full model-customisation lifecycle in one project and shows how distillation, fine-tuning, LoRA, overfitting control and quantisation interact.

## 2. Concept Primers

### Fine Tuning

Continuing training of a pre-trained model on task- or domain-specific data to change its behaviour, style or knowledge.

- **SFT**: Supervised fine-tuning on prompt → ideal response pairs.
- **DPO / preference tuning**: Optimising directly on chosen-vs-rejected pairs without a separate reward model.
- **Catastrophic forgetting**: Losing general capabilities while specialising.
- **PEFT**: Parameter-efficient methods (LoRA, adapters, prefix tuning) that train a small fraction of weights.

> **Mental model:** Fine-tune for form (format, tone, task behaviour). Retrieve for facts. Data quality beats data quantity almost every time.

### LoRA

Low-Rank Adaptation: freeze the base weights W and learn a low-rank update ΔW = B·A (rank r ≪ d), training well under 1% of parameters with small, swappable adapters.

- **Rank r / alpha**: Capacity of the update and its scaling (alpha/r).
- **Target modules**: Which projections get adapters (q,k,v,o, MLP).
- **QLoRA**: LoRA on top of a 4-bit quantised base model; fine-tune big models on one GPU.
- **Multi-LoRA serving**: One base model in memory serving many adapters per request (S-LoRA, Punica).

> **Mental model:** The base model is the OS and adapters are apps. They are cheap to train, store and switch, so give each tenant, task or style its own adapter.

### Overfitting

When a model fits noise or specifics of the training set so closely that it performs worse on unseen data; train loss keeps dropping while validation loss rises.

- **Generalisation gap**: Difference between training and validation/test performance.
- **Regularisation**: Weight decay, dropout, data augmentation, early stopping, smaller capacity.
- **Memorisation**: Verbatim reproduction of training examples; measured with canaries or extraction probes.
- **Data leakage**: Test information leaking into training, which hides overfitting behind inflated scores.

> **Mental model:** A low training loss is a claim. The held-out set is the evidence. Always keep a test set you never tune against.

### Knowledge Distillation

Training a small 'student' model to mimic a large 'teacher' model's outputs (soft labels, logits, rationales), keeping most of the quality at a fraction of the cost.

- **Soft targets**: Teacher probability distributions softened with temperature T carry 'dark knowledge'.
- **KD loss**: KL divergence between teacher and student distributions, mixed with hard-label cross-entropy.
- **Sequence-level KD**: Student trained on teacher-generated outputs (synthetic data distillation).
- **Capacity gap**: Very large teacher/student gaps can hurt; teacher assistants bridge them.

> **Mental model:** You are compressing a function, not a dataset. The teacher's uncertainty is signal, so preserve it.

### Quantization

Representing weights (and sometimes activations/KV cache) with fewer bits (int8, int4, fp8) to cut memory and increase speed, with some accuracy loss.

- **PTQ vs QAT**: Post-training quantisation vs quantisation-aware training.
- **GPTQ / AWQ**: Weight-only int4 methods using second-order info or activation-aware scaling.
- **GGUF**: llama.cpp file format with k-quants (Q4_K_M, Q5_K_M, …) for CPU/edge inference.
- **Outliers**: A few large-magnitude activation channels that break naive quantisation.

> **Mental model:** Memory bandwidth, not FLOPs, bounds LLM decoding. Halving bytes per weight roughly doubles tokens/sec until you hit compute limits.

### LLMs

Large Language Models: transformer networks trained on large text corpora to predict the next token, then aligned (SFT and RLHF/RLAIF) to follow instructions and use tools.

- **Prompting**: System prompts, few-shot examples, structured outputs, tool use.
- **Sampling**: Temperature, top-p and max tokens shape determinism and length.
- **Evals**: Task-specific test sets plus LLM-as-judge with calibrated rubrics.
- **Routing**: Sending each request to the cheapest model that meets its quality bar.

> **Mental model:** Treat the LLM as a probabilistic component. Wrap it with evals, guardrails, retries and observability, the same way you would a flaky network dependency.

## 3. What I Want to Learn

- [ ] Synthetic data generation quality.
- [ ] Guarded training.
- [ ] Compression after fine-tuning.
- [ ] Model report cards.

## 4. Reading & Research List

- Wang et al., *Self-Instruct* (2022)
- Mukherjee et al., *Orca* (2023)
- Gunasekar et al., *Textbooks Are All You Need* (2023)
- Mitchell et al., *Model Cards for Model Reporting* (2019)

## 5. Experiments to Run

| # | Hypothesis | How to test |
|---|---|---|
| 1 | Diversity grid reduces overfitting vs naive generation | Compare gap curves. |
| 2 | QLoRA-then-AWQ vs LoRA-then-GGUF quality | Compare post-quant scores. |

## 6. Ideas Backlog

- Multi-teacher consensus.
- Tool-use experts.
- Auto-refresh when docs change.

## 7. Decision Log

| Date | Decision | Reason |
|---|---|---|
| YYYY-MM-DD | Real held-out data mandatory | Synthetic-only validation hid overfitting in early tests. |

## 8. Scratchpad

- _Dump raw thoughts, links and half-formed ideas here._
