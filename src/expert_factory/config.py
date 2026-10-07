"""Factory spec: the TaskSpec plus per-stage settings, loaded from a YAML file."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from expert_factory.resources import load_config, load_prompt_yaml
from expert_factory.schemas import Difficulty, TaskSpec


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Section):
    docs: Path
    seeds: Path
    extensions: list[str] = Field(default_factory=lambda: [".md", ".txt", ".pdf"])
    chunk_words: int = Field(180, ge=40)
    chunk_overlap: int = Field(30, ge=0)
    min_seeds: int = Field(50, ge=4)
    val_fraction: float = Field(0.5, gt=0, lt=1)

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> DataConfig:
        if self.chunk_overlap >= self.chunk_words:
            raise ValueError("chunk_overlap must be smaller than chunk_words")
        return self


class CoverageConfig(_Section):
    n_topics: int = Field(8, ge=1)
    top_terms: int = Field(3, ge=1)


class TeacherOptions(_Section):
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    max_tokens: int = Field(4096, ge=256)
    concurrency: int = Field(4, ge=1)
    fallbacks: bool = True
    max_retries: int = Field(4, ge=0)
    # Use the Message Batches API (half price, asynchronous) for generation and votes.
    batch: bool = False
    # Offline teacher only: share of drafts deliberately answered from the wrong
    # passage, so the grounding filter has realistic teacher mistakes to catch.
    offline_error_rate: float = Field(0.05, ge=0, le=1)


class GenerationConfig(_Section):
    n_examples: int = Field(1000, ge=1)
    difficulties: list[Difficulty] = Field(default_factory=lambda: ["easy", "medium", "hard"])
    formats: list[str] | dict[str, str] = Field(
        default_factory=lambda: ["direct_question", "troubleshooting", "procedure", "definition"]
    )
    personas: list[str] | dict[str, str] = Field(
        default_factory=lambda: [
            "apprentice_technician",
            "senior_engineer",
            "safety_officer",
            "plant_manager",
        ]
    )
    max_per_chunk: int = Field(6, ge=1)
    votes: int = Field(3, ge=0)
    agreement_threshold: float = Field(0.35, ge=0, le=1)
    use_feedback: bool = True

    def descriptions(self, axis: Literal["formats", "personas"]) -> dict[str, str]:
        """Resolve names to descriptions, falling back to prompts/generation_grid.yaml."""
        defaults: dict[str, str] = load_prompt_yaml("generation_grid")[axis]
        value = getattr(self, axis)
        if isinstance(value, dict):
            return {k: v or defaults.get(k, k.replace("_", " ")) for k, v in value.items()}
        return {k: defaults.get(k, k.replace("_", " ")) for k in value}


class FilterConfig(_Section):
    min_votes_agree: int = Field(2, ge=0)
    grounding_threshold: float = Field(0.5, ge=0, le=1)
    dedupe_threshold: float = Field(0.8, gt=0, le=1)
    leakage_threshold: float = Field(0.5, gt=0, le=1)
    shingle_size: int = Field(3, ge=1)
    num_perm: int = Field(128, ge=16)
    synth_val_fraction: float = Field(0.1, gt=0, lt=0.5)
    n_canaries: int = Field(10, ge=0)


class TrainingConfig(_Section):
    backend: Literal["auto", "tinylm", "hf"] = "auto"
    rank_sweep: list[int] = Field(default_factory=list)
    epochs: int | None = Field(None, ge=1)
    learning_rate: float | None = Field(None, gt=0)
    batch_size: int | None = Field(None, ge=1)
    evals_per_epoch: int = Field(1, ge=1)
    overrides: dict[str, Any] = Field(default_factory=dict)


class GuardConfig(_Section):
    patience: int = Field(2, ge=1)  # consecutive real-val-loss increases before stopping
    min_delta: float = Field(0.0, ge=0)
    max_gap_increase: float = Field(0.5, gt=0)  # delta on (real - synthetic) val-loss gap
    max_canary_hit_rate: float = Field(0.3, ge=0, le=1)
    stop_on_canary: bool = False
    canary_alternatives: int = Field(200, ge=10)


class QuantizeConfig(_Section):
    epsilon: float = Field(2.0, ge=0)  # allowed score drop in points
    ladder: list[str] | None = None  # backend default when None
    awq: bool = True
    gate_max_examples: int = Field(100, ge=1)


class EvalConfig(_Section):
    judge: Literal["auto", "lexical", "llm"] = "auto"
    judge_model: str = "claude-opus-5-5"
    max_new_tokens: int = Field(96, ge=8)
    retrieval_k: int = Field(3, ge=1)
    extraction_probes: int = Field(30, ge=0)
    bias_probes: int = Field(20, ge=0)
    general_tasks: list[str] = Field(default_factory=list)
    general_limit: int = Field(100, ge=1)


class ReleaseConfig(_Section):
    min_student_teacher_ratio: float = Field(0.90, ge=0)
    require_quality_target: bool = False
    max_canary_extractions: int = Field(0, ge=0)
    max_cost_ratio: float = Field(0.05, ge=0)


class FactorySpec(_Section):
    task: TaskSpec
    data: DataConfig
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)
    teacher: TeacherOptions = Field(default_factory=TeacherOptions)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    quantize: QuantizeConfig = Field(default_factory=QuantizeConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    release: ReleaseConfig = Field(default_factory=ReleaseConfig)
    seed: int = 42

    @model_validator(mode="after")
    def _known_device(self) -> FactorySpec:
        devices = load_config("devices")
        if self.task.target_device not in devices:
            raise ValueError(
                f"unknown target_device {self.task.target_device!r}; "
                f"choose one of {sorted(devices)}"
            )
        return self

    # ----------------------------------------------------------------- helpers
    def backend(self) -> Literal["tinylm", "hf"]:
        if self.training.backend != "auto":
            return self.training.backend
        info = base_model_info(self.task.base_model)
        return "tinylm" if info.get("backend") == "tinylm" else "hf"

    def device(self) -> dict[str, Any]:
        return dict(load_config("devices")[self.task.target_device])

    def section_hash(self, *sections: str) -> str:
        payload = {s: self.model_dump(mode="json")[s] for s in sections}
        payload["seed"] = self.seed
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def base_model_info(name: str) -> dict[str, Any]:
    """Look up a base model in configs/base_models.yaml, inferring hf defaults."""
    known = load_config("base_models")
    if name in known:
        return dict(known[name])
    return {"backend": "hf", "params_b": _guess_params_b(name), "chat": True}


def _guess_params_b(name: str) -> float:
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", name.replace("-", " "))
    return float(match.group(1)) if match else 3.0


def size_class(params_b: float) -> str:
    if params_b < 1.0:
        return "xs"
    if params_b <= 4.5:
        return "s"
    return "m"


def effective_budget(spec: FactorySpec) -> float:
    """The lower of the spec budget and the FACTORY_BUDGET_USD environment cap."""
    budget = spec.task.teacher_budget_usd
    env_cap = os.environ.get("FACTORY_BUDGET_USD", "").strip()
    if env_cap:
        try:
            budget = min(budget, float(env_cap))
        except ValueError as exc:
            raise ValueError(f"FACTORY_BUDGET_USD must be a number, got {env_cap!r}") from exc
    return budget


def load_spec(path: str | Path) -> FactorySpec:
    """Load a spec YAML file; data paths are resolved relative to the file."""
    spec_path = Path(path).resolve()
    with spec_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{spec_path} must contain a mapping")
    data = raw.get("data", {})
    for key in ("docs", "seeds"):
        if key in data:
            data[key] = str((spec_path.parent / data[key]).resolve())
    return FactorySpec.model_validate(raw)


def dump_spec(spec: FactorySpec, path: Path) -> None:
    path.write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
