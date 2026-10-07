"""Data contracts passed between pipeline stages (TECHNICAL.md section 3).

Every stage reads and writes these models as JSON / JSONL, so they are the
schemas checked by the stage contract tests.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Difficulty = Literal["easy", "medium", "hard"]
OutputFormat = Literal["answer", "json", "classification"]
StageStatus = Literal["pending", "running", "success", "cached", "failed", "blocked", "skipped"]


class TaskSpec(BaseModel):
    """What expert we are building and under which constraints."""

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str
    output_format: OutputFormat = "answer"
    base_model: str  # e.g. Qwen/Qwen2.5-3B-Instruct
    teacher_model: str  # e.g. claude-opus-5-5, or "offline"
    teacher_budget_usd: float = Field(ge=0)
    target_device: str  # "rtx4060-8gb" | "cpu-16gb"


class Document(BaseModel):
    id: str
    path: str
    sha256: str
    n_words: int


class Chunk(BaseModel):
    id: str
    doc_id: str
    heading: str
    text: str
    n_words: int
    topic: str | None = None


class Topic(BaseModel):
    key: str  # "t03"
    label: str  # top terms, e.g. "seal / pump / leak"
    n_chunks: int
    n_seeds: int = 0
    terms: list[str] = Field(default_factory=list)


class SeedExample(BaseModel):
    """A real, human-written example. Real data is the judge (DESIGN.md)."""

    id: str
    question: str
    answer: str
    topic: str | None = None
    source: Literal["seed", "feedback"] = "seed"


class GridCell(BaseModel):
    topic: str
    difficulty: Difficulty
    format: str
    persona: str


class Draft(BaseModel):
    """Raw teacher output for one grid cell, before voting."""

    instruction: str
    response: str
    rationale: str
    source_chunk_ids: list[str]


class GenExample(BaseModel):
    """A synthetic training example (TECHNICAL.md section 3, plus lineage fields)."""

    instruction: str
    response: str
    rationale: str
    source_chunk_ids: list[str]
    topic: str
    difficulty: Difficulty
    votes_agree: int
    # lineage / grid fields
    id: str
    format: str = "direct_question"
    persona: str = "default"
    votes_total: int = 0
    teacher_model: str = "unknown"
    is_canary: bool = False


class RejectedExample(BaseModel):
    example: GenExample
    reasons: list[str]


class Canary(BaseModel):
    """A unique secret planted once in the training set to measure memorisation."""

    id: str
    instruction: str
    prefix: str
    secret: str

    @property
    def response(self) -> str:
        return f"{self.prefix} {self.secret}."


class EvalPoint(BaseModel):
    step: int
    epoch: float
    train_loss: float
    real_val_loss: float
    synth_val_loss: float
    gap: float  # real_val_loss - synth_val_loss
    canary_hit_rate: float | None = None


class GuardEvent(BaseModel):
    step: int
    epoch: float
    kind: Literal["best", "alert", "stop"]
    reason: str
    detail: dict[str, Any] = Field(default_factory=dict)


class TrainRunResult(BaseModel):
    rank: int
    alpha: float
    curve: list[EvalPoint]
    events: list[GuardEvent]
    stopped_early: bool
    stop_reason: str | None
    best_step: int
    best_epoch: float
    best_real_val_loss: float
    best_checkpoint: str
    trainable_params: int
    total_params: int
    duration_s: float


class GateResult(BaseModel):
    name: str
    passed: bool
    blocking: bool = True
    detail: str = ""


class StageReport(BaseModel):
    """Written by every stage as report.json and consumed by the report card."""

    stage: str
    status: StageStatus
    started_at: str
    finished_at: str
    duration_s: float
    fingerprint: str = ""
    summary: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    gates: list[GateResult] = Field(default_factory=list)


class FeedbackItem(BaseModel):
    """Production feedback on a deployed expert (FR-07)."""

    question: str
    model_answer: str
    rating: int = Field(ge=1, le=5)
    correction: str | None = None
    topic: str | None = None
