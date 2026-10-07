"""Student backend interface: train LoRA, merge, quantise and load models for eval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from expert_factory.schemas import Canary, Chunk, GenExample, SeedExample, TrainRunResult

if TYPE_CHECKING:
    from expert_factory.pipeline.assets.guard import OverfitGuard


@dataclass
class TrainData:
    train: list[GenExample]
    synth_val: list[GenExample]
    real_val: list[SeedExample]
    canaries: list[Canary]
    chunks: list[Chunk]


@dataclass
class Hyperparams:
    rank: int
    alpha: float
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    evals_per_epoch: int = 1
    grad_accum: int = 1
    max_length: int = 1024
    qlora: bool = False
    target_modules: list[str] = field(default_factory=list)
    canary_alternatives: int = 200
    seed: int = 42


class StudentModel(ABC):
    """A loaded model variant (fp16 merged, int4, GGUF Q4_K_M, ...) ready for eval."""

    label: str
    path: Path

    @abstractmethod
    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        """Greedy answers for a batch of instructions."""

    @abstractmethod
    def canary_ranks(self, canaries: Sequence[Canary], n_alternatives: int, seed: int) -> list[int]:
        """Rank (1 = most likely) of each true secret among random alternatives."""

    @abstractmethod
    def mean_nll(self, pairs: Sequence[tuple[str, str]]) -> float:
        """Mean per-token negative log-likelihood of responses given instructions."""

    def general_perplexity(self) -> float | None:
        """Perplexity on general (non-domain) text, if the backend has a probe set."""
        return None

    @property
    def size_bytes(self) -> int:
        from expert_factory.artifacts import dir_size_bytes

        return dir_size_bytes(self.path)


class StudentBackend(ABC):
    name: str

    @abstractmethod
    def prepare(self, data: TrainData, work_dir: Path) -> dict[str, Any]:
        """One-off setup before any training run (tokenizer, base weights)."""

    @abstractmethod
    def train(
        self, data: TrainData, hp: Hyperparams, guard: OverfitGuard, out_dir: Path
    ) -> TrainRunResult:
        """Train one LoRA run under the guard; best checkpoint path in the result."""

    @abstractmethod
    def merge(self, checkpoint: Path, out_dir: Path) -> Path:
        """Merge the adapter into a full-precision (fp16) copy of the base."""

    @abstractmethod
    def quantize(self, merged: Path, fmt: str, out_dir: Path, calibration: Sequence[str]) -> Path:
        """Produce a quantised artefact in ``fmt``; raises QuantizerUnavailableError."""

    @abstractmethod
    def load(self, artifact: Path, label: str) -> StudentModel:
        """Load a merged or quantised artefact for evaluation."""

    @abstractmethod
    def load_base(self) -> StudentModel:
        """The untouched base model, for forgetting / general-regression checks."""

    @abstractmethod
    def quant_families(self, device: dict[str, Any]) -> dict[str, list[str]]:
        """Precision ladders per format family, in device preference order."""


class QuantizerUnavailableError(RuntimeError):
    """The toolchain for a quantisation format is not installed on this machine."""
