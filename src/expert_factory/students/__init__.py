"""Student model backends: ``tinylm`` (NumPy, CPU) and ``hf`` (TRL + PEFT)."""

from __future__ import annotations

from pathlib import Path

from expert_factory.config import FactorySpec, base_model_info
from expert_factory.students.base import (
    Hyperparams,
    QuantizerUnavailableError,
    StudentBackend,
    StudentModel,
    TrainData,
)

__all__ = [
    "Hyperparams",
    "QuantizerUnavailableError",
    "StudentBackend",
    "StudentModel",
    "TrainData",
    "make_backend",
]


def make_backend(spec: FactorySpec, work_dir: Path) -> StudentBackend:
    """Backend for the spec's base model; ``work_dir`` holds base weights / caches."""
    info = base_model_info(spec.task.base_model)
    if spec.backend() == "tinylm":
        from expert_factory.students.tinylm import TinyLMBackend

        return TinyLMBackend(info, seed=spec.seed, work_dir=work_dir)
    from expert_factory.students.hf import HFBackend

    return HFBackend(spec.task.base_model, info, seed=spec.seed, work_dir=work_dir)
