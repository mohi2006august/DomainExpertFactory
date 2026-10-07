"""Teacher LLMs used for distillation data, votes and the teacher baseline."""

from __future__ import annotations

from collections.abc import Sequence

from expert_factory.budget import BudgetMeter
from expert_factory.config import FactorySpec
from expert_factory.schemas import Chunk
from expert_factory.teachers.base import DraftRequest, Teacher, TeacherError
from expert_factory.teachers.offline import OfflineTeacher

__all__ = ["DraftRequest", "OfflineTeacher", "Teacher", "TeacherError", "make_teacher"]


def make_teacher(spec: FactorySpec, chunks: Sequence[Chunk], meter: BudgetMeter) -> Teacher:
    if spec.task.teacher_model == "offline":
        return OfflineTeacher(chunks, seed=spec.seed, error_rate=spec.teacher.offline_error_rate)
    if spec.task.teacher_model.startswith("claude-"):
        from expert_factory.teachers.anthropic_teacher import AnthropicTeacher

        return AnthropicTeacher(spec.task.teacher_model, spec.task, spec.teacher, meter)
    raise ValueError(
        f"unsupported teacher_model {spec.task.teacher_model!r}: use 'offline' or a claude-* model"
    )
