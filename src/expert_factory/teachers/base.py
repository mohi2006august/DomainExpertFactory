"""Teacher interface: drafts grid-cell examples and answers questions from passages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from expert_factory.schemas import Chunk, Draft, GridCell


class TeacherError(RuntimeError):
    """A teacher call failed in a way that retrying the same request will not fix."""


@dataclass(frozen=True)
class DraftRequest:
    cell: GridCell
    chunk: Chunk
    topic_label: str
    difficulty_description: str
    format_description: str
    persona_description: str
    hard_example: str | None = None


class Teacher(ABC):
    model: str
    is_paid: bool = False

    @abstractmethod
    def draft(self, request: DraftRequest) -> Draft | None:
        """Write one example for a grid cell, or None if the teacher declined."""

    @abstractmethod
    def answer(self, question: str, chunks: Sequence[Chunk], attempt: int = 0) -> str | None:
        """Answer a question from passages. ``attempt`` varies independent samples."""
