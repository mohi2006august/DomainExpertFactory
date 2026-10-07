"""Answer graders: a free lexical judge and a rubric-based LLM judge (0-100 scale)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from expert_factory.resources import load_prompt_yaml, render_prompt
from expert_factory.text import rouge_l, token_f1

JUDGE_SCHEMA_TEMPLATE: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
}


class Judge(ABC):
    name: str

    @abstractmethod
    def score(self, question: str, reference: str, candidate: str) -> float:
        """Score a candidate answer from 0 to 100 against the reference."""


class LexicalJudge(Judge):
    """Mean of content-word F1 and ROUGE-L. Cheap, deterministic, style-sensitive."""

    name = "lexical"

    def score(self, question: str, reference: str, candidate: str) -> float:
        if not candidate.strip():
            return 0.0
        return 100.0 * 0.5 * (token_f1(candidate, reference) + rouge_l(candidate, reference))


class LLMJudge(Judge):
    """Rubric judge (prompts/judge_rubric.yaml) run on the teacher API with the budget meter."""

    name = "llm"

    def __init__(self, teacher: Any) -> None:
        self.teacher = teacher
        self.rubric = load_prompt_yaml("judge_rubric")
        criteria = self.rubric["criteria"]
        self.schema = {
            **JUDGE_SCHEMA_TEMPLATE,
            "properties": {
                **{name: {"type": "integer", "enum": [1, 2, 3, 4, 5]} for name in criteria},
                "explanation": {"type": "string"},
            },
            "required": [*criteria, "explanation"],
        }
        self.system = "You are a strict, fair grader of domain-expert answers."
        self._fallback = LexicalJudge()

    def score(self, question: str, reference: str, candidate: str) -> float:
        if not candidate.strip():
            return 0.0
        prompt = render_prompt(
            "judge.j2",
            question=question,
            reference=reference,
            candidate=candidate,
            rubric=self.rubric,
        )
        data = self.teacher.call_json(prompt, self.schema, "judge", system=self.system)
        if data is None:  # refusal / truncation: fall back rather than drop the item
            return self._fallback.score(question, reference, candidate)
        criteria = self.rubric["criteria"]
        total_weight = sum(float(c["weight"]) for c in criteria.values())
        weighted = sum(float(c["weight"]) * int(data[name]) for name, c in criteria.items())
        return 100.0 * ((weighted / total_weight) - 1.0) / 4.0
