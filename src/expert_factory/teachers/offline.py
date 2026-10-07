"""Deterministic, extractive offline teacher.

It needs no API key and costs nothing, which makes the full pipeline runnable in CI
and on a laptop. Answers are copied from the source passage, so they are grounded by
construction; a configurable share of drafts is answered from the *wrong* passage to
mimic teacher mistakes that the filter stage has to catch.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from expert_factory.schemas import Chunk, Draft
from expert_factory.teachers.base import DraftRequest, Teacher
from expert_factory.text import content_words, sentences

_FORMAT_CUES = {
    "procedure": re.compile(
        r"\b(step|remove|install|replace|tighten|check|open|close|drain|fill|torque|isolate|"
        r"lock|apply|clean|inspect|record|then|first|next|finally|before|after)\b",
        re.I,
    ),
    "troubleshooting": re.compile(
        r"\b(if|when|cause[sd]?|fault|fail\w*|symptom|noise|vibrat\w*|leak\w*|overheat\w*|"
        r"trip\w*|alarm|excessive|worn|indicat\w*|problem)\b",
        re.I,
    ),
    "definition": re.compile(
        r"\b(is|are|means|refers to|defined|called|known as|consists of|rated)\b", re.I
    ),
}

_TEMPLATES = {
    "direct_question": [
        "What does the manual say about {kp}?",
        "What is the requirement for {kp}?",
        "Can you tell me about {kp}?",
    ],
    "troubleshooting": [
        "What should I check if there is a problem with {kp}?",
        "We are seeing an issue with {kp}. What could cause it and what should we do?",
        "How do I troubleshoot {kp}?",
    ],
    "procedure": [
        "What are the steps for {kp}?",
        "How do I carry out {kp}?",
        "Walk me through {kp}.",
    ],
    "definition": [
        "What is meant by {kp}?",
        "What is {kp} and why does it matter?",
        "Explain {kp}.",
    ],
}

_PERSONA_PREFIX = {
    "apprentice_technician": [
        "I'm new on the maintenance team. ",
        "Sorry if this is basic, but ",
        "",
    ],
    "senior_engineer": ["Quick check: ", "", "For the record, "],
    "safety_officer": ["From a safety point of view, ", "Before we sign off the permit, ", ""],
    "plant_manager": ["For planning the shutdown, ", "To estimate downtime, ", ""],
}

_DIFFICULTY_SENTENCES = {"easy": 1, "medium": 2, "hard": 3}
_DIFFICULTY_SUFFIX = {
    "easy": "",
    "medium": " Please be specific.",
    "hard": " Explain the reasoning and what condition decides it.",
}


def _rng(*parts: object) -> np.random.Generator:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def _usable(sentence: str) -> bool:
    n = len(sentence.split())
    return 5 <= n <= 60


class OfflineTeacher(Teacher):
    model = "offline"
    is_paid = False

    def __init__(self, chunks: Sequence[Chunk], seed: int = 42, error_rate: float = 0.05) -> None:
        self.seed = seed
        self.error_rate = error_rate
        self._chunks = list(chunks)
        df: Counter[str] = Counter()
        for chunk in self._chunks:
            df.update(set(content_words(chunk.text)))
        n = max(len(self._chunks), 1)
        self._idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}

    # ------------------------------------------------------------------ drafting
    def draft(self, request: DraftRequest) -> Draft | None:
        cell, chunk = request.cell, request.chunk
        rng = _rng(self.seed, chunk.id, cell.difficulty, cell.format, cell.persona, cell.topic)
        sents = [s for s in sentences(chunk.text) if _usable(s)]
        if not sents:
            return None
        n = min(_DIFFICULTY_SENTENCES.get(cell.difficulty, 1), len(sents))
        cue = _FORMAT_CUES.get(cell.format)
        scores = np.array(
            [
                (len(cue.findall(s)) if cue else 0) + rng.random()
                for s in sents[: len(sents) - n + 1]
            ]
        )
        top = np.argsort(-scores)[: min(3, len(scores))]
        start = int(rng.choice(top))
        selected = sents[start : start + n]
        keyphrase = self._keyphrase(" ".join(selected), rng)
        if not keyphrase:
            return None

        templates = _TEMPLATES.get(cell.format, _TEMPLATES["direct_question"])
        question = templates[int(rng.integers(len(templates)))].format(kp=keyphrase)
        prefixes = _PERSONA_PREFIX.get(cell.persona, [""])
        prefix = prefixes[int(rng.integers(len(prefixes)))]
        if prefix and not prefix.endswith(". "):
            question = question[0].lower() + question[1:]
        instruction = prefix + question + _DIFFICULTY_SUFFIX.get(cell.difficulty, "")

        response = " ".join(selected)
        if rng.random() < self.error_rate and len(self._chunks) > 1:
            # Simulated teacher mistake: confidently answer from an unrelated passage.
            others = [c for c in self._chunks if c.doc_id != chunk.doc_id] or [
                c for c in self._chunks if c.id != chunk.id
            ]
            wrong = others[int(rng.integers(len(others)))]
            wrong_sents = [s for s in sentences(wrong.text) if _usable(s)] or [wrong.text]
            response = " ".join(wrong_sents[:n])

        heading = chunk.heading or chunk.doc_id
        lead = " ".join(selected[0].split()[:14])
        rationale = f"Stated in '{heading}' ({chunk.id}), which says: \"{lead} ...\""
        return Draft(
            instruction=instruction,
            response=response,
            rationale=rationale,
            source_chunk_ids=[chunk.id],
        )

    def _keyphrase(self, text: str, rng: np.random.Generator) -> str:
        words = content_words(text)
        if not words:
            return ""
        counts = Counter(words)
        scored = sorted(
            counts, key=lambda w: -(counts[w] * self._idf.get(w, 1.0) + 0.01 * rng.random())
        )
        k = 2 if len(scored) < 6 else 3
        chosen = set(scored[:k])
        ordered: list[str] = []
        for w in words:  # keep the passage's word order so the phrase reads naturally
            if w in chosen and w not in ordered:
                ordered.append(w)
        return " ".join(ordered)

    # ------------------------------------------------------------------ answering
    def answer(self, question: str, chunks: Sequence[Chunk], attempt: int = 0) -> str | None:
        q_words = set(content_words(question))
        rng = _rng(self.seed, "answer", question, attempt)
        best: tuple[float, list[str], int] | None = None
        for chunk in chunks:
            sents = [s for s in sentences(chunk.text) if _usable(s)]
            for i, sentence in enumerate(sents):
                overlap = len(q_words & set(content_words(sentence)))
                score = overlap / math.sqrt(len(sentence.split())) + 0.35 * rng.random()
                if best is None or score > best[0]:
                    best = (score, sents, i)
        if best is None:
            return None
        _, sents, i = best
        lowered = question.lower()
        n = 3 if any(k in lowered for k in ("steps", "walk me", "explain", "reasoning")) else 2
        return " ".join(sents[i : i + n])
