"""Teacher spend tracking with a hard cap (PRD section 8: budget cap on teacher spend).

Before each paid call the caller reserves the call's worst-case cost (estimated input
plus ``max_tokens`` of output). A call is refused when spent + outstanding
reservations + its worst case would cross the cap, so concurrent workers can never
push spend over the budget. Actual usage is settled into an append-only ledger, which
also lets a resumed run keep counting from where it stopped.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from expert_factory.artifacts import utcnow
from expert_factory.resources import load_config


class BudgetExceededError(RuntimeError):
    """Raised when a call could push teacher spend past the cap."""


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_write: float = 0.0
    cache_read: float = 0.0

    def cost(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        return (
            input_tokens * self.input
            + output_tokens * self.output
            + cache_write_tokens * self.cache_write
            + cache_read_tokens * self.cache_read
        ) / 1_000_000


def price_for(model: str) -> Price:
    table: dict[str, dict[str, float]] = load_config("budgets")["prices_per_mtok"]
    if model not in table:
        raise KeyError(
            f"no price for teacher model {model!r}; add it to configs/budgets.yaml so the "
            "budget cap can be enforced"
        )
    return Price(**table[model])


def estimate_tokens(text: str) -> int:
    """Deliberately pessimistic (about 3 characters per token) for reservations."""
    return len(text) // 3 + 64


@dataclass
class Reservation:
    worst_case_usd: float
    actual_usd: float = 0.0
    entry: dict[str, Any] = field(default_factory=dict)

    def settle(self, model: str, usage: Any, purpose: str) -> float:
        """Record actual usage from an Anthropic ``usage`` object (or a dict)."""

        def get(name: str) -> int:
            value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, 0)
            return int(value or 0)

        tokens = {
            "input_tokens": get("input_tokens"),
            "output_tokens": get("output_tokens"),
            "cache_write_tokens": get("cache_creation_input_tokens"),
            "cache_read_tokens": get("cache_read_input_tokens"),
        }
        self.actual_usd = price_for(model).cost(**tokens)
        self.entry = {"model": model, "purpose": purpose, **tokens, "cost_usd": self.actual_usd}
        return self.actual_usd


class BudgetMeter:
    def __init__(self, cap_usd: float, ledger_path: Path | None = None) -> None:
        self.cap_usd = cap_usd
        self.ledger_path = ledger_path
        self._lock = threading.Lock()
        self._reserved = 0.0
        self.spent_usd = 0.0
        self.calls = 0
        self.refused = 0
        self._by_purpose: dict[str, float] = defaultdict(float)
        self._tokens: dict[str, int] = defaultdict(int)
        if ledger_path and ledger_path.exists():
            for line in ledger_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._apply(json.loads(line))

    def _apply(self, entry: dict[str, Any]) -> None:
        self.spent_usd += float(entry.get("cost_usd", 0.0))
        self.calls += 1
        self._by_purpose[str(entry.get("purpose", "other"))] += float(entry.get("cost_usd", 0.0))
        for key in ("input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens"):
            self._tokens[key] += int(entry.get(key, 0))

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd - self._reserved)

    @contextmanager
    def reserve(self, worst_case_usd: float, purpose: str) -> Iterator[Reservation]:
        with self._lock:
            if self.spent_usd + self._reserved + worst_case_usd > self.cap_usd + 1e-9:
                self.refused += 1
                raise BudgetExceededError(
                    f"{purpose}: worst case ${worst_case_usd:.4f} would exceed the "
                    f"${self.cap_usd:.2f} cap (spent ${self.spent_usd:.4f}, "
                    f"reserved ${self._reserved:.4f})"
                )
            self._reserved += worst_case_usd
        reservation = Reservation(worst_case_usd)
        try:
            yield reservation
        finally:
            with self._lock:
                self._reserved -= worst_case_usd
                if reservation.entry:
                    entry = {"ts": utcnow(), **reservation.entry}
                    self._apply(entry)
                    if self.ledger_path:
                        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
                        with self.ledger_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(entry) + "\n")

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cap_usd": round(self.cap_usd, 4),
                "spent_usd": round(self.spent_usd, 4),
                "remaining_usd": round(max(0.0, self.cap_usd - self.spent_usd), 4),
                "calls": self.calls,
                "refused_calls": self.refused,
                "by_purpose_usd": {k: round(v, 4) for k, v in self._by_purpose.items()},
                "tokens": dict(self._tokens),
            }
