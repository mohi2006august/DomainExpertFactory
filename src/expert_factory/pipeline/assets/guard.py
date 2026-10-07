"""Overfitting guard (FR-04), shared by every training backend.

Primary signal is loss on the *real* held-out validation set. Training stops when:

* real val loss increased for ``patience`` consecutive evaluations, or
* the synthetic/real gap (real val loss - synthetic val loss) widened by more than
  ``max_gap_increase`` over its value at the first evaluation. The gap's baseline is
  subtracted because teacher-written and human-written text differ in difficulty
  from the start; what matters is the model drifting towards the teacher's style, or
* the canary hit rate exceeded its limit and ``stop_on_canary`` is set (otherwise a
  canary breach is recorded as an alert).

The caller restores the best checkpoint (lowest real val loss) after a stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from expert_factory.config import GuardConfig
from expert_factory.schemas import EvalPoint, GuardEvent


@dataclass
class GuardDecision:
    stop: bool
    is_best: bool
    reason: str | None = None


@dataclass
class OverfitGuard:
    config: GuardConfig
    best_loss: float = float("inf")
    best_step: int = -1
    best_epoch: float = 0.0
    events: list[GuardEvent] = field(default_factory=list)
    history: list[EvalPoint] = field(default_factory=list)
    stopped: bool = False
    stop_reason: str | None = None
    _consecutive_increases: int = 0
    _gap0: float | None = None

    def update(self, point: EvalPoint) -> GuardDecision:
        cfg = self.config
        previous = self.history[-1] if self.history else None
        self.history.append(point)
        if self._gap0 is None:
            self._gap0 = point.gap

        is_best = point.real_val_loss < self.best_loss - cfg.min_delta
        if is_best:
            self.best_loss, self.best_step, self.best_epoch = (
                point.real_val_loss,
                point.step,
                point.epoch,
            )
            self.events.append(
                GuardEvent(
                    step=point.step,
                    epoch=point.epoch,
                    kind="best",
                    reason="new best real val loss",
                    detail={"real_val_loss": round(point.real_val_loss, 5)},
                )
            )

        if previous is not None and point.real_val_loss > previous.real_val_loss + cfg.min_delta:
            self._consecutive_increases += 1
        else:
            self._consecutive_increases = 0

        reason: str | None = None
        widening = point.gap - self._gap0
        if self._consecutive_increases >= cfg.patience:
            reason = (
                f"real val loss rose for {self._consecutive_increases} consecutive evals "
                f"({previous.real_val_loss if previous else 0:.4f} -> {point.real_val_loss:.4f})"
            )
        elif widening > cfg.max_gap_increase:
            reason = (
                f"synthetic/real gap widened by {widening:.3f} (> {cfg.max_gap_increase}): "
                "model is fitting teacher style, not the domain"
            )

        if point.canary_hit_rate is not None and point.canary_hit_rate > cfg.max_canary_hit_rate:
            self.events.append(
                GuardEvent(
                    step=point.step,
                    epoch=point.epoch,
                    kind="alert",
                    reason=f"canary hit rate {point.canary_hit_rate:.0%} above "
                    f"{cfg.max_canary_hit_rate:.0%}: memorising single training examples",
                    detail={"canary_hit_rate": point.canary_hit_rate},
                )
            )
            if cfg.stop_on_canary and reason is None:
                reason = "canary memorisation limit exceeded"

        if reason is not None:
            self.stopped, self.stop_reason = True, reason
            self.events.append(
                GuardEvent(
                    step=point.step,
                    epoch=point.epoch,
                    kind="stop",
                    reason=reason,
                    detail={
                        "best_step": self.best_step,
                        "best_real_val_loss": round(self.best_loss, 5),
                        "gap_widening": round(widening, 5),
                    },
                )
            )
        return GuardDecision(stop=reason is not None, is_best=is_best, reason=reason)

    def recommendations(self) -> list[str]:
        """Human-readable next steps when the guard fires (SYSTEM_ARCHITECTURE.md section 9)."""
        if not self.stopped:
            return []
        recs = []
        reason = self.stop_reason or ""
        if "gap" in reason:
            recs.append("Increase data diversity: more personas/formats, or a lower max_per_chunk.")
            recs.append("Add more real seed examples so validation reflects real usage.")
        if "rose" in reason:
            recs.append("Reduce epochs or LoRA rank; the best checkpoint was restored.")
        if "canary" in reason:
            recs.append("Lower rank / learning rate and dedupe harder: the model memorises.")
        if self.best_step <= (self.history[0].step if self.history else 0):
            recs.append("Guard fired at the first evaluations: review data quality by hand.")
        return recs
