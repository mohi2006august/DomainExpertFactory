"""Anthropic (Claude) teacher with structured outputs, budget reservations and batching.

* Every request asks for JSON through ``output_config.format`` (json_schema), so
  drafts and answers parse without free-text scraping.
* Sync requests opt into server-side refusal fallbacks (``fallbacks: "default"``);
  a request the whole chain declines comes back with ``stop_reason == "refusal"``
  and is skipped rather than turned into training data.
* Batch mode uses the Message Batches API at half price (no fallbacks there).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from typing import Any

from expert_factory.budget import BudgetMeter, Price, estimate_tokens, price_for
from expert_factory.config import TeacherOptions
from expert_factory.log import get_logger
from expert_factory.resources import render_prompt
from expert_factory.schemas import Chunk, Draft, TaskSpec
from expert_factory.teachers.base import DraftRequest, Teacher, TeacherError

FALLBACK_BETA = "server-side-fallback-2026-07-01"
BATCH_DISCOUNT = 0.5
MAX_BATCH_REQUESTS = 10_000

DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string"},
        "response": {"type": "string"},
        "rationale": {"type": "string"},
        "source_chunk_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["instruction", "response", "rationale", "source_chunk_ids"],
    "additionalProperties": False,
}
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

log = get_logger("teacher.anthropic")


def make_client(max_retries: int) -> Any:
    """Anthropic client; TEACHER_API_KEY wins, else the SDK's own credential chain."""
    import anthropic

    api_key = os.environ.get("TEACHER_API_KEY", "").strip() or None
    if api_key:
        return anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
    return anthropic.Anthropic(max_retries=max_retries)


class AnthropicTeacher(Teacher):
    is_paid = True

    def __init__(
        self,
        model: str,
        task: TaskSpec,
        options: TeacherOptions,
        meter: BudgetMeter,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.options = options
        self.meter = meter
        self.price: Price = price_for(model)
        self.client = client if client is not None else make_client(options.max_retries)
        self.system = render_prompt("teacher_system.j2", task=task)

    # ------------------------------------------------------------- public API
    def draft(self, request: DraftRequest) -> Draft | None:
        data = self.call_json(self.draft_prompt(request), DRAFT_SCHEMA, "generate")
        return Draft.model_validate(data) if data else None

    def answer(self, question: str, chunks: Sequence[Chunk], attempt: int = 0) -> str | None:
        data = self.call_json(self.answer_prompt(question, chunks), ANSWER_SCHEMA, "answer")
        return str(data["answer"]).strip() if data else None

    def draft_prompt(self, request: DraftRequest) -> str:
        return render_prompt(
            "generate_example.j2",
            cell=request.cell,
            chunk=request.chunk,
            topic_label=request.topic_label,
            difficulty_description=request.difficulty_description,
            format_description=request.format_description,
            persona_description=request.persona_description,
            hard_example=request.hard_example,
        )

    def answer_prompt(self, question: str, chunks: Sequence[Chunk]) -> str:
        return render_prompt("answer_question.j2", question=question, chunks=list(chunks))

    # ------------------------------------------------------------- sync calls
    def call_json(
        self, user: str, schema: dict[str, Any], purpose: str, system: str | None = None
    ) -> dict[str, Any] | None:
        import anthropic

        system = system if system is not None else self.system
        worst = self.price.cost(
            input_tokens=estimate_tokens(system + user), output_tokens=self.options.max_tokens
        )
        params = self._params(system, user, schema)
        with self.meter.reserve(worst, purpose) as reservation:
            try:
                if self.options.fallbacks:
                    response = self.client.beta.messages.create(
                        **params, betas=[FALLBACK_BETA], fallbacks="default"
                    )
                else:
                    response = self.client.messages.create(**params)
            except anthropic.BadRequestError as exc:
                raise TeacherError(f"teacher rejected the request: {exc.message}") from exc
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
                raise TeacherError(
                    "teacher credentials missing or invalid: set TEACHER_API_KEY"
                ) from exc
            except anthropic.NotFoundError as exc:
                raise TeacherError(f"unknown teacher model {self.model!r}") from exc
            except anthropic.RateLimitError as exc:
                raise TeacherError("teacher rate limit persisted after retries") from exc
            except anthropic.APIStatusError as exc:
                raise TeacherError(f"teacher API error {exc.status_code}: {exc.message}") from exc
            except anthropic.APIConnectionError as exc:
                raise TeacherError(f"cannot reach the teacher API: {exc}") from exc
            reservation.settle(self._billing_model(response), response.usage, purpose)
        return self._parse(response, purpose)

    def _params(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.options.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "effort": self.options.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        }

    def _billing_model(self, response: Any) -> str:
        served = str(getattr(response, "model", "") or self.model)
        try:
            price_for(served)
        except KeyError:
            log.warning("unpriced_fallback_model", served=served, billed_as=self.model)
            return self.model
        return served

    @staticmethod
    def _parse(response: Any, purpose: str) -> dict[str, Any] | None:
        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            details = getattr(response, "stop_details", None)
            log.warning(
                "teacher_refusal",
                purpose=purpose,
                category=getattr(details, "category", None) if details else None,
            )
            return None
        if stop == "max_tokens":
            log.warning("teacher_truncated", purpose=purpose)
            return None
        text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), None)
        if text is None:
            log.warning("teacher_no_text", purpose=purpose, stop_reason=stop)
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log.warning("teacher_bad_json", purpose=purpose, preview=text[:120])
            return None
        return data if isinstance(data, dict) else None

    # ------------------------------------------------------------- batch calls
    def run_batch(
        self, prompts: Sequence[str], schema: dict[str, Any], purpose: str
    ) -> list[dict[str, Any] | None]:
        """Run prompts through the Batches API (50% price). Order of results = prompts.

        Prompts that do not fit in the remaining budget are not submitted and come
        back as None, so the caller can report the truncation.
        """
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        worst_each = [
            self.price.cost(
                input_tokens=estimate_tokens(self.system + p),
                output_tokens=self.options.max_tokens,
            )
            * BATCH_DISCOUNT
            for p in prompts
        ]
        affordable, total = 0, 0.0
        for cost in worst_each:
            if total + cost > self.meter.remaining_usd:
                break
            affordable, total = affordable + 1, total + cost
        results: list[dict[str, Any] | None] = [None] * len(prompts)
        if affordable < len(prompts):
            log.warning("batch_truncated_by_budget", requested=len(prompts), submitted=affordable)
        for start in range(0, affordable, MAX_BATCH_REQUESTS):
            end = min(start + MAX_BATCH_REQUESTS, affordable)
            with self.meter.reserve(sum(worst_each[start:end]), f"{purpose}-batch") as res:
                params = [self._params(self.system, prompts[i], schema) for i in range(start, end)]
                batch = self.client.messages.batches.create(
                    requests=[
                        Request(
                            custom_id=f"req-{start + j}",
                            params=MessageCreateParamsNonStreaming(**p),  # type: ignore[typeddict-item]
                        )
                        for j, p in enumerate(params)
                    ]
                )
                log.info("batch_submitted", batch_id=batch.id, size=end - start, purpose=purpose)
                self._wait(batch.id)
                usage: dict[str, int] = {}
                for item in self.client.messages.batches.results(batch.id):
                    index = int(item.custom_id.split("-", 1)[1])
                    if item.result.type != "succeeded":
                        log.warning("batch_item_failed", index=index, kind=item.result.type)
                        continue
                    message = item.result.message
                    for key in (
                        "input_tokens",
                        "output_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    ):
                        usage[key] = usage.get(key, 0) + int(getattr(message.usage, key, 0) or 0)
                    results[index] = self._parse(message, purpose)
                res.settle(self.model, usage, f"{purpose}-batch")
                res.actual_usd *= BATCH_DISCOUNT
                res.entry["cost_usd"] = res.actual_usd
        return results

    def _wait(self, batch_id: str) -> None:
        delay = 10.0
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                return
            log.info("batch_waiting", batch_id=batch_id, status=batch.processing_status)
            time.sleep(delay)
            delay = min(delay * 1.5, 120.0)
