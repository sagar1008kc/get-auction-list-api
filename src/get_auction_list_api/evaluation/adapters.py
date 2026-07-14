"""Lazy optional adapters; deterministic CI never imports vendor evaluation stacks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def ragas_available() -> bool:
    try:
        import ragas  # noqa: F401
    except ImportError:
        return False
    return True


def to_ragas_rows(
    *,
    questions: Sequence[str],
    answers: Sequence[str],
    contexts: Sequence[Sequence[str]],
    references: Sequence[str],
) -> list[dict[str, object]]:
    if not (len(questions) == len(answers) == len(contexts) == len(references)):
        raise ValueError("Ragas adapter columns must have equal lengths.")
    return [
        {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": list(context),
            "reference": reference,
        }
        for question, answer, context, reference in zip(
            questions, answers, contexts, references, strict=True
        )
    ]


def publish_langfuse_scores(client: Any, trace_id: str, scores: Mapping[str, float]) -> None:
    """Publish aggregate eval scores without prompts, answers, or source payloads."""

    for name, value in scores.items():
        client.create_score(
            trace_id=trace_id,
            name=f"eval.{name}",
            value=value,
            metadata={"evaluation": "deterministic"},
        )
