"""Safety services for support workflows.

This module owns the provider-specific moderation API call. LangGraph nodes
should call this service rather than creating OpenAI clients directly.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from backonthelangchain.agents.schemas import SafetyResult


class OpenAIModerationSafetyService:
    """Check user input with OpenAI's moderation endpoint."""

    def __init__(
        self,
        *,
        client: OpenAI | None = None,
        model: str = "omni-moderation-latest",
    ) -> None:
        self.client = client or OpenAI()
        self.model = model

    def check(self, text: str) -> SafetyResult:
        """Return a normalized safety result for a user query."""
        result = self.client.moderations.create(
            model=self.model,
            input=text,
        )
        moderation = result.results[0]

        flagged = bool(moderation.flagged)
        categories = _model_to_dict(moderation.categories)
        category_scores = _model_to_dict(moderation.category_scores)

        return SafetyResult(
            is_safe=not flagged,
            flagged=flagged,
            model=self.model,
            categories={str(k): bool(v) for k, v in categories.items()},
            category_scores={str(k): float(v) for k, v in category_scores.items()},
            reason=(
                "OpenAI moderation did not flag the request."
                if not flagged
                else "OpenAI moderation flagged the request."
            ),
        )


def _model_to_dict(value: Any) -> dict[str, Any]:
    """Convert OpenAI SDK response objects to plain dictionaries."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return dict(value)
