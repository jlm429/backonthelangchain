"""Jev decision service for the experimental support-router workflow."""

from __future__ import annotations

from typing import Any

from backonthelangchain.agents.schemas import JevSupportDecision

JEV_MODEL = "jev-1.13.0"

JEV_SUPPORT_QUESTIONS: dict[str, dict[str, Any]] = {
    "needs_human_escalation": {
        "type": "noul",
        "instructions": (
            "Does `support_query` require a human support agent instead of "
            "automated technical or billing support?"
        ),
        "criteria": {
            "true": (
                "The user explicitly asks for a human, reports repeated failed "
                "support attempts, threatens legal or regulatory action, reports "
                "account compromise or suspected fraud, or describes a high-impact "
                "issue that automated support should not resolve."
            ),
            "false": (
                "The query can be handled by normal automated technical support or "
                "billing support. Frustration or angry language alone is not enough."
            ),
        },
    },
    "support_route": {
        "type": "choice",
        "instructions": "Which support domain should handle `support_query`?",
        "criteria": {
            "tech_support": (
                "Login issues, bugs, errors, setup, configuration, performance, "
                "account access, or other product behavior."
            ),
            "billing": (
                "Invoices, refunds, charges, subscriptions, payment methods, or "
                "plan changes."
            ),
        },
    },
}


class JevSupportRouterError(RuntimeError):
    """Raised when Jev cannot provide a valid support-routing judgment."""


class JevSupportRouterService:
    """Evaluate escalation and routing in one System One request."""

    def __init__(self, *, client: Any | None = None) -> None:
        self._client = client

    def evaluate(self, user_query: str) -> JevSupportDecision:
        """Return normalized judgments without exposing TypeSafe SDK objects."""
        try:
            if self._client is not None:
                response = self._request(self._client, user_query)
            else:
                from typesafe_sdk import TypeSafeClient

                with TypeSafeClient() as client:
                    response = self._request(client, user_query)

            route_answer = response.answers["support_route"]
            escalation_answer = response.answers["needs_human_escalation"]

            return JevSupportDecision(
                support_route=route_answer.choice,
                support_route_confidence=route_answer.confidence,
                support_route_probabilities=dict(route_answer.probabilities),
                needs_human_escalation=escalation_answer.noul,
                model=response.model,
            )
        except Exception as exc:
            raise JevSupportRouterError(
                "Jev could not provide a valid support-routing judgment."
            ) from exc

    def _request(self, client: Any, user_query: str) -> Any:
        return client.system_one(
            model=JEV_MODEL,
            state={"support_query": user_query},
            questions=JEV_SUPPORT_QUESTIONS,
        )
