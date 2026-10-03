"""Reusable execution service for the safety-gated Jev support router."""

from __future__ import annotations

import asyncio
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel

from backonthelangchain.agents.graphs import build_jev_support_router_graph
from backonthelangchain.agents.nodes import JEV_HUMAN_ESCALATION_THRESHOLD


class InvokableGraph(Protocol):
    """The small graph surface needed by this service."""

    def invoke(self, input: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
        """Invoke a graph with a request-scoped thread."""


class SafetySummary(BaseModel):
    """Normalized safety decision safe to return to a caller."""

    allowed: bool
    model: str
    reason: str


class RoutingSummary(BaseModel):
    """Normalized routing decision safe to return to a caller."""

    destination: Literal[
        "blocked",
        "human_escalation",
        "tech_support",
        "billing",
    ]
    reason: str
    used_fallback: bool


class JevDecisionSummary(BaseModel):
    """Application-owned Jev signals, without provider response objects."""

    model: str | None
    route_confidence: float | None
    route_probabilities: dict[str, float]
    human_escalation_probability: float | None


class JevSupportResult(BaseModel):
    """Structured result shared by the CLI and web API."""

    outcome: Literal["completed", "blocked", "escalated"]
    answer: str | dict[str, Any]
    safety: SafetySummary
    routing: RoutingSummary
    jev: JevDecisionSummary | None


def run_jev_support_router(
    user_query: str,
    *,
    graph: InvokableGraph | None = None,
) -> JevSupportResult:
    """Run the existing graph and return a stable, provider-neutral result."""
    active_graph = graph or build_jev_support_router_graph()
    state = active_graph.invoke(
        {"user_query": user_query},
        config={"configurable": {"thread_id": f"jev-support-{uuid4().hex}"}},
    )

    is_safe = bool(state.get("is_safe", False))
    escalation = state.get("needs_human_escalation")
    if not is_safe:
        outcome = "blocked"
        destination = "blocked"
    elif (
        escalation is not None
        and escalation >= JEV_HUMAN_ESCALATION_THRESHOLD
    ):
        outcome = "escalated"
        destination = "human_escalation"
    else:
        outcome = "completed"
        destination = state["domain"]

    jev = None
    if state.get("jev_decision_available", False):
        jev = JevDecisionSummary(
            model=state.get("jev_model"),
            route_confidence=state.get("jev_route_confidence"),
            route_probabilities=state.get("jev_route_probabilities", {}),
            human_escalation_probability=state.get(
                "jev_human_escalation_probability"
            ),
        )

    return JevSupportResult(
        outcome=outcome,
        answer=state["answer"],
        safety=SafetySummary(
            allowed=is_safe,
            model=state.get("moderation_model", "unknown"),
            reason=state.get("safety_reason", "Safety check completed."),
        ),
        routing=RoutingSummary(
            destination=destination,
            reason=state.get("route_reason", "Request stopped by the safety gate."),
            used_fallback=bool(state.get("jev_used_fallback", False)),
        ),
        jev=jev,
    )


async def handle_jev_support_router(user_query: str) -> dict[str, Any]:
    """Run blocking provider calls away from FastAPI's event loop."""
    result = await asyncio.to_thread(run_jev_support_router, user_query)
    return result.model_dump(mode="json")
