"""Pydantic and TypedDict schemas for agent workflows.

Keep schemas separate from graph logic so they can be reused by notebooks,
API endpoints, tests, and evaluation scripts.
"""

from typing import Any, Literal, TypedDict, Union

from pydantic import BaseModel, Field


RouteDomain = Literal["tech_support", "billing"]
SystemName = Literal["authentication", "billing", "checkout", "api"]
SystemStatusLevel = Literal["operational", "degraded", "outage"]
RouteNodeName = Literal["tech_support_answer", "billing_answer"]
JevRouteNodeName = Literal[
    "human_escalation",
    "tech_support_answer",
    "billing_answer",
]
SafetyGateNodeName = Literal["router", "blocked_response"]


class RouteDecision(BaseModel):
    """Structured output returned by the router model."""

    domain: RouteDomain = Field(
        description="Which support domain should handle the request."
    )
    reason: str = Field(
        description="Brief explanation for why this route was selected."
    )


class BillingResponse(BaseModel):
    """Structured response for the billing route."""

    summary: str = Field(description="Short summary of the user's billing issue.")
    next_step: str = Field(
        description="Recommended next step for the support team or user."
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="Estimated urgency of the billing issue."
    )


class SafetyResult(BaseModel):
    """Normalized result returned by the safety service."""

    is_safe: bool = Field(description="True when the request may continue.")
    flagged: bool = Field(description="Raw moderation flagged value.")
    model: str = Field(description="Moderation model used for the check.")
    categories: dict[str, bool] = Field(
        default_factory=dict,
        description="Moderation category booleans returned by the provider.",
    )
    category_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Moderation category scores returned by the provider.",
    )
    reason: str = Field(description="Short explanation of the safety decision.")


class JevSupportDecision(BaseModel):
    """Application-owned result for Jev support-routing judgments."""

    support_route: RouteDomain
    support_route_confidence: float = Field(ge=0.0, le=1.0)
    support_route_probabilities: dict[RouteDomain, float]
    needs_human_escalation: float = Field(ge=0.0, le=1.0)
    model: str


class SupportRouterState(TypedDict, total=False):
    """Internal graph state.

    This can include implementation details that callers do not need to see,
    such as safety decisions, routing decisions, route reasons, and tool outputs.
    """

    user_query: str
    simulated_status: dict[SystemName, SystemStatusLevel]

    # Safety / pre-router gate
    is_safe: bool
    moderation_flagged: bool
    moderation_model: str
    moderation_categories: dict[str, bool]
    moderation_category_scores: dict[str, float]
    safety_reason: str

    # Explicit demo evidence, never real monitoring data
    status_evidence: dict[str, Any]
    status_context: str
    reported_outage: bool

    # Routing
    domain: RouteDomain
    route_reason: str
    jev_model: str
    jev_route_confidence: float
    jev_route_probabilities: dict[RouteDomain, float]
    jev_human_escalation_probability: float
    jev_decision_available: bool
    needs_human_escalation: float
    jev_used_fallback: bool

    # Execution
    tool_result: str
    rag_context: str
    rag_sources: list[dict[str, Any]]
    answer: Union[str, dict]


class SupportRouterInput(TypedDict):
    """Public graph input schema."""

    user_query: str


class UnifiedSupportInput(TypedDict):
    """Public input for the unified support graph."""

    user_query: str
    simulated_status: dict[SystemName, SystemStatusLevel]


class SupportRouterOutput(TypedDict, total=False):
    """Public graph output schema shared by CLI and API adapters."""

    answer: Union[str, dict]
    is_safe: bool
    moderation_flagged: bool
    moderation_model: str
    safety_reason: str
    domain: RouteDomain
    route_reason: str
    jev_model: str
    jev_route_confidence: float
    jev_route_probabilities: dict[RouteDomain, float]
    jev_human_escalation_probability: float
    jev_decision_available: bool
    needs_human_escalation: float
    jev_used_fallback: bool


class UnifiedSupportOutput(SupportRouterOutput, total=False):
    """Public output including explicit demo context and optional retrieval."""

    status_evidence: dict[str, Any]
    reported_outage: bool
    rag_sources: list[dict[str, Any]]
