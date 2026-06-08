"""Pydantic and TypedDict schemas for agent workflows.

Keep schemas separate from graph logic so they can be reused by notebooks,
API endpoints, tests, and evaluation scripts.
"""

from typing import Literal, TypedDict, Union

from pydantic import BaseModel, Field


RouteDomain = Literal["tech_support", "billing"]
RouteNodeName = Literal["tech_support_answer", "billing_answer"]
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
    next_step: str = Field(description="Recommended next step for the support team or user.")
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


class SupportRouterState(TypedDict, total=False):
    """Internal graph state.

    This can include implementation details that callers do not need to see,
    such as safety decisions, routing decisions, route reasons, and tool outputs.
    """

    user_query: str

    # Safety / pre-router gate
    is_safe: bool
    moderation_flagged: bool
    moderation_model: str
    moderation_categories: dict[str, bool]
    moderation_category_scores: dict[str, float]
    safety_reason: str

    # Routing
    domain: RouteDomain
    route_reason: str

    # Execution
    tool_result: str
    answer: Union[str, dict]


class SupportRouterInput(TypedDict):
    """Public graph input schema."""

    user_query: str


class SupportRouterOutput(TypedDict):
    """Public graph output schema."""

    answer: Union[str, dict]
