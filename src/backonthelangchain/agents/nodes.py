"""Node functions for support-router graphs.

Nodes are intentionally thin LangGraph adapters. Reusable business or provider
logic lives in services/ so it can be tested and reused outside these graphs.
"""

from backonthelangchain.agents.schemas import (
    RouteNodeName,
    SafetyGateNodeName,
    SupportRouterState,
)
from backonthelangchain.agents.services import (
    BillingService,
    OpenAIModerationSafetyService,
    RouterService,
    TechSupportRAGService,
    TechSupportService,
)


def make_safety_check_node(
    safety_service: OpenAIModerationSafetyService,
):
    """Create a node that checks the user query before routing."""

    def safety_check_node(state: SupportRouterState) -> SupportRouterState:
        result = safety_service.check(state["user_query"])
        return {
            "is_safe": result.is_safe,
            "moderation_flagged": result.flagged,
            "moderation_model": result.model,
            "moderation_categories": result.categories,
            "moderation_category_scores": result.category_scores,
            "safety_reason": result.reason,
        }

    return safety_check_node


def safety_gate(state: SupportRouterState) -> SafetyGateNodeName:
    """Route safe requests onward and block flagged requests."""

    if state.get("is_safe", False):
        return "router"

    return "blocked_response"


def blocked_response_node(state: SupportRouterState) -> SupportRouterState:
    """Return a safe response when moderation blocks the request."""

    return {
        "answer": (
            "I cannot assist with that request. "
            "Please rephrase your question in a safe and appropriate way."
        )
    }


def make_router_node(router_service: RouterService):
    """Create a router node bound to a router service."""

    def router_node(state: SupportRouterState) -> SupportRouterState:
        decision = router_service.route(state["user_query"])
        return {
            "domain": decision.domain,
            "route_reason": decision.reason,
        }

    return router_node


def pick_route(state: SupportRouterState) -> RouteNodeName:
    """Conditional edge function used by LangGraph."""

    if state["domain"] == "tech_support":
        return "tech_support_answer"

    return "billing_answer"


def make_tech_support_node(tech_support_service: TechSupportService):
    """Create a tech-support node bound to a basic tech-support service."""

    def tech_support_answer(state: SupportRouterState) -> SupportRouterState:
        answer, tool_result = tech_support_service.answer(state["user_query"])
        return {
            "tool_result": tool_result,
            "answer": answer,
        }

    return tech_support_answer


def make_tech_support_rag_node(
    tech_support_rag_service: TechSupportRAGService,
):
    """Create a tech-support node backed by FAQ RAG."""

    def tech_support_rag_answer(state: SupportRouterState) -> SupportRouterState:
        result = tech_support_rag_service.answer(state["user_query"])

        return {
            "answer": result["answer"],
            "rag_context": result["rag_context"],
            "rag_sources": result["rag_sources"],
        }

    return tech_support_rag_answer


def make_billing_node(billing_service: BillingService):
    """Create a billing node bound to a billing service."""

    def billing_answer(state: SupportRouterState) -> SupportRouterState:
        answer, tool_result = billing_service.answer(state["user_query"])
        return {
            "tool_result": tool_result,
            "answer": answer.model_dump(),
        }

    return billing_answer