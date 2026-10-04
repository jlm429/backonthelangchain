"""Node functions for support-router graphs.

Nodes are intentionally thin LangGraph adapters. Reusable business or provider
logic lives in services/ so it can be tested and reused outside these graphs.
"""

import json
from pathlib import Path

from backonthelangchain.agents.schemas import (
    JevRouteNodeName,
    RouteNodeName,
    SafetyGateNodeName,
    SupportRouterState,
)
from backonthelangchain.agents.services import (
    BillingService,
    JevSupportRouterError,
    JevSupportRouterService,
    OpenAIModerationSafetyService,
    RouterService,
    TechSupportRAGService,
    TechSupportService,
    build_simulated_status_evidence,
)

HUMAN_ESCALATION_ANSWER = (
    "This request needs review by a human support agent. "
    "Please contact support so an agent can assist you."
)
JEV_ROUTE_CONFIDENCE_THRESHOLD = 0.70
JEV_HUMAN_ESCALATION_THRESHOLD = 0.80


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
            "jev_decision_available": False,
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


def make_jev_router_node(
    jev_router_service: JevSupportRouterService,
    fallback_router_service: RouterService,
):
    """Create a Jev router that falls back when its route is uncertain."""

    def fallback(
        user_query: str,
        reason: str,
        context: dict | None,
    ) -> SupportRouterState:
        decision = fallback_router_service.route(user_query, context=context)
        return {
            "domain": decision.domain,
            "route_reason": f"{reason} Fallback router: {decision.reason}",
            "needs_human_escalation": 0.0,
            "jev_used_fallback": True,
        }

    def jev_router_node(state: SupportRouterState) -> SupportRouterState:
        user_query = state["user_query"]
        status_evidence = state.get("status_evidence")

        try:
            result = jev_router_service.evaluate(
                user_query,
                context=status_evidence,
            )
        except JevSupportRouterError:
            return {
                **fallback(user_query, "Jev routing failed.", status_evidence),
                "jev_decision_available": False,
            }

        jev_state: SupportRouterState = {
            "jev_model": result.model,
            "jev_route_confidence": result.support_route_confidence,
            "jev_route_probabilities": result.support_route_probabilities,
            "jev_human_escalation_probability": result.needs_human_escalation,
            "jev_decision_available": True,
            "needs_human_escalation": result.needs_human_escalation,
            "jev_used_fallback": False,
        }

        if result.needs_human_escalation >= JEV_HUMAN_ESCALATION_THRESHOLD:
            return {
                **jev_state,
                "route_reason": "Jev requested human escalation.",
            }

        if result.support_route_confidence >= JEV_ROUTE_CONFIDENCE_THRESHOLD:
            return {
                **jev_state,
                "domain": result.support_route,
                "route_reason": (
                    "Jev selected the support route at or above the confidence "
                    "threshold."
                ),
            }

        return {
            **jev_state,
            **fallback(
                user_query,
                "Jev route confidence was below the threshold.",
                status_evidence,
            ),
        }

    return jev_router_node


def pick_jev_route(state: SupportRouterState) -> JevRouteNodeName:
    """Route Jev results to escalation or the selected support domain."""

    if state.get("needs_human_escalation", 0.0) >= JEV_HUMAN_ESCALATION_THRESHOLD:
        return "human_escalation"
    if state.get("domain") == "tech_support":
        return "tech_support_answer"
    if state.get("domain") == "billing":
        return "billing_answer"
    raise ValueError("Jev routing did not produce a supported domain.")


def human_escalation_node(state: SupportRouterState) -> SupportRouterState:
    """Return the deterministic response for human escalation."""

    return {"answer": HUMAN_ESCALATION_ANSWER}


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


def simulated_status_context_node(
    state: SupportRouterState,
) -> SupportRouterState:
    """Turn caller-selected demo state into explicit routing evidence."""
    evidence = build_simulated_status_evidence(
        state["user_query"],
        state["simulated_status"],
    )
    return {
        "status_evidence": evidence,
        "status_context": json.dumps(evidence, sort_keys=True),
    }


def make_faq_retrieval_node(rag_pipeline):
    """Create a visible FAQ retrieval stage from the existing RAG pipeline."""

    def faq_retrieval_node(state: SupportRouterState) -> SupportRouterState:
        result = rag_pipeline.run(state["user_query"])
        return {
            "rag_context": result.context,
            "rag_sources": [
                {
                    "chunk_id": item.chunk_id,
                    "title": item.metadata.get("title"),
                    "source": Path(item.source).name,
                    "retrieval_score": item.retrieval_score,
                    "rerank_score": item.rerank_score,
                }
                for item in result.reranked_chunks
            ],
        }

    return faq_retrieval_node


def make_contextual_tech_support_node(
    tech_support_service: TechSupportService,
):
    """Create a technical answer node that consumes graph-produced context."""

    def tech_support_answer(state: SupportRouterState) -> SupportRouterState:
        answer, tool_result = tech_support_service.answer(
            state["user_query"],
            system_context=state.get("status_context"),
            rag_context=state.get("rag_context"),
        )
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
