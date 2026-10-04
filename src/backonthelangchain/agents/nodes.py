"""Node functions for support-router graphs.

Nodes are intentionally thin LangGraph adapters. Reusable business or provider
logic lives in services/ so it can be tested and reused outside these graphs.
"""

import json
from pathlib import Path
from typing import Any

from backonthelangchain.agents.schemas import (
    JevRouteNodeName,
    RouteNodeName,
    SafetyGateNodeName,
    StageEvidence,
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


def _stage_evidence(
    stage_id: str,
    label: str,
    summary: str,
    *,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the only public evidence shape emitted by graph nodes."""
    return [
        StageEvidence(
            stage_id=stage_id,
            label=label,
            summary=summary,
            inputs=inputs,
            outputs=outputs,
        ).model_dump(mode="json")
    ]


def _escalation_context(state: SupportRouterState) -> dict[str, Any]:
    probability = state.get("jev_human_escalation_probability")
    return {
        "probability": probability,
        "threshold": JEV_HUMAN_ESCALATION_THRESHOLD,
        "threshold_met": (
            probability >= JEV_HUMAN_ESCALATION_THRESHOLD
            if probability is not None
            else None
        ),
    }


def _response_inputs(
    state: SupportRouterState,
    *,
    selected_route: str,
    retrieved_knowledge: str | None,
) -> dict[str, Any]:
    return {
        "user_query": state["user_query"],
        "selected_route": selected_route,
        "system_evidence": state.get("status_evidence"),
        "retrieved_knowledge_supplied": retrieved_knowledge,
        "escalation": _escalation_context(state),
    }


def _concise_snippet(text: str, *, limit: int = 280) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1].rstrip()}…"


def make_safety_check_node(
    safety_service: OpenAIModerationSafetyService,
):
    """Create a node that checks the user query before routing."""

    def safety_check_node(state: SupportRouterState) -> SupportRouterState:
        result = safety_service.check(state["user_query"])
        decision = "allowed" if result.is_safe else "blocked"
        return {
            "is_safe": result.is_safe,
            "moderation_flagged": result.flagged,
            "moderation_model": result.model,
            "moderation_categories": result.categories,
            "moderation_category_scores": result.category_scores,
            "safety_reason": result.reason,
            "jev_decision_available": False,
            "stage_evidence": _stage_evidence(
                "safety_check",
                "OpenAI Moderation",
                f"Moderation {decision} the request.",
                inputs={"user_query": state["user_query"]},
                outputs={
                    "decision": decision,
                    "allowed": result.is_safe,
                    "flagged": result.flagged,
                    "model": result.model,
                    "result": result.reason,
                },
            ),
        }

    return safety_check_node


def safety_gate(state: SupportRouterState) -> SafetyGateNodeName:
    """Route safe requests onward and block flagged requests."""

    if state.get("is_safe", False):
        return "router"

    return "blocked_response"


def blocked_response_node(state: SupportRouterState) -> SupportRouterState:
    """Return a safe response when moderation blocks the request."""
    answer = (
        "I cannot assist with that request. "
        "Please rephrase your question in a safe and appropriate way."
    )
    return {
        "answer": answer,
        "response_context": {
            "user_query": state["user_query"],
            "moderation_decision": "blocked",
        },
        "stage_evidence": _stage_evidence(
            "blocked_response",
            "Blocked response",
            "A deterministic safe response ended the workflow.",
            inputs={
                "user_query": state["user_query"],
                "moderation_decision": "blocked",
            },
            outputs={
                "response": answer,
                "production": "deterministic_moderation_response",
            },
        ),
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
        if context is None:
            decision = fallback_router_service.route(user_query)
        else:
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

        def with_evidence(
            route_state: SupportRouterState,
            *,
            decision_available: bool,
            model: str | None,
            classified_route: str | None,
            confidence: float | None,
            probabilities: dict[str, float],
            escalation_probability: float | None,
        ) -> SupportRouterState:
            escalation_met = (
                escalation_probability >= JEV_HUMAN_ESCALATION_THRESHOLD
                if escalation_probability is not None
                else None
            )
            confidence_met = (
                confidence >= JEV_ROUTE_CONFIDENCE_THRESHOLD
                if confidence is not None
                else None
            )
            selected_route = (
                "human_escalation"
                if escalation_met
                else route_state.get("domain", "unknown")
            )
            return {
                **route_state,
                "stage_evidence": _stage_evidence(
                    "jev_router",
                    "Jev support routing",
                    route_state["route_reason"],
                    inputs={
                        "user_query": user_query,
                        "system_evidence": status_evidence,
                    },
                    outputs={
                        "decision_available": decision_available,
                        "model": model or "unknown",
                        "classified_route": classified_route or "unknown",
                        "selected_route": selected_route,
                        "route_confidence": confidence,
                        "route_confidence_threshold": (
                            JEV_ROUTE_CONFIDENCE_THRESHOLD
                        ),
                        "route_confidence_threshold_met": confidence_met,
                        "route_probabilities": probabilities,
                        "human_escalation_probability": escalation_probability,
                        "human_escalation_threshold": (
                            JEV_HUMAN_ESCALATION_THRESHOLD
                        ),
                        "human_escalation_threshold_met": escalation_met,
                        "fallback_used": route_state.get(
                            "jev_used_fallback", False
                        ),
                        "route_reason": route_state["route_reason"],
                    },
                ),
            }

        try:
            if status_evidence is None:
                result = jev_router_service.evaluate(user_query)
            else:
                result = jev_router_service.evaluate(
                    user_query,
                    context=status_evidence,
                )
        except JevSupportRouterError:
            fallback_state: SupportRouterState = {
                **fallback(user_query, "Jev routing failed.", status_evidence),
                "jev_decision_available": False,
            }
            return with_evidence(
                fallback_state,
                decision_available=False,
                model=None,
                classified_route=None,
                confidence=None,
                probabilities={},
                escalation_probability=None,
            )

        jev_state: SupportRouterState = {
            "jev_model": result.model,
            "jev_route_confidence": result.support_route_confidence,
            "jev_route_probabilities": result.support_route_probabilities,
            "jev_human_escalation_probability": result.needs_human_escalation,
            "jev_decision_available": True,
            "jev_classified_route": result.support_route,
            "needs_human_escalation": result.needs_human_escalation,
            "jev_used_fallback": False,
        }

        if result.needs_human_escalation >= JEV_HUMAN_ESCALATION_THRESHOLD:
            route_state: SupportRouterState = {
                **jev_state,
                "route_reason": (
                    "Jev escalation probability met the configured threshold."
                ),
            }
            return with_evidence(
                route_state,
                decision_available=True,
                model=result.model,
                classified_route=result.support_route,
                confidence=result.support_route_confidence,
                probabilities=result.support_route_probabilities,
                escalation_probability=result.needs_human_escalation,
            )

        if result.support_route_confidence >= JEV_ROUTE_CONFIDENCE_THRESHOLD:
            route_state = {
                **jev_state,
                "domain": result.support_route,
                "route_reason": (
                    "Jev selected the support route at or above the confidence "
                    "threshold."
                ),
            }
            return with_evidence(
                route_state,
                decision_available=True,
                model=result.model,
                classified_route=result.support_route,
                confidence=result.support_route_confidence,
                probabilities=result.support_route_probabilities,
                escalation_probability=result.needs_human_escalation,
            )

        route_state = {
            **jev_state,
            **fallback(
                user_query,
                "Jev route confidence was below the threshold.",
                status_evidence,
            ),
        }
        return with_evidence(
            route_state,
            decision_available=True,
            model=result.model,
            classified_route=result.support_route,
            confidence=result.support_route_confidence,
            probabilities=result.support_route_probabilities,
            escalation_probability=result.needs_human_escalation,
        )

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
    response_context = _response_inputs(
        state,
        selected_route="human_escalation",
        retrieved_knowledge=None,
    )
    return {
        "answer": HUMAN_ESCALATION_ANSWER,
        "response_context": response_context,
        "stage_evidence": _stage_evidence(
            "human_escalation",
            "Human escalation",
            "The Jev escalation score met the threshold and triggered handoff.",
            inputs=response_context,
            outputs={
                "response": HUMAN_ESCALATION_ANSWER,
                "production": "deterministic_human_handoff",
                "human_escalation_triggered": True,
            },
        ),
    }


def pick_route(state: SupportRouterState) -> RouteNodeName:
    """Conditional edge function used by LangGraph."""

    if state["domain"] == "tech_support":
        return "tech_support_answer"

    return "billing_answer"


def make_tech_support_node(tech_support_service: TechSupportService):
    """Create a tech-support node bound to a basic tech-support service."""

    def tech_support_answer(state: SupportRouterState) -> SupportRouterState:
        answer, tool_result = tech_support_service.answer(state["user_query"])
        response_context = _response_inputs(
            state,
            selected_route="tech_support",
            retrieved_knowledge=None,
        )
        response_context["system_evidence"] = tool_result
        return {
            "tool_result": tool_result,
            "answer": answer,
            "response_context": response_context,
            "stage_evidence": _stage_evidence(
                "tech_support_answer",
                "Technical response",
                "The response model generated technical guidance.",
                inputs=response_context,
                outputs={
                    "generated_response": answer,
                    "production": "response_llm",
                },
            ),
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
    report_relations = ", ".join(
        f"{report['system']}={report['relation']}"
        for report in evidence["reports"]
    )
    summary = f"System evidence was {evidence['assessment'].replace('_', ' ')}."
    if report_relations:
        summary = f"{summary} Service relations: {report_relations}."
    return {
        "status_evidence": evidence,
        "status_context": json.dumps(evidence, sort_keys=True),
        "stage_evidence": _stage_evidence(
            "simulated_status_context",
            "Simulated status",
            summary,
            inputs={
                "user_query": state["user_query"],
                "configured_statuses": state["simulated_status"],
            },
            outputs={
                "source": evidence["source"],
                "is_real_monitoring": evidence["is_real_monitoring"],
                "relevant_services": evidence["relevant_services"],
                "configured_statuses": evidence["statuses"],
                "assessment": evidence["assessment"],
                "reports": evidence["reports"],
                "notice": evidence["notice"],
            },
        ),
    }


def make_faq_retrieval_node(rag_pipeline):
    """Create a visible FAQ retrieval stage from the existing RAG pipeline."""

    def faq_retrieval_node(state: SupportRouterState) -> SupportRouterState:
        result = rag_pipeline.run(state["user_query"])
        reranked_chunks = result.reranked_chunks
        documents = [
            {
                "rank": rank,
                "document_id": item.chunk_id,
                "chunk_id": item.chunk_id,
                "title": item.metadata.get("title") or item.chunk_id,
                "source": Path(item.source).name,
                "retrieval_score": item.retrieval_score,
                "rerank_score": item.rerank_score,
                "snippet": _concise_snippet(getattr(item, "text", "")),
            }
            for rank, item in enumerate(reranked_chunks, start=1)
        ]
        retrieval_query = getattr(result, "query", state["user_query"])
        retrieved_chunks = getattr(result, "retrieved_chunks", reranked_chunks)
        return {
            "rag_context": result.context,
            "rag_sources": documents,
            "rag_query": retrieval_query,
            "rag_result_count": len(retrieved_chunks),
            "stage_evidence": _stage_evidence(
                "faq_retrieval",
                "Tier 1 FAQ retrieval",
                f"Retrieval supplied {len(documents)} ranked document(s).",
                inputs={
                    "enabled": True,
                    "retrieval_query": retrieval_query,
                },
                outputs={
                    "retrieval_ran": True,
                    "result_count": len(retrieved_chunks),
                    "supplied_document_count": len(documents),
                    "documents": documents,
                    "context_supplied_to_response": result.context,
                },
            ),
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
        response_context = _response_inputs(
            state,
            selected_route="tech_support",
            retrieved_knowledge=state.get("rag_context"),
        )
        return {
            "tool_result": tool_result,
            "answer": answer,
            "response_context": response_context,
            "stage_evidence": _stage_evidence(
                "tech_support_answer",
                "Technical response",
                "The response model generated technical guidance from the supplied context.",
                inputs=response_context,
                outputs={
                    "generated_response": answer,
                    "production": "response_llm",
                },
            ),
        }

    return tech_support_answer


def make_tech_support_rag_node(
    tech_support_rag_service: TechSupportRAGService,
):
    """Create a tech-support node backed by FAQ RAG."""

    def tech_support_rag_answer(state: SupportRouterState) -> SupportRouterState:
        result = tech_support_rag_service.answer(state["user_query"])
        response_context = _response_inputs(
            state,
            selected_route="tech_support",
            retrieved_knowledge=result["rag_context"],
        )

        return {
            "answer": result["answer"],
            "rag_context": result["rag_context"],
            "rag_sources": result["rag_sources"],
            "response_context": response_context,
            "stage_evidence": _stage_evidence(
                "tech_support_answer",
                "Technical response",
                "The response model generated technical guidance from retrieved context.",
                inputs=response_context,
                outputs={
                    "generated_response": result["answer"],
                    "production": "response_llm",
                },
            ),
        }

    return tech_support_rag_answer


def make_billing_node(billing_service: BillingService):
    """Create a billing node bound to a billing service."""

    def billing_answer(state: SupportRouterState) -> SupportRouterState:
        answer, tool_result = billing_service.answer(state["user_query"])
        serialized_answer = answer.model_dump()
        response_context = _response_inputs(
            state,
            selected_route="billing",
            retrieved_knowledge=None,
        )
        return {
            "tool_result": tool_result,
            "answer": serialized_answer,
            "response_context": response_context,
            "stage_evidence": _stage_evidence(
                "billing_answer",
                "Billing response",
                "The billing model produced a structured response.",
                inputs=response_context,
                outputs={
                    "generated_response": serialized_answer,
                    "production": "structured_billing_model",
                },
            ),
        }

    return billing_answer
