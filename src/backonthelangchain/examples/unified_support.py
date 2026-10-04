"""Description and live execution adapters for the unified support graph."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4

from backonthelangchain.agents.graphs import (
    UNIFIED_SUPPORT_NODE_METADATA,
    UnifiedSupportOptions,
    build_unified_support_graph,
)
from backonthelangchain.agents.nodes import (
    JEV_HUMAN_ESCALATION_THRESHOLD,
    JEV_ROUTE_CONFIDENCE_THRESHOLD,
)
from backonthelangchain.agents.schemas import (
    ExecutionSummary,
    ExecutionSummaryFact,
    StageEvidence,
)
from backonthelangchain.api.schemas import SupportGraphOptions, SupportRunRequest

GraphFactory = Callable[[UnifiedSupportOptions], Any]

BRANCH_LABELS = {
    ("safety_check", "blocked_response"): "flagged",
    ("safety_check", "simulated_status_context"): "allowed",
    ("jev_router", "human_escalation"): "escalate",
    ("jev_router", "billing_answer"): "billing",
    ("jev_router", "faq_retrieval"): "technical",
    ("jev_router", "tech_support_answer"): "technical",
}


class _DescriptionOnlyService:
    """Satisfy graph construction without creating or calling providers."""

    def __getattr__(self, _name: str):
        def unavailable(*_args, **_kwargs):
            raise RuntimeError("Description-only graph nodes cannot execute.")

        return unavailable


def _description_graph(options: UnifiedSupportOptions):
    placeholder = _DescriptionOnlyService()
    return build_unified_support_graph(
        options=options,
        safety_service=placeholder,
        jev_router_service=placeholder,
        fallback_router_service=placeholder,
        tech_support_service=placeholder,
        billing_service=placeholder,
        rag_pipeline=placeholder,
    )


def describe_unified_support_graph(
    options: SupportGraphOptions,
) -> dict[str, Any]:
    """Serialize nodes and edges from the executable LangGraph builder."""
    runtime_options = UnifiedSupportOptions(faq_retrieval=options.faq_retrieval)
    raw_graph = _description_graph(runtime_options).get_graph().to_json()
    executable_node_ids = {node["id"] for node in raw_graph["nodes"]}

    nodes = []
    for node_id, metadata in UNIFIED_SUPPORT_NODE_METADATA.items():
        nodes.append(
            {
                "id": node_id,
                **metadata,
                "enabled": node_id in executable_node_ids,
            }
        )

    edges = [
        {
            "source": edge["source"],
            "target": edge["target"],
            "conditional": bool(edge.get("conditional", False)),
            "branch": (
                BRANCH_LABELS[(edge["source"], edge["target"])]
                if edge.get("conditional", False)
                else None
            ),
        }
        for edge in raw_graph["edges"]
    ]
    return {
        "graph_id": "unified-support",
        "options": options.model_dump(mode="json"),
        "nodes": sorted(nodes, key=lambda node: (node["stage"], node["id"])),
        "edges": edges,
    }


def _node_evidence(node_id: str, output: Any) -> dict[str, Any] | None:
    """Validate and serialize only a node's application-owned evidence."""
    if not isinstance(output, dict):
        return None
    evidence_items = output.get("stage_evidence")
    if not isinstance(evidence_items, list):
        return None
    for item in reversed(evidence_items):
        try:
            evidence = StageEvidence.model_validate(item)
        except (TypeError, ValueError):
            continue
        if evidence.stage_id == node_id:
            return evidence.model_dump(mode="json")
    return None


def _format_probability(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2f}"


def _threshold_result(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "met" if value else "not met"


def _evidence_for(
    evidence: list[dict[str, Any]], stage_id: str
) -> dict[str, Any] | None:
    return next((item for item in evidence if item["stage_id"] == stage_id), None)


def _build_execution_summary(
    *,
    outcome: str,
    destination: str,
    evidence: list[dict[str, Any]],
    retrieval: dict[str, Any],
    jev: dict[str, Any] | None,
) -> dict[str, Any]:
    """Summarize allowlisted stage evidence without asking a model to explain."""
    safety = _evidence_for(evidence, "safety_check")
    status = _evidence_for(evidence, "simulated_status_context")
    response = next(
        (
            item
            for item in evidence
            if item["stage_id"]
            in {
                "blocked_response",
                "human_escalation",
                "tech_support_answer",
                "billing_answer",
            }
        ),
        None,
    )
    facts: list[ExecutionSummaryFact] = []

    if safety is not None:
        safety_outputs = safety["outputs"]
        facts.append(
            ExecutionSummaryFact(
                category="safety",
                text=(
                    f"OpenAI Moderation {safety_outputs['decision']} the request "
                    f"using {safety_outputs['model']}."
                ),
            )
        )

    if status is None:
        status_text = "System evidence was not evaluated because the stage was not reached."
    else:
        status_outputs = status["outputs"]
        services = status_outputs["relevant_services"]
        service_text = (
            ", ".join(
                f"{item['system']}={item['configured_status']}" for item in services
            )
            or "no relevant service"
        )
        status_text = (
            f"Simulated evidence assessment was {status_outputs['assessment']}; "
            f"relevant service state: {service_text}."
        )
        report_text = ", ".join(
            f"{report['system']}={report['relation']}"
            for report in status_outputs["reports"]
        )
        if report_text:
            status_text = f"{status_text} Service relations: {report_text}."
    facts.append(ExecutionSummaryFact(category="system_evidence", text=status_text))

    if jev is None:
        routing_text = "Jev routing was not reached."
    elif not jev["decision_available"]:
        routing_text = (
            f"Jev returned no usable decision; the fallback selected {destination}."
        )
    else:
        routing_text = (
            f"Jev classified {jev['classified_route']} at confidence "
            f"{_format_probability(jev['route_confidence'])}; the "
            f"{jev['route_confidence_threshold']:.2f} route threshold was "
            f"{_threshold_result(jev['route_confidence_threshold_met'])}. "
            f"Escalation probability was "
            f"{_format_probability(jev['human_escalation_probability'])}; the "
            f"{jev['human_escalation_threshold']:.2f} escalation threshold was "
            f"{_threshold_result(jev['human_escalation_threshold_met'])}."
        )
    facts.append(ExecutionSummaryFact(category="routing", text=routing_text))

    if retrieval["executed"]:
        titles = [item["title"] for item in retrieval["documents"]]
        retrieval_text = (
            f"Retrieval ran for {retrieval['query']!r}, returned "
            f"{retrieval['result_count']} result(s), and supplied "
            f"{', '.join(titles) if titles else 'no documents'} to response generation."
        )
    elif retrieval["enabled"]:
        retrieval_text = "Retrieval was enabled, but the technical branch was not taken."
    else:
        retrieval_text = "Retrieval was disabled, so no demo knowledge was supplied."
    facts.append(ExecutionSummaryFact(category="retrieval", text=retrieval_text))

    if response is None:
        response_text = "No response-producing stage completed."
    else:
        response_inputs = response["inputs"]
        response_text = (
            f"The {response['label']} stage received route "
            f"{response_inputs.get('selected_route', destination)}, system evidence "
            f"{'yes' if response_inputs.get('system_evidence') is not None else 'no'}, "
            f"and retrieved knowledge "
            f"{'yes' if response_inputs.get('retrieved_knowledge_supplied') else 'no'}."
        )
    facts.append(
        ExecutionSummaryFact(category="response_context", text=response_text)
    )
    facts.append(
        ExecutionSummaryFact(
            category="outcome",
            text=f"The workflow outcome was {outcome} at {destination}.",
        )
    )
    return ExecutionSummary(
        headline=f"Request {outcome} through {destination}.",
        facts=facts,
    ).model_dump(mode="json")


def _normalize_result(
    state: dict[str, Any],
    *,
    request: SupportRunRequest,
    path: list[str],
    evidence: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    is_safe = bool(state.get("is_safe", False))
    escalation = state.get("needs_human_escalation")
    if not is_safe:
        outcome = "blocked"
        destination = "blocked"
    elif escalation is not None and escalation >= JEV_HUMAN_ESCALATION_THRESHOLD:
        outcome = "escalated"
        destination = "human_escalation"
    else:
        outcome = "completed"
        destination = state.get("domain", "tech_support")

    jev = None
    if "jev_router" in path:
        route_confidence = state.get("jev_route_confidence")
        escalation_probability = state.get("jev_human_escalation_probability")
        jev = {
            "decision_available": bool(state.get("jev_decision_available", False)),
            "model": state.get("jev_model", "unknown"),
            "classified_route": state.get("jev_classified_route", "unknown"),
            "selected_route": destination,
            "route_confidence": route_confidence,
            "route_probabilities": state.get("jev_route_probabilities", {}),
            "route_confidence_threshold": JEV_ROUTE_CONFIDENCE_THRESHOLD,
            "route_confidence_threshold_met": (
                route_confidence >= JEV_ROUTE_CONFIDENCE_THRESHOLD
                if route_confidence is not None
                else None
            ),
            "human_escalation_probability": escalation_probability,
            "human_escalation_threshold": JEV_HUMAN_ESCALATION_THRESHOLD,
            "human_escalation_threshold_met": (
                escalation_probability >= JEV_HUMAN_ESCALATION_THRESHOLD
                if escalation_probability is not None
                else None
            ),
        }

    status_evidence = state.get("status_evidence")
    retrieval_documents = state.get("rag_sources", [])
    retrieval = {
        "enabled": request.options.faq_retrieval,
        "executed": "faq_retrieval" in path,
        "query": state.get("rag_query"),
        "result_count": state.get("rag_result_count", 0),
        "supplied_document_count": len(retrieval_documents),
        "documents": retrieval_documents,
        "sources": retrieval_documents,
        "context_supplied_to_response": state.get("rag_context"),
    }
    execution_summary = _build_execution_summary(
        outcome=outcome,
        destination=destination,
        evidence=evidence,
        retrieval=retrieval,
        jev=jev,
    )
    response_stage = next(
        (
            item
            for item in reversed(evidence)
            if item["stage_id"]
            in {
                "blocked_response",
                "human_escalation",
                "tech_support_answer",
                "billing_answer",
            }
        ),
        None,
    )
    production = (
        response_stage["outputs"].get("production", "unknown")
        if response_stage
        else "unknown"
    )
    provenance = {
        "user_query": request.query,
        "simulated_statuses": request.simulated_status.model_dump(mode="json"),
        "system_evidence_assessment": (
            status_evidence.get("assessment") if status_evidence else "not_evaluated"
        ),
        "jev_route": (
            jev["classified_route"] if jev is not None else "not_evaluated"
        ),
        "retrieved_documents": [item["title"] for item in retrieval_documents],
        "human_escalation_triggered": outcome == "escalated",
        "response_production": production,
    }
    result = {
        "outcome": outcome,
        "answer": state.get("answer", "No response was produced."),
        "execution_path": path,
        "safety": {
            "allowed": is_safe,
            "model": state.get("moderation_model", "unknown"),
            "reason": state.get("safety_reason", "Safety check completed."),
        },
        "system_status": {
            "evaluated": status_evidence is not None,
            "simulated": True,
            "statuses": request.simulated_status.model_dump(mode="json"),
            "evidence": status_evidence,
            "notice": "Simulated demo state, not real monitoring data.",
        },
        "routing": {
            "destination": destination,
            "reason": state.get(
                "route_reason",
                "Request stopped by the mandatory safety gate.",
            ),
            "used_fallback": bool(state.get("jev_used_fallback", False)),
        },
        "jev": jev,
        "retrieval": retrieval,
        "response_generation": {
            "inputs": state.get("response_context", {}),
            "generated_response": state.get("answer", "No response was produced."),
            "production": production,
        },
        "execution_summary": execution_summary,
        "provenance": provenance,
    }
    final_evidence = StageEvidence(
        stage_id="__end__",
        label="Final result",
        summary=execution_summary["headline"],
        inputs={"executed_path": path},
        outputs={
            "outcome": outcome,
            "answer": result["answer"],
            "execution_summary": execution_summary,
            "provenance": provenance,
        },
    ).model_dump(mode="json")
    result["stage_evidence"] = [*evidence, final_evidence]
    return result, final_evidence


class UnifiedSupportRunner:
    """Run the real graph and expose safe application events."""

    def __init__(self, graph_factory: GraphFactory | None = None) -> None:
        self.graph_factory = graph_factory or (
            lambda options: build_unified_support_graph(options=options)
        )

    async def events(
        self,
        request: SupportRunRequest,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield node lifecycle events from LangGraph's event stream."""
        run_id = uuid4().hex
        path: list[str] = []
        running_node: str | None = None
        start_evidence = StageEvidence(
            stage_id="__start__",
            label="User query",
            summary="Validated application input entered the graph.",
            inputs={
                "user_query": request.query,
                "simulated_statuses": request.simulated_status.model_dump(mode="json"),
                "faq_retrieval_enabled": request.options.faq_retrieval,
            },
            outputs={"accepted": True},
        ).model_dump(mode="json")
        completed_evidence = [start_evidence]
        yield {
            "type": "run_started",
            "run_id": run_id,
            "options": request.options.model_dump(mode="json"),
            "evidence": start_evidence,
        }

        try:
            graph = self.graph_factory(
                UnifiedSupportOptions(faq_retrieval=request.options.faq_retrieval)
            )
            graph_input = {
                "user_query": request.query,
                "simulated_status": request.simulated_status.model_dump(mode="json"),
            }
            async for event in graph.astream_events(
                graph_input,
                version="v2",
            ):
                event_name = event.get("event")
                name = event.get("name")
                tags = event.get("tags", [])
                is_node = (
                    isinstance(name, str)
                    and name in UNIFIED_SUPPORT_NODE_METADATA
                    and any(tag.startswith("graph:step:") for tag in tags)
                )
                if is_node and event_name == "on_chain_start":
                    running_node = name
                    path.append(name)
                    yield {"type": "node_started", "node_id": name}
                elif is_node and event_name == "on_chain_end":
                    evidence = _node_evidence(
                        name,
                        event.get("data", {}).get("output"),
                    )
                    if evidence is not None:
                        completed_evidence.append(evidence)
                    yield {
                        "type": "node_completed",
                        "node_id": name,
                        "evidence": evidence,
                        "output": evidence["outputs"] if evidence else {},
                    }
                    running_node = None
                elif event_name == "on_chain_end" and name == "LangGraph":
                    state = event.get("data", {}).get("output", {})
                    result, final_evidence = _normalize_result(
                        state,
                        request=request,
                        path=path,
                        evidence=completed_evidence,
                    )
                    yield {
                        "type": "run_completed",
                        "run_id": run_id,
                        "evidence": final_evidence,
                        "result": result,
                    }
        except Exception:
            if running_node is not None:
                yield {
                    "type": "node_failed",
                    "node_id": running_node,
                    "message": "This stage could not be completed.",
                }
            yield {
                "type": "run_failed",
                "run_id": run_id,
                "message": "The support workflow could not be completed.",
            }
