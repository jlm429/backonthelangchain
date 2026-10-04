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
from backonthelangchain.agents.nodes import JEV_HUMAN_ESCALATION_THRESHOLD
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


def _node_output(node_id: str, output: Any) -> dict[str, Any]:
    """Return only useful application-level fields for a streamed node event."""
    if not isinstance(output, dict):
        return {}
    allowed_fields = {
        "safety_check": {
            "is_safe",
            "moderation_flagged",
            "moderation_model",
            "safety_reason",
        },
        "simulated_status_context": {
            "status_evidence",
        },
        "jev_router": {
            "domain",
            "route_reason",
            "jev_model",
            "jev_route_confidence",
            "jev_route_probabilities",
            "jev_human_escalation_probability",
            "jev_decision_available",
            "jev_used_fallback",
        },
        "faq_retrieval": {"rag_sources"},
        "blocked_response": {"answer"},
        "human_escalation": {"answer"},
        "tech_support_answer": {"answer"},
        "billing_answer": {"answer"},
    }
    return {
        key: value
        for key, value in output.items()
        if key in allowed_fields.get(node_id, set())
    }


def _normalize_result(
    state: dict[str, Any],
    *,
    request: SupportRunRequest,
    path: list[str],
) -> dict[str, Any]:
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
    if state.get("jev_decision_available", False):
        jev = {
            "model": state.get("jev_model"),
            "route_confidence": state.get("jev_route_confidence"),
            "route_probabilities": state.get("jev_route_probabilities", {}),
            "human_escalation_probability": state.get(
                "jev_human_escalation_probability"
            ),
        }

    status_evidence = state.get("status_evidence")
    return {
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
        "retrieval": {
            "enabled": request.options.faq_retrieval,
            "executed": "faq_retrieval" in path,
            "sources": state.get("rag_sources", []),
        },
    }


class UnifiedSupportRunner:
    """Run the real graph and expose safe application events."""

    def __init__(self, graph_factory: GraphFactory | None = None) -> None:
        self.graph_factory = graph_factory or (
            lambda options: build_unified_support_graph(options=options)
        )
        self._graphs: dict[UnifiedSupportOptions, Any] = {}

    def _graph(self, options: UnifiedSupportOptions):
        if options not in self._graphs:
            self._graphs[options] = self.graph_factory(options)
        return self._graphs[options]

    async def events(
        self,
        request: SupportRunRequest,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield node lifecycle events from LangGraph's event stream."""
        run_id = uuid4().hex
        path: list[str] = []
        running_node: str | None = None
        yield {
            "type": "run_started",
            "run_id": run_id,
            "options": request.options.model_dump(mode="json"),
        }

        try:
            graph = self._graph(
                UnifiedSupportOptions(faq_retrieval=request.options.faq_retrieval)
            )
            graph_input = {
                "user_query": request.query,
                "simulated_status": request.simulated_status.model_dump(mode="json"),
            }
            config = {"configurable": {"thread_id": f"support-{run_id}"}}
            async for event in graph.astream_events(
                graph_input,
                config=config,
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
                    yield {
                        "type": "node_completed",
                        "node_id": name,
                        "output": _node_output(
                            name,
                            event.get("data", {}).get("output"),
                        ),
                    }
                    running_node = None
                elif event_name == "on_chain_end" and name == "LangGraph":
                    state = event.get("data", {}).get("output", {})
                    yield {
                        "type": "run_completed",
                        "run_id": run_id,
                        "result": _normalize_result(
                            state,
                            request=request,
                            path=path,
                        ),
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
