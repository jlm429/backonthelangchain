"""Graph builders for agent workflows.

Graphs compose reusable services into runnable LangGraph workflows.
"""

from dataclasses import dataclass
from threading import Lock
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backonthelangchain.agents.models import (
    get_billing_model,
    get_chat_model,
    get_router_model,
)
from backonthelangchain.agents.nodes import (
    blocked_response_node,
    human_escalation_node,
    make_billing_node,
    make_contextual_tech_support_node,
    make_faq_retrieval_node,
    make_jev_router_node,
    make_router_node,
    make_safety_check_node,
    make_tech_support_node,
    make_tech_support_rag_node,
    pick_jev_route,
    pick_route,
    safety_gate,
    simulated_status_context_node,
)
from backonthelangchain.agents.schemas import (
    SupportRouterInput,
    SupportRouterOutput,
    SupportRouterState,
    UnifiedSupportInput,
    UnifiedSupportOutput,
)
from backonthelangchain.agents.services import (
    BillingService,
    JevSupportRouterService,
    OpenAIModerationSafetyService,
    RouterService,
    TechSupportRAGService,
    TechSupportService,
)


@dataclass(frozen=True)
class UnifiedSupportOptions:
    """Backend-authoritative optional stages for the unified graph."""

    faq_retrieval: bool = False


def _build_default_rag_pipeline():
    from backonthelangchain.rag.pipelines import TechSupportRAGPipeline
    from backonthelangchain.rag.rerankers import NoOpReranker

    return TechSupportRAGPipeline(
        reranker=NoOpReranker(),
        retrieve_top_k=5,
        rerank_top_k=3,
    )


class _LazyDefaultRAGPipeline:
    """Create provider-backed retrieval only after its graph node is reached."""

    def __init__(self) -> None:
        self._pipeline = None
        self._lock = Lock()

    def run(self, query: str):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    self._pipeline = _build_default_rag_pipeline()
        return self._pipeline.run(query)


_DEFAULT_RAG_PIPELINE = _LazyDefaultRAGPipeline()


UNIFIED_SUPPORT_NODE_METADATA: dict[str, dict[str, Any]] = {
    "__start__": {
        "label": "User query",
        "description": "Validated query and simulated demo state enter the graph.",
        "kind": "boundary",
        "required": True,
        "stage": 0,
    },
    "safety_check": {
        "label": "OpenAI Moderation",
        "description": "Authoritative mandatory safety gate.",
        "kind": "safety",
        "required": True,
        "stage": 1,
    },
    "blocked_response": {
        "label": "Blocked response",
        "description": "Safe deterministic response for moderated requests.",
        "kind": "response",
        "required": True,
        "stage": 2,
    },
    "simulated_status_context": {
        "label": "Simulated Status",
        "description": "Compares the report with configured demo service state.",
        "kind": "context",
        "required": True,
        "stage": 2,
    },
    "jev_router": {
        "label": "Jev support routing",
        "description": "Evaluates escalation and route with an OpenAI fallback.",
        "kind": "routing",
        "required": True,
        "stage": 3,
    },
    "faq_retrieval": {
        "label": "Tier 1 FAQ retrieval",
        "description": "Optional retrieval over fictional organization knowledge.",
        "kind": "retrieval",
        "required": False,
        "stage": 4,
    },
    "human_escalation": {
        "label": "Human escalation",
        "description": "Returns the deterministic handoff response.",
        "kind": "escalation",
        "required": True,
        "stage": 4,
    },
    "tech_support_answer": {
        "label": "Technical response",
        "description": "Generates guidance from supplied status and RAG context.",
        "kind": "response",
        "required": True,
        "stage": 5,
    },
    "billing_answer": {
        "label": "Billing response",
        "description": "Returns the existing structured billing response.",
        "kind": "response",
        "required": True,
        "stage": 4,
    },
    "__end__": {
        "label": "Final result",
        "description": "Returns normalized application-level output.",
        "kind": "boundary",
        "required": True,
        "stage": 6,
    },
}


def build_support_router_graph(
    *,
    model: str = "gpt-5.4-mini",
    checkpointer=None,
):
    """Build the basic support-router graph without safety or RAG."""

    router_model = get_router_model(model=model)
    billing_model = get_billing_model(model=model)
    answer_model = get_chat_model(model=model, temperature=0.7)

    router_service = RouterService(router_model)
    tech_support_service = TechSupportService(answer_model)
    billing_service = BillingService(billing_model)

    builder = StateGraph(
        SupportRouterState,
        input=SupportRouterInput,
        output=SupportRouterOutput,
    )

    builder.add_node("router", make_router_node(router_service))
    builder.add_node(
        "tech_support_answer", make_tech_support_node(tech_support_service)
    )
    builder.add_node("billing_answer", make_billing_node(billing_service))

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        pick_route,
        {
            "tech_support_answer": "tech_support_answer",
            "billing_answer": "billing_answer",
        },
    )
    builder.add_edge("tech_support_answer", END)
    builder.add_edge("billing_answer", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def build_safe_support_router_graph(
    *,
    model: str = "gpt-5.4-mini",
    moderation_model: str = "omni-moderation-latest",
    checkpointer=None,
):
    """Build a safety-gated support-router graph without RAG."""

    router_model = get_router_model(model=model)
    billing_model = get_billing_model(model=model)
    answer_model = get_chat_model(model=model, temperature=0.7)

    safety_service = OpenAIModerationSafetyService(model=moderation_model)
    router_service = RouterService(router_model)
    tech_support_service = TechSupportService(answer_model)
    billing_service = BillingService(billing_model)

    builder = StateGraph(
        SupportRouterState,
        input=SupportRouterInput,
        output=SupportRouterOutput,
    )

    builder.add_node("safety_check", make_safety_check_node(safety_service))
    builder.add_node("blocked_response", blocked_response_node)
    builder.add_node("router", make_router_node(router_service))
    builder.add_node(
        "tech_support_answer", make_tech_support_node(tech_support_service)
    )
    builder.add_node("billing_answer", make_billing_node(billing_service))

    builder.add_edge(START, "safety_check")
    builder.add_conditional_edges(
        "safety_check",
        safety_gate,
        {
            "router": "router",
            "blocked_response": "blocked_response",
        },
    )
    builder.add_conditional_edges(
        "router",
        pick_route,
        {
            "tech_support_answer": "tech_support_answer",
            "billing_answer": "billing_answer",
        },
    )
    builder.add_edge("blocked_response", END)
    builder.add_edge("tech_support_answer", END)
    builder.add_edge("billing_answer", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def build_jev_support_router_graph(
    *,
    model: str = "gpt-5.4-mini",
    moderation_model: str = "omni-moderation-latest",
    checkpointer=None,
    safety_service=None,
    jev_router_service=None,
    fallback_router_service=None,
    tech_support_service=None,
    billing_service=None,
):
    """Build the safety-gated experimental Jev support-router graph."""

    safety_service = safety_service or OpenAIModerationSafetyService(
        model=moderation_model
    )
    jev_router_service = jev_router_service or JevSupportRouterService()
    fallback_router_service = fallback_router_service or RouterService(
        get_router_model(model=model)
    )
    tech_support_service = tech_support_service or TechSupportService(
        get_chat_model(model=model, temperature=0.7)
    )
    billing_service = billing_service or BillingService(get_billing_model(model=model))

    builder = StateGraph(
        SupportRouterState,
        input_schema=SupportRouterInput,
        output_schema=SupportRouterOutput,
    )

    builder.add_node("safety_check", make_safety_check_node(safety_service))
    builder.add_node("blocked_response", blocked_response_node)
    builder.add_node(
        "jev_router",
        make_jev_router_node(
            jev_router_service,
            fallback_router_service,
        ),
    )
    builder.add_node("human_escalation", human_escalation_node)
    builder.add_node(
        "tech_support_answer", make_tech_support_node(tech_support_service)
    )
    builder.add_node("billing_answer", make_billing_node(billing_service))

    builder.add_edge(START, "safety_check")
    builder.add_conditional_edges(
        "safety_check",
        safety_gate,
        {
            "router": "jev_router",
            "blocked_response": "blocked_response",
        },
    )
    builder.add_conditional_edges(
        "jev_router",
        pick_jev_route,
        {
            "human_escalation": "human_escalation",
            "tech_support_answer": "tech_support_answer",
            "billing_answer": "billing_answer",
        },
    )
    builder.add_edge("blocked_response", END)
    builder.add_edge("human_escalation", END)
    builder.add_edge("tech_support_answer", END)
    builder.add_edge("billing_answer", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def build_safe_rag_support_router_graph(
    *,
    model: str = "gpt-5.4-mini",
    moderation_model: str = "omni-moderation-latest",
    reranker_model: str = "rerank-2.5",
    checkpointer=None,
):
    """Build a safety-gated support-router graph with RAG for tech support.

    Workflow:

        safety_check
            -> router
                -> tech_support_answer with FAQ RAG
                -> billing_answer
            -> blocked_response

    Tech support RAG flow:

        FAQ chunks
        -> OpenAI embeddings
        -> FAISS top-10 retrieval
        -> Voyage rerank top-5
        -> inject context into support answer prompt
    """

    from backonthelangchain.rag.pipelines import TechSupportRAGPipeline
    from backonthelangchain.rag.rerankers import VoyageReranker

    router_model = get_router_model(model=model)
    billing_model = get_billing_model(model=model)
    answer_model = get_chat_model(model=model, temperature=0.1)

    safety_service = OpenAIModerationSafetyService(model=moderation_model)
    router_service = RouterService(router_model)
    billing_service = BillingService(billing_model)

    rag_pipeline = TechSupportRAGPipeline(
        reranker=VoyageReranker(model=reranker_model),
        retrieve_top_k=10,
        rerank_top_k=3,
    )
    tech_support_rag_service = TechSupportRAGService(
        chat_model=answer_model,
        rag_pipeline=rag_pipeline,
    )

    builder = StateGraph(
        SupportRouterState,
        input=SupportRouterInput,
        output=SupportRouterOutput,
    )

    builder.add_node("safety_check", make_safety_check_node(safety_service))
    builder.add_node("blocked_response", blocked_response_node)
    builder.add_node("router", make_router_node(router_service))
    builder.add_node(
        "tech_support_answer",
        make_tech_support_rag_node(tech_support_rag_service),
    )
    builder.add_node("billing_answer", make_billing_node(billing_service))

    builder.add_edge(START, "safety_check")
    builder.add_conditional_edges(
        "safety_check",
        safety_gate,
        {
            "router": "router",
            "blocked_response": "blocked_response",
        },
    )
    builder.add_conditional_edges(
        "router",
        pick_route,
        {
            "tech_support_answer": "tech_support_answer",
            "billing_answer": "billing_answer",
        },
    )
    builder.add_edge("blocked_response", END)
    builder.add_edge("tech_support_answer", END)
    builder.add_edge("billing_answer", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def build_unified_support_graph(
    *,
    options: UnifiedSupportOptions | None = None,
    model: str = "gpt-5.4-mini",
    moderation_model: str = "omni-moderation-latest",
    checkpointer=None,
    safety_service=None,
    jev_router_service=None,
    fallback_router_service=None,
    tech_support_service=None,
    billing_service=None,
    rag_pipeline=None,
):
    """Build the primary support graph used by the interactive web application."""
    active_options = options or UnifiedSupportOptions()
    safety_service = safety_service or OpenAIModerationSafetyService(
        model=moderation_model
    )
    jev_router_service = jev_router_service or JevSupportRouterService()
    fallback_router_service = fallback_router_service or RouterService(
        get_router_model(model=model)
    )
    tech_support_service = tech_support_service or TechSupportService(
        get_chat_model(model=model, temperature=0.1)
    )
    billing_service = billing_service or BillingService(get_billing_model(model=model))

    if active_options.faq_retrieval and rag_pipeline is None:
        rag_pipeline = _DEFAULT_RAG_PIPELINE

    builder = StateGraph(
        SupportRouterState,
        input_schema=UnifiedSupportInput,
        output_schema=UnifiedSupportOutput,
    )
    builder.add_node("safety_check", make_safety_check_node(safety_service))
    builder.add_node("blocked_response", blocked_response_node)
    builder.add_node("simulated_status_context", simulated_status_context_node)
    builder.add_node(
        "jev_router",
        make_jev_router_node(jev_router_service, fallback_router_service),
    )
    builder.add_node("human_escalation", human_escalation_node)
    builder.add_node(
        "tech_support_answer",
        make_contextual_tech_support_node(tech_support_service),
    )
    builder.add_node("billing_answer", make_billing_node(billing_service))
    if active_options.faq_retrieval:
        builder.add_node("faq_retrieval", make_faq_retrieval_node(rag_pipeline))

    builder.add_edge(START, "safety_check")
    builder.add_conditional_edges(
        "safety_check",
        safety_gate,
        {
            "router": "simulated_status_context",
            "blocked_response": "blocked_response",
        },
    )
    builder.add_edge("simulated_status_context", "jev_router")
    builder.add_conditional_edges(
        "jev_router",
        pick_jev_route,
        {
            "human_escalation": "human_escalation",
            "tech_support_answer": (
                "faq_retrieval"
                if active_options.faq_retrieval
                else "tech_support_answer"
            ),
            "billing_answer": "billing_answer",
        },
    )
    if active_options.faq_retrieval:
        builder.add_edge("faq_retrieval", "tech_support_answer")
    builder.add_edge("blocked_response", END)
    builder.add_edge("human_escalation", END)
    builder.add_edge("tech_support_answer", END)
    builder.add_edge("billing_answer", END)

    return builder.compile(checkpointer=checkpointer)
