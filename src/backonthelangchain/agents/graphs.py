"""Graph builders for agent workflows.

Graphs compose reusable services into runnable LangGraph workflows.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backonthelangchain.agents.models import (
    get_billing_model,
    get_chat_model,
    get_router_model,
)
from backonthelangchain.agents.nodes import (
    blocked_response_node,
    make_billing_node,
    make_router_node,
    make_safety_check_node,
    make_tech_support_node,
    pick_route,
    safety_gate,
)
from backonthelangchain.agents.schemas import (
    SupportRouterInput,
    SupportRouterOutput,
    SupportRouterState,
)
from backonthelangchain.agents.services import (
    BillingService,
    OpenAIModerationSafetyService,
    RouterService,
    TechSupportService,
)


def build_support_router_graph(
    *,
    model: str = "gpt-5.4-mini",
    checkpointer=None,
):
    """Build the basic support-router graph.

    Pattern demonstrated:
    1. Router service asks the LLM for a structured route decision.
    2. Conditional edge sends state to the selected specialized service.

    This graph does not run a pre-router safety check. It is useful as a
    baseline graph for tests, demos, and regression comparisons.
    """

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
    builder.add_node("tech_support_answer", make_tech_support_node(tech_support_service))
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
    """Build a safety-gated support-router graph.

    Pattern demonstrated:
    1. Safety service checks the user query before any route-specific work.
    2. Safety gate blocks flagged requests before the router runs.
    3. Router service asks the LLM for a structured route decision.
    4. Conditional edge sends state to the selected specialized service.

    Services are reusable components. Nodes adapt those services to LangGraph
    state. The graph is the orchestration recipe.
    """

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
    builder.add_node("tech_support_answer", make_tech_support_node(tech_support_service))
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