"""Graph builders for agent workflows."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backonthelangchain.agents.models import (
    get_billing_model,
    get_chat_model,
    get_router_model,
)
from backonthelangchain.agents.nodes import (
    make_billing_node,
    make_router_node,
    make_tech_support_node,
    pick_route,
)
from backonthelangchain.agents.schemas import (
    SupportRouterInput,
    SupportRouterOutput,
    SupportRouterState,
)


def build_support_router_graph(
    *,
    model: str = "gpt-5.4-mini",
    checkpointer=None,
):
    """Build a basic LangGraph router pattern.

    Pattern demonstrated:
    1. Router node asks the LLM for a structured route decision.
    2. Conditional edge sends state to the selected specialized node.
    3. Each route can use different tools, prompts, and output formats.

    This is intentionally small enough for notebooks, but modular enough to
    expand into more routes, better tools, tests, and evaluation datasets.
    """

    router_model = get_router_model(model=model)
    billing_model = get_billing_model(model=model)
    answer_model = get_chat_model(model=model, temperature=0.7)

    builder = StateGraph(
        SupportRouterState,
        input=SupportRouterInput,
        output=SupportRouterOutput,
    )

    builder.add_node("router", make_router_node(router_model))
    builder.add_node("tech_support_answer", make_tech_support_node(answer_model))
    builder.add_node("billing_answer", make_billing_node(billing_model))

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", pick_route)
    builder.add_edge("tech_support_answer", END)
    builder.add_edge("billing_answer", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
