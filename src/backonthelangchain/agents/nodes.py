"""Node functions for the support-router graph."""

from langchain_core.messages import HumanMessage, SystemMessage

from backonthelangchain.agents.prompts import (
    BILLING_PROMPT,
    ROUTER_PROMPT,
    TECH_SUPPORT_PROMPT,
)
from backonthelangchain.agents.schemas import (
    RouteNodeName,
    SupportRouterState,
)
from backonthelangchain.agents.tools import (
    check_system_status,
    get_subscription_status,
)


def make_router_node(router_model):
    """Create a router node bound to a structured-output model."""

    def router_node(state: SupportRouterState) -> SupportRouterState:
        decision = router_model.invoke(
            [
                ROUTER_PROMPT,
                HumanMessage(content=state["user_query"]),
            ]
        )
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


def make_tech_support_node(chat_model):
    """Create a tech-support node bound to a chat model."""

    def tech_support_answer(state: SupportRouterState) -> SupportRouterState:
        tool_result = check_system_status.invoke({})
        response = chat_model.invoke(
            [
                TECH_SUPPORT_PROMPT,
                SystemMessage(content=f"Tool result:\n{tool_result}"),
                HumanMessage(content=state["user_query"]),
            ]
        )
        return {
            "tool_result": tool_result,
            "answer": response.content,
        }

    return tech_support_answer


def make_billing_node(billing_model):
    """Create a billing node bound to a structured-output model."""

    def billing_answer(state: SupportRouterState) -> SupportRouterState:
        tool_result = get_subscription_status.invoke({})
        response = billing_model.invoke(
            [
                BILLING_PROMPT,
                SystemMessage(content=f"Tool result:\n{tool_result}"),
                HumanMessage(content=state["user_query"]),
            ]
        )
        return {
            "tool_result": tool_result,
            "answer": response.model_dump(),
        }

    return billing_answer
