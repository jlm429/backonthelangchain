"""Model factory helpers.

Centralizing model creation keeps notebooks cleaner and makes it easier to
swap model names, providers, temperatures, or structured-output wrappers.
"""

from langchain_openai import ChatOpenAI

from backonthelangchain.agents.schemas import BillingResponse, RouteDecision


def get_chat_model(model: str = "gpt-5.4-mini", temperature: float = 0.1) -> ChatOpenAI:
    """Create a ChatOpenAI model instance."""
    return ChatOpenAI(model=model, temperature=temperature)


def get_router_model(model: str = "gpt-5.4-mini"):
    """Create a low-temperature model that returns RouteDecision objects."""
    return get_chat_model(model=model, temperature=0.1).with_structured_output(
        RouteDecision
    )


def get_billing_model(model: str = "gpt-5.4-mini"):
    """Create a low-temperature model that returns BillingResponse objects."""
    return get_chat_model(model=model, temperature=0.1).with_structured_output(
        BillingResponse
    )
