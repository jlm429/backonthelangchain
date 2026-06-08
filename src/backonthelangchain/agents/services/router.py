"""Router service for support workflows."""

from langchain_core.messages import HumanMessage

from backonthelangchain.agents.prompts import ROUTER_PROMPT
from backonthelangchain.agents.schemas import RouteDecision


class RouterService:
    """Classify a user query into one support domain."""

    def __init__(self, router_model) -> None:
        self.router_model = router_model

    def route(self, user_query: str) -> RouteDecision:
        """Return a structured route decision."""
        return self.router_model.invoke(
            [
                ROUTER_PROMPT,
                HumanMessage(content=user_query),
            ]
        )
