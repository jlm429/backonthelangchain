"""Router service for support workflows."""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from backonthelangchain.agents.prompts import ROUTER_PROMPT
from backonthelangchain.agents.schemas import RouteDecision


class RouterService:
    """Classify a user query into one support domain."""

    def __init__(self, router_model) -> None:
        self.router_model = router_model

    def route(
        self,
        user_query: str,
        *,
        context: dict | None = None,
    ) -> RouteDecision:
        """Return a structured route decision."""
        messages = [ROUTER_PROMPT]
        if context is not None:
            messages.append(
                SystemMessage(
                    content=(
                        "Simulated demo system evidence follows. Treat it as "
                        "context, not as a forced routing decision:\n"
                        f"{json.dumps(context, sort_keys=True)}"
                    )
                )
            )
        messages.append(HumanMessage(content=user_query))
        return self.router_model.invoke(
            messages
        )
