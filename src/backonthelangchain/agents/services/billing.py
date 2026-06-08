"""Billing service for support workflows."""

from langchain_core.messages import HumanMessage, SystemMessage

from backonthelangchain.agents.prompts import BILLING_PROMPT
from backonthelangchain.agents.schemas import BillingResponse
from backonthelangchain.agents.tools import get_subscription_status


class BillingService:
    """Generate a structured billing response using billing context and tools."""

    def __init__(self, billing_model) -> None:
        self.billing_model = billing_model

    def answer(self, user_query: str) -> tuple[BillingResponse, str]:
        """Return the structured billing answer and the tool result used."""
        tool_result = get_subscription_status.invoke({})
        response = self.billing_model.invoke(
            [
                BILLING_PROMPT,
                SystemMessage(content=f"Tool result:\n{tool_result}"),
                HumanMessage(content=user_query),
            ]
        )
        return response, tool_result
