"""Tech-support service for support workflows."""

from langchain_core.messages import HumanMessage, SystemMessage

from backonthelangchain.agents.prompts import TECH_SUPPORT_PROMPT
from backonthelangchain.agents.tools import check_system_status


class TechSupportService:
    """Generate a tech-support answer using support context and tools."""

    def __init__(self, chat_model) -> None:
        self.chat_model = chat_model

    def answer(self, user_query: str) -> tuple[str, str]:
        """Return the answer and the tool result used to produce it."""
        tool_result = check_system_status.invoke({})
        response = self.chat_model.invoke(
            [
                TECH_SUPPORT_PROMPT,
                SystemMessage(content=f"Tool result:\n{tool_result}"),
                HumanMessage(content=user_query),
            ]
        )
        return response.content, tool_result
