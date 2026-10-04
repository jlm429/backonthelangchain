"""Tech-support service for support workflows."""

from langchain_core.messages import HumanMessage, SystemMessage

from backonthelangchain.agents.prompts import TECH_SUPPORT_PROMPT
from backonthelangchain.agents.tools import check_system_status


class TechSupportService:
    """Generate a tech-support answer using support context and tools."""

    def __init__(self, chat_model) -> None:
        self.chat_model = chat_model

    def answer(
        self,
        user_query: str,
        *,
        system_context: str | None = None,
        rag_context: str | None = None,
    ) -> tuple[str, str]:
        """Return the answer and the tool result used to produce it."""
        tool_result = system_context or check_system_status.invoke({})
        context_parts = [f"System evidence:\n{tool_result}"]
        if rag_context:
            context_parts.append(f"Relevant Tier 1 FAQ context:\n{rag_context}")
        response = self.chat_model.invoke(
            [
                TECH_SUPPORT_PROMPT,
                SystemMessage(content="\n\n".join(context_parts)),
                HumanMessage(content=user_query),
            ]
        )
        return response.content, tool_result
