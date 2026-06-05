"""Small demo tools for agent workflows.

Non-demo versions could call internal APIs, databases, ticketing
systems, status pages, or billing services.
"""

from langchain_core.tools import tool


@tool
def check_system_status() -> str:
    """Check whether the application is currently experiencing an outage."""
    return "System status: all services are operational."


@tool
def get_subscription_status() -> str:
    """Retrieve the customer's current subscription information."""
    return "Plan: Professional\nStatus: Active\nNext renewal: 2026-07-01"
