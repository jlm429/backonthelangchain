"""Run the safety-gated support-router graph from the command line.

This version runs an OpenAI moderation check before routing.

Usage:
    python examples/run_safe_support_router.py
    python examples/run_safe_support_router.py "I cannot log in after enabling MFA"
    python examples/run_safe_support_router.py "I was charged twice this month."
    python examples/run_safe_support_router.py "I hate your support team. They are worthless idiots. I was charged twice this month." -> SAFETY GATE BLOCKS
"""

import sys
from pprint import pprint

from backonthelangchain.agents import build_safe_support_router_graph
from backonthelangchain.utils.env import load_project_env


def main() -> None:
    load_project_env()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("User query> ").strip()

    graph = build_safe_support_router_graph()

    response = graph.invoke(
        {"user_query": query},
        config={
            "configurable": {
                "thread_id": "safe-support-router-demo",
            }
        },
    )

    print("\nAnswer:")
    pprint(response["answer"])

    if "safety" in response:
        print("\nSafety:")
        pprint(response["safety"])


if __name__ == "__main__":
    main()