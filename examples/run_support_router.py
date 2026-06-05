"""Run the basic support-router graph from the command line.

Usage:
    python examples/run_support_router.py
    python examples/run_support_router.py "I was charged twice this month"
"""

import sys
from pprint import pprint

from backonthelangchain.agents import build_support_router_graph
from backonthelangchain.utils.env import load_project_env


EXAMPLE_QUERIES = [
    "I cannot log in after enabling MFA.",
    "I was charged twice this month.",
]


def main() -> None:
    load_project_env()

    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else EXAMPLE_QUERIES

    graph = build_support_router_graph()

    for i, query in enumerate(queries, start=1):
        print(f"\n--- Example {i} ---")
        print(f"User query: {query}")

        response = graph.invoke(
            {"user_query": query},
            config={"configurable": {"thread_id": f"support-router-demo-{i}"}},
        )

        print("\nAnswer:")
        pprint(response["answer"])


if __name__ == "__main__":
    main()