"""Run the safety-gated support router with RAG for tech-support requests.

Usage:
    poetry run python examples/run_safe_rag_support_router.py
    poetry run python examples/run_safe_rag_support_router.py "my account is locked"
    poetry run python examples/run_safe_rag_support_router.py "I can’t get into the admin area. Can you give me access?"

Required environment variables:
    OPENAI_API_KEY
    VOYAGE_API_KEY
"""

import sys
from pprint import pprint

from backonthelangchain.agents.graphs import build_safe_rag_support_router_graph
from backonthelangchain.utils.env import load_project_env


def main() -> None:
    load_project_env()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("User query> ").strip()

    graph = build_safe_rag_support_router_graph()

    response = graph.invoke(
        {"user_query": query},
        config={"configurable": {"thread_id": "safe-rag-support-router-demo"}},
    )

    print("\nAnswer:")
    pprint(response["answer"])

    if "rag_context" in response:
        print("\nRAG context:")
        for i, chunk in enumerate(response["rag_context"], start=1):
            metadata = chunk.get("metadata", {})
            print(
                f"{i}. {metadata.get('faq_id')} - {metadata.get('faq_title')} "
                f"| rerank_score={chunk.get('rerank_score')}"
            )

    if "safety" in response:
        print("\nSafety:")
        pprint(response["safety"])


if __name__ == "__main__":
    main()
