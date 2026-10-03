"""Run the safety-gated experimental Jev support router.

Usage:
    poetry run python examples/run_safe_jev_support_router.py
    poetry run python examples/run_safe_jev_support_router.py "I was charged twice"

Required environment variables:
    OPENAI_API_KEY
    TYPESAFE_API_KEY
"""

import sys
from pprint import pprint

from backonthelangchain.examples.jev_support import run_jev_support_router
from backonthelangchain.utils.env import load_project_env


def main() -> None:
    load_project_env()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("User query> ").strip()

    response = run_jev_support_router(query)

    print("\nResult:")
    pprint(response.model_dump(mode="json"))


if __name__ == "__main__":
    main()
