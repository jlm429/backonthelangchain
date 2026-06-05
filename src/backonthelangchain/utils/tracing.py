"""Tracing helpers.
"""

import os


def langsmith_enabled() -> bool:
    """Return True when LangSmith tracing is enabled."""
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true"


def enable_langsmith_tracing(project_name: str | None = None) -> None:
    """Enable LangSmith tracing for the current process."""

    os.environ["LANGSMITH_TRACING"] = "true"

    if project_name:
        os.environ["LANGSMITH_PROJECT"] = project_name


def disable_langsmith_tracing() -> None:
    """Disable LangSmith tracing for the current process."""

    os.environ["LANGSMITH_TRACING"] = "false"