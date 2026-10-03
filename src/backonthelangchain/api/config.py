"""Server-only configuration validation."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppSettings:
    """Provider configuration that is never serialized into API responses."""

    openai_api_key: str | None
    typesafe_api_key: str | None

    @classmethod
    def from_environment(cls) -> AppSettings:
        """Read required provider values without logging or returning them."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            typesafe_api_key=os.getenv("TYPESAFE_API_KEY"),
        )

    def missing(self, variable_names: tuple[str, ...]) -> tuple[str, ...]:
        """Return missing variable names, never their values."""
        configured = {
            "OPENAI_API_KEY": self.openai_api_key,
            "TYPESAFE_API_KEY": self.typesafe_api_key,
        }
        return tuple(name for name in variable_names if not configured.get(name))


def cors_origins_from_environment() -> tuple[str, ...]:
    """Return exact allowed origins with safe local-development defaults."""
    configured = os.getenv("BACKONTHELANGCHAIN_CORS_ORIGINS")
    if not configured:
        return ("http://localhost:3000", "http://127.0.0.1:3000")
    return tuple(origin.strip() for origin in configured.split(",") if origin.strip())
