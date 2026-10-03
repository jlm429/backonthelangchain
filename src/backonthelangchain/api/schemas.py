"""HTTP request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_QUERY_LENGTH = 2_000
MAX_REQUEST_BODY_BYTES = 16_384


class ExampleMetadataResponse(BaseModel):
    """Browser-safe example metadata."""

    id: str
    display_name: str
    description: str
    default_prompt: str
    sample_prompts: list[str]


class ExampleRunRequest(BaseModel):
    """Validated provider-backed request."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """Reject whitespace-only input and avoid sending padding to providers."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Query must contain non-whitespace characters.")
        return normalized


class ExampleRunResponse(BaseModel):
    """Stable envelope for any registered example result."""

    example_id: str
    result: dict[str, Any]
