"""HTTP request and response schemas."""

from __future__ import annotations

from typing import Any, Literal

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


class SupportGraphOptions(BaseModel):
    """The only stage callers may enable or disable."""

    model_config = ConfigDict(extra="forbid")

    faq_retrieval: bool = False


class SimulatedSystemState(BaseModel):
    """Explicit demo state supplied by the browser, not real monitoring data."""

    model_config = ConfigDict(extra="forbid")

    authentication: Literal["operational", "degraded", "outage"]
    billing: Literal["operational", "degraded", "outage"]
    checkout: Literal["operational", "degraded", "outage"]
    api: Literal["operational", "degraded", "outage"]


class SupportRunRequest(BaseModel):
    """Validated input for a streamed unified support execution."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    options: SupportGraphOptions = Field(default_factory=SupportGraphOptions)
    simulated_status: SimulatedSystemState

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """Reject whitespace-only input and avoid sending padding to providers."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Query must contain non-whitespace characters.")
        return normalized


class GraphNodeResponse(BaseModel):
    """Browser-safe application node metadata."""

    id: str
    label: str
    description: str
    kind: str
    required: bool
    enabled: bool
    stage: int


class GraphEdgeResponse(BaseModel):
    """An edge serialized from the compiled LangGraph graph."""

    source: str
    target: str
    conditional: bool = False
    branch: str | None = None


class SupportGraphResponse(BaseModel):
    """Description derived from the same builder used for execution."""

    graph_id: str
    options: SupportGraphOptions
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]
