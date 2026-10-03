"""Single catalog of examples exposed by the backend."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from backonthelangchain.examples.jev_support import handle_jev_support_router

ExampleHandler = Callable[[str], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ExampleDefinition:
    """Metadata and an async-capable backend handler for one example."""

    id: str
    display_name: str
    description: str
    default_prompt: str
    sample_prompts: tuple[str, ...]
    handler: ExampleHandler
    required_configuration: tuple[str, ...] = ()

    def public_metadata(self) -> dict[str, Any]:
        """Return only browser-safe catalog fields."""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "default_prompt": self.default_prompt,
            "sample_prompts": list(self.sample_prompts),
        }


class ExampleRegistry:
    """Ordered registry with explicit duplicate protection."""

    def __init__(self, examples: Iterable[ExampleDefinition] = ()) -> None:
        self._examples: dict[str, ExampleDefinition] = {}
        for example in examples:
            self.register(example)

    def register(self, example: ExampleDefinition) -> None:
        """Register an example once."""
        if example.id in self._examples:
            raise ValueError(f"Example id is already registered: {example.id}")
        self._examples[example.id] = example

    def get(self, example_id: str) -> ExampleDefinition | None:
        """Find an example without exposing handler internals."""
        return self._examples.get(example_id)

    def public_catalog(self) -> list[dict[str, Any]]:
        """Return browser-safe metadata in registration order."""
        return [example.public_metadata() for example in self._examples.values()]


def build_example_registry(
    *,
    jev_handler: ExampleHandler = handle_jev_support_router,
) -> ExampleRegistry:
    """Build the application catalog with the Jev router registered first."""
    samples = (
        "I cannot log in after enabling MFA.",
        "I was charged twice for my subscription.",
        (
            "I have reset my password five times and still cannot log in. "
            "Connect me to a human."
        ),
        (
            "Our production checkout is down and the entire business is blocked. "
            "I need human support immediately."
        ),
    )
    return ExampleRegistry(
        [
            ExampleDefinition(
                id="jev-support-router",
                display_name="Jev Support Router",
                description=(
                    "Moderates the request first, then uses Jev for escalation and "
                    "support routing with an OpenAI fallback for uncertain routes."
                ),
                default_prompt=samples[0],
                sample_prompts=samples,
                handler=jev_handler,
                required_configuration=("OPENAI_API_KEY", "TYPESAFE_API_KEY"),
            )
        ]
    )
