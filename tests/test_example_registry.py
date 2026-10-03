import pytest

from backonthelangchain.examples.registry import (
    ExampleDefinition,
    ExampleRegistry,
    build_example_registry,
)


async def fake_handler(query: str) -> dict:
    return {"query": query}


def example_definition(example_id: str = "example") -> ExampleDefinition:
    return ExampleDefinition(
        id=example_id,
        display_name="Example",
        description="An example.",
        default_prompt="Hello",
        sample_prompts=("Hello",),
        handler=fake_handler,
    )


def test_registry_preserves_order_and_returns_only_public_metadata():
    registry = ExampleRegistry(
        [example_definition("first"), example_definition("second")]
    )

    assert [item["id"] for item in registry.public_catalog()] == ["first", "second"]
    assert "handler" not in registry.public_catalog()[0]
    assert registry.get("first").handler is fake_handler
    assert registry.get("missing") is None


def test_registry_rejects_duplicate_ids():
    registry = ExampleRegistry([example_definition()])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(example_definition())


def test_default_registry_contains_the_required_jev_samples():
    catalog = build_example_registry(jev_handler=fake_handler).public_catalog()

    assert len(catalog) == 1
    assert catalog[0]["id"] == "jev-support-router"
    assert catalog[0]["sample_prompts"] == [
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
    ]
