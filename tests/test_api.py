import asyncio
import json
from collections import deque

from fastapi.testclient import TestClient

from backonthelangchain.api.app import create_app
from backonthelangchain.api.config import AppSettings
from backonthelangchain.api.rate_limit import InMemoryRateLimiter
from backonthelangchain.api.schemas import MAX_REQUEST_BODY_BYTES
from backonthelangchain.examples.registry import build_example_registry

OPENAI_SENTINEL = "openai-test-secret-never-return"
TYPESAFE_SENTINEL = "typesafe-test-secret-never-return"


def configured_settings() -> AppSettings:
    return AppSettings(
        openai_api_key=OPENAI_SENTINEL,
        typesafe_api_key=TYPESAFE_SENTINEL,
    )


def make_client(handler, *, settings_provider=configured_settings, limiter=None):
    registry = build_example_registry(jev_handler=handler)
    app = create_app(
        registry=registry,
        settings_provider=settings_provider,
        rate_limiter=limiter,
        cors_origins=("http://localhost:3000",),
    )
    return TestClient(app)


def post_body_chunks(app, chunks: list[bytes]) -> tuple[int, dict]:
    async def invoke() -> tuple[int, dict]:
        requests = deque(
            {
                "type": "http.request",
                "body": chunk,
                "more_body": index < len(chunks) - 1,
            }
            for index, chunk in enumerate(chunks)
        )
        responses = []

        async def receive():
            if requests:
                return requests.popleft()
            return {"type": "http.disconnect"}

        async def send(message):
            responses.append(message)

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/examples/jev-support-router/run",
                "raw_path": b"/api/examples/jev-support-router/run",
                "query_string": b"",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"transfer-encoding", b"chunked"),
                ],
                "client": ("testclient", 50000),
                "server": ("testserver", 80),
                "root_path": "",
            },
            receive,
            send,
        )
        start = next(
            message
            for message in responses
            if message["type"] == "http.response.start"
        )
        body = b"".join(
            message.get("body", b"")
            for message in responses
            if message["type"] == "http.response.body"
        )
        return start["status"], json.loads(body)

    return asyncio.run(invoke())


async def successful_handler(query: str) -> dict:
    return {
        "outcome": "completed",
        "answer": "Try a recovery code.",
        "safety": {"allowed": True, "model": "fake", "reason": "Allowed."},
        "routing": {
            "destination": "tech_support",
            "reason": "Jev selected technical support.",
            "used_fallback": False,
        },
        "jev": {
            "model": "jev-1.13.0",
            "route_confidence": 0.91,
            "route_probabilities": {"tech_support": 0.91, "billing": 0.09},
            "human_escalation_probability": 0.05,
        },
        "received_query": query,
    }


def test_catalog_and_jev_endpoint_use_backend_registry():
    client = make_client(successful_handler)

    catalog_response = client.get("/api/examples")
    run_response = client.post(
        "/api/examples/jev-support-router/run",
        json={"query": "  I cannot log in after enabling MFA.  "},
    )

    assert catalog_response.status_code == 200
    assert catalog_response.json()[0]["display_name"] == "Jev Support Router"
    assert run_response.status_code == 200
    body = run_response.json()
    assert body["example_id"] == "jev-support-router"
    assert body["result"]["routing"]["destination"] == "tech_support"
    assert body["result"]["received_query"] == "I cannot log in after enabling MFA."


def test_invalid_example_id_is_safe():
    client = make_client(successful_handler)

    response = client.post("/api/examples/unknown/run", json={"query": "hello"})

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "example_not_found",
            "message": "The requested example is not available.",
        }
    }


def test_empty_and_oversized_queries_are_rejected_without_echoing_input():
    client = make_client(successful_handler)
    oversized = "private-user-input" * 200

    empty_response = client.post(
        "/api/examples/jev-support-router/run", json={"query": "   "}
    )
    oversized_response = client.post(
        "/api/examples/jev-support-router/run", json={"query": oversized}
    )

    assert empty_response.status_code == 422
    assert oversized_response.status_code == 422
    assert oversized not in oversized_response.text
    assert oversized_response.json()["error"]["code"] == "invalid_request"


def test_oversized_request_body_is_rejected_before_validation():
    client = make_client(successful_handler)
    oversized = "sensitive-padding" * 2_000

    response = client.post(
        "/api/examples/jev-support-router/run", json={"query": oversized}
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert oversized not in response.text


def test_one_byte_chunks_are_coalesced_before_validation():
    registry = build_example_registry(jev_handler=successful_handler)
    app = create_app(
        registry=registry,
        settings_provider=configured_settings,
        cors_origins=("http://localhost:3000",),
    )
    request_body = json.dumps({"query": "one byte at a time"}).encode()

    status, body = post_body_chunks(
        app,
        [request_body[index : index + 1] for index in range(len(request_body))],
    )

    assert status == 200
    assert body["result"]["received_query"] == "one byte at a time"


def test_oversized_one_byte_chunks_are_rejected_while_streaming():
    registry = build_example_registry(jev_handler=successful_handler)
    app = create_app(
        registry=registry,
        settings_provider=configured_settings,
        cors_origins=("http://localhost:3000",),
    )

    status, body = post_body_chunks(app, [b"x"] * (MAX_REQUEST_BODY_BYTES + 1))

    assert status == 413
    assert body["error"]["code"] == "request_too_large"


def test_missing_configuration_stops_before_handler_runs():
    calls = []

    async def handler(query: str) -> dict:
        calls.append(query)
        return {}

    client = make_client(
        handler,
        settings_provider=lambda: AppSettings(
            openai_api_key=None,
            typesafe_api_key=None,
        ),
    )

    response = client.post(
        "/api/examples/jev-support-router/run", json={"query": "hello"}
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_not_configured"
    assert calls == []


def test_provider_failure_is_opaque_and_never_returns_configured_keys():
    async def failing_handler(_query: str) -> dict:
        raise RuntimeError(
            f"provider authorization failed: {OPENAI_SENTINEL} {TYPESAFE_SENTINEL}"
        )

    client = make_client(failing_handler)

    response = client.post(
        "/api/examples/jev-support-router/run", json={"query": "private query"}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "example_failed"
    assert OPENAI_SENTINEL not in response.text
    assert TYPESAFE_SENTINEL not in response.text
    assert "private query" not in response.text


def test_rate_limiter_blocks_additional_paid_runs():
    limiter = InMemoryRateLimiter(requests=1, window_seconds=60)
    client = make_client(successful_handler, limiter=limiter)

    first = client.post(
        "/api/examples/jev-support-router/run", json={"query": "first"}
    )
    second = client.post(
        "/api/examples/jev-support-router/run", json={"query": "second"}
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "60"


def test_cors_allows_only_the_configured_origin():
    client = make_client(successful_handler)

    allowed = client.options(
        "/api/examples",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    denied = client.options(
        "/api/examples",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in denied.headers
