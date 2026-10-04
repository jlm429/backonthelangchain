"""FastAPI entry point for the unified support system and legacy examples."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backonthelangchain.api.config import (
    AppSettings,
    cors_origins_from_environment,
)
from backonthelangchain.api.rate_limit import InMemoryRateLimiter
from backonthelangchain.api.schemas import (
    ExampleMetadataResponse,
    ExampleRunRequest,
    ExampleRunResponse,
    SupportGraphOptions,
    SupportGraphResponse,
    SupportRunRequest,
    MAX_REQUEST_BODY_BYTES,
)
from backonthelangchain.examples.registry import ExampleRegistry, build_example_registry
from backonthelangchain.examples.unified_support import (
    UnifiedSupportRunner,
    describe_unified_support_graph,
)

SettingsProvider = Callable[[], AppSettings]


class RequestBodyLimitMiddleware:
    """Reject request bodies that exceed a byte limit while streaming."""

    def __init__(self, app: Any, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", ()))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = self.max_body_bytes + 1
            if declared_size < 0 or declared_size > self.max_body_bytes:
                await self._send_too_large(scope, receive, send)
                return

        body = bytearray()
        disconnected = False
        while True:
            message = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > self.max_body_bytes:
                    await self._send_too_large(scope, receive, send)
                    return
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                disconnected = True
                break

        request_body = bytes(body)
        body.clear()
        replayed = False

        async def replay_receive() -> dict:
            nonlocal replayed
            if not replayed:
                replayed = True
                if disconnected:
                    return {"type": "http.disconnect"}
                return {
                    "type": "http.request",
                    "body": request_body,
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _send_too_large(scope: dict, receive: Any, send: Any) -> None:
        response = error_response(
            413,
            "request_too_large",
            "The request body is too large.",
        )
        await response(scope, receive, send)


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """Build the only error shape returned by application routes."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def create_app(
    *,
    registry: ExampleRegistry | None = None,
    settings_provider: SettingsProvider = AppSettings.from_environment,
    rate_limiter: InMemoryRateLimiter | None = None,
    cors_origins: tuple[str, ...] | None = None,
    support_runner: UnifiedSupportRunner | None = None,
) -> FastAPI:
    """Create an injectable API application for production and tests."""
    app = FastAPI(
        title="backonthelangchain API",
        description="Server-side execution for the unified LangGraph support system.",
        version="0.1.0",
    )
    active_registry = registry or build_example_registry()
    limiter = rate_limiter or InMemoryRateLimiter()
    active_support_runner = support_runner or UnifiedSupportRunner()
    allowed_origins = (
        cors_origins_from_environment() if cors_origins is None else cors_origins
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=MAX_REQUEST_BODY_BYTES,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request,
        _error: RequestValidationError,
    ) -> JSONResponse:
        return error_response(
            422,
            "invalid_request",
            "Enter a query between 1 and 2000 characters.",
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        _request: Request,
        _error: Exception,
    ) -> JSONResponse:
        return error_response(
            500,
            "internal_error",
            "The request could not be completed. Please try again later.",
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get(
        "/api/examples",
        response_model=list[ExampleMetadataResponse],
    )
    async def list_examples() -> list[dict]:
        return active_registry.public_catalog()

    @app.get(
        "/api/support/graph",
        response_model=SupportGraphResponse,
    )
    async def support_graph(faq_retrieval: bool = False) -> dict[str, Any]:
        """Return a description derived from the executable graph builder."""
        return describe_unified_support_graph(
            SupportGraphOptions(faq_retrieval=faq_retrieval)
        )

    @app.post("/api/support/run")
    async def run_support_graph(
        payload: SupportRunRequest,
        request: Request,
    ):
        """Stream application-level lifecycle events from LangGraph."""
        client_id = request.client.host if request.client else "unknown"
        if not await limiter.allow(client_id):
            response = error_response(
                429,
                "rate_limit_exceeded",
                "Too many runs. Please wait before trying again.",
            )
            response.headers["Retry-After"] = "60"
            return response

        required_configuration = ("OPENAI_API_KEY", "TYPESAFE_API_KEY")
        if settings_provider().missing(required_configuration):
            return error_response(
                503,
                "service_not_configured",
                "This support workflow is not configured on the server.",
            )

        async def stream_events():
            async for event in active_support_runner.events(payload):
                yield f"event: execution\ndata: {json.dumps(event)}\n\n"

        return StreamingResponse(
            stream_events(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    @app.post(
        "/api/examples/{example_id}/run",
        response_model=ExampleRunResponse,
    )
    async def run_example(
        example_id: str,
        payload: ExampleRunRequest,
        request: Request,
    ):
        example = active_registry.get(example_id)
        if example is None:
            return error_response(
                404,
                "example_not_found",
                "The requested example is not available.",
            )

        client_id = request.client.host if request.client else "unknown"
        if not await limiter.allow(client_id):
            response = error_response(
                429,
                "rate_limit_exceeded",
                "Too many runs. Please wait before trying again.",
            )
            response.headers["Retry-After"] = "60"
            return response

        if settings_provider().missing(example.required_configuration):
            return error_response(
                503,
                "service_not_configured",
                "This example is not configured on the server.",
            )

        try:
            result = await example.handler(payload.query)
        except Exception:
            return error_response(
                502,
                "example_failed",
                "The example could not be completed. Please try again later.",
            )
        return ExampleRunResponse(example_id=example.id, result=result)

    return app


app = create_app()
