"""FastAPI entry point for the interactive example catalog."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backonthelangchain.api.config import (
    AppSettings,
    cors_origins_from_environment,
)
from backonthelangchain.api.rate_limit import InMemoryRateLimiter
from backonthelangchain.api.schemas import (
    ExampleMetadataResponse,
    ExampleRunRequest,
    ExampleRunResponse,
    MAX_REQUEST_BODY_BYTES,
)
from backonthelangchain.examples.registry import ExampleRegistry, build_example_registry

SettingsProvider = Callable[[], AppSettings]


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
) -> FastAPI:
    """Create an injectable API application for production and tests."""
    app = FastAPI(
        title="backonthelangchain API",
        description="Server-side execution for interactive agent examples.",
        version="0.1.0",
    )
    active_registry = registry or build_example_registry()
    limiter = rate_limiter or InMemoryRateLimiter()
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
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                body_size = int(content_length)
            except ValueError:
                body_size = MAX_REQUEST_BODY_BYTES + 1
            if body_size > MAX_REQUEST_BODY_BYTES:
                response = error_response(
                    413,
                    "request_too_large",
                    "The request body is too large.",
                )
            else:
                response = await call_next(request)
        else:
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
