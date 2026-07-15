"""
API key authentication middleware.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from core.config import API_KEY

PUBLIC_PATHS = frozenset(
    {
        "/health",
        "/ready",
        "/openapi.json",
        "/docs",
        "/redoc",
    }
)


def extract_api_key(request: Request) -> str:
    """Read API key from X-API-Key header or Authorization: Bearer."""
    header_key = request.headers.get("X-API-Key", "").strip()
    if header_key:
        return header_key

    authorization = request.headers.get("Authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require a valid API key on protected routes when API_KEY is configured."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        if not API_KEY:
            return await call_next(request)

        provided = extract_api_key(request)
        if provided != API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)
