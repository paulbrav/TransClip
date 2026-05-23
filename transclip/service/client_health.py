from __future__ import annotations

from urllib.error import URLError

from transclip.settings import Settings

from .client import InferenceClient
from .types import ServiceHealthResponse

SERVICE_READY_STATUSES = frozenset({"ready", "recording"})

_CLIENT_HEALTH_ERRORS = (URLError, OSError, ConnectionError)


def fetch_service_health_result(
    settings: Settings,
) -> tuple[ServiceHealthResponse | None, str | None]:
    try:
        return InferenceClient(settings).health(), None
    except _CLIENT_HEALTH_ERRORS as exc:
        return None, str(exc)


def service_health_is_ready(health: ServiceHealthResponse | None) -> bool:
    return health is not None and health.get("status") in SERVICE_READY_STATUSES


def service_health_check_detail(
    health: ServiceHealthResponse | None,
    *,
    error: str | None = None,
) -> str:
    if error is not None:
        return f"/health failed: {error}"
    status = health.get("status") if health is not None else None
    return (
        f"/health status={status}; asr={health.get('asr_backend')}; "
        f"cleanup={health.get('cleanup_backend')}"
        if health is not None
        else "/health failed: no response"
    )
