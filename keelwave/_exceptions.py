# src/keelwave/_exceptions.py


class KeelwaveError(Exception):
    """Base for all keelwave SDK errors. Users catch this to handle ALL SDK failures."""


class KeelwaveAuthError(KeelwaveError):
    """401 — API key missing, malformed, or revoked."""


class KeelwaveValidationError(KeelwaveError):
    """400 — payload rejected by server validator."""


class KeelwaveRateLimited(KeelwaveError):
    """429 — per-IP or per-key bucket exhausted."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class KeelwaveBufferFull(KeelwaveError):
    """503 — ingest buffer full server-side. Retry after Retry-After seconds."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class KeelwaveServerError(KeelwaveError):
    """5xx — server bug or DB outage."""


class KeelwaveTransportError(KeelwaveError):
    """Network failure — DNS, TCP, TLS, connection refused, timeout."""
