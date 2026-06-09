# src/vigil/_exceptions.py

class VigilError(Exception):
    """Base for all vigil SDK errors. Users catch this to handle ALL SDK failures."""

class VigilAuthError(VigilError):
    """401 — API key missing, malformed, or revoked."""

class VigilValidationError(VigilError):
    """400 — payload rejected by server validator."""

class VigilRateLimited(VigilError):
    """429 — per-IP or per-key bucket exhausted."""
    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after

class VigilBufferFull(VigilError):
    """503 — ingest buffer full server-side. Retry after Retry-After seconds."""
    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after

class VigilServerError(VigilError):
    """5xx — server bug or DB outage."""

class VigilTransportError(VigilError):
    """Network failure — DNS, TCP, TLS, connection refused, timeout."""
