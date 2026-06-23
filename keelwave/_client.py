import httpx

from ._exceptions import (
    KeelwaveAuthError,
    KeelwaveBufferFull,
    KeelwaveError,
    KeelwaveRateLimited,
    KeelwaveServerError,
    KeelwaveValidationError,
)


def raise_for_status(resp: httpx.Response) -> None:
    if resp.is_success:
        return

    try:
        message = resp.json().get("error", resp.text)
    except ValueError:
        message = resp.text

    retry_after = parse_retry_after(resp.headers.get("Retry-After"))

    match resp.status_code:
        case 400:
            raise KeelwaveValidationError(message)
        case 401:
            raise KeelwaveAuthError(message)
        case 429:
            raise KeelwaveRateLimited(message, retry_after=retry_after)
        case 503:
            raise KeelwaveBufferFull(message, retry_after=retry_after)
        case code if 500 <= code < 600:
            raise KeelwaveServerError(f"{code}: {message}")
        case _:
            raise KeelwaveError(f"{resp.status_code}: {message}")


def parse_retry_after(raw: str | None) -> int | None:
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None
