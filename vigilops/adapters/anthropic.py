"""Helpers + drop-in wrappers for the Anthropic SDK.

`parse_response` collapses the ThinkingBlock / TextBlock / ToolUseBlock
dispatch every Anthropic caller writes by hand. `wrap_client` returns a
proxy that intercepts `.messages.create(...)`, times the call, and
records a row in vigil's `ai_traces` table — auto-linked to the active
`Run` if one is open (via the ContextVar set in Run.__enter__).

Anthropic is an optional dependency: imports happen inside functions so
core SDK install does not require the package.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._context import get_current_run

if TYPE_CHECKING:
    from ..async_client import AsyncVigil
    from ..client import Vigil


@dataclass
class ParsedAnthropicResponse:
    """Universal observability fields extracted from an Anthropic
    messages.create response.

    Only fields that have been stable across Anthropic SDK versions
    appear here — id, usage, stop_reason. The full response is kept in
    `raw` so callers can walk `raw.content` (text / thinking / tool_use
    blocks) themselves; that surface evolves and is not vigil's job to
    track.
    """

    request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    stop_reason: str | None = None
    raw: Any = None

    def tokens_total(self) -> int:
        """input + output, falling back to total_tokens, else 0."""
        if self.input_tokens is not None and self.output_tokens is not None:
            return self.input_tokens + self.output_tokens
        return self.total_tokens or 0


def parse_response(resp: Any) -> ParsedAnthropicResponse:
    """Pull the universal observability fields off an Anthropic response.

    Does NOT flatten content blocks (text / thinking / tool_use). Those
    evolve. Walk `resp.content` yourself if you need them.
    """
    usage = getattr(resp, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
    output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None

    return ParsedAnthropicResponse(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=(input_tokens + output_tokens) if (input_tokens is not None and output_tokens is not None) else None,
        request_id=getattr(resp, "id", None),
        stop_reason=getattr(resp, "stop_reason", None),
        raw=resp,
    )


# ---------------------------------------------------------------------------
# Sync wrapper

class _SyncMessagesProxy:
    """Proxies anthropic.Messages — only `create` is intercepted."""

    def __init__(self, real_messages: Any, vigil: Vigil, provider: str) -> None:
        self._real = real_messages
        self._vigil = vigil
        self._provider = provider

    def create(self, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        status = "success"
        err: str | None = None
        resp: Any = None
        try:
            resp = self._real.create(**kwargs)
            return resp
        except Exception as e:
            status = "error"
            err = str(e)[:2000]
            raise
        finally:
            self._record(kwargs, resp, status, err, int((time.monotonic() - t0) * 1000))

    def _record(self, kwargs: dict, resp: Any, status: str, err: str | None, latency_ms: int) -> None:
        usage = getattr(resp, "usage", None) if resp is not None else None
        run = get_current_run()
        try:
            self._vigil.ingest_ai(
                model=kwargs.get("model", "unknown"),
                status=status,
                provider=self._provider,
                input_tokens=getattr(usage, "input_tokens", None) if usage else None,
                output_tokens=getattr(usage, "output_tokens", None) if usage else None,
                latency_ms=latency_ms,
                error_message=err,
                request_id=getattr(resp, "id", None) if resp is not None else None,
                agent_run_id=run.id if run is not None else None,
                metadata={
                    "tool_choice": kwargs.get("tool_choice"),
                    "stop_reason": getattr(resp, "stop_reason", None) if resp is not None else None,
                    "num_tools_in_request": len(kwargs.get("tools") or []),
                },
            )
        except Exception:
            # Observability must never crash the user's program.
            pass

        # Auto-emit think steps from thinking blocks — zero user code needed.
        if run is not None and resp is not None:
            try:
                _emit_think_steps_sync(run, resp, usage)
            except Exception:
                pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _SyncAnthropicProxy:
    """Drop-in wrapper around anthropic.Anthropic. Only `.messages` is
    swapped; every other attribute delegates to the real client."""

    def __init__(self, real_client: Any, vigil: Vigil, provider: str) -> None:
        self._real = real_client
        self.messages = _SyncMessagesProxy(real_client.messages, vigil, provider)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_client(client: Any, vigil: Vigil, *, provider: str = "anthropic") -> _SyncAnthropicProxy:
    """Return a proxy that auto-records every messages.create() call."""
    return _SyncAnthropicProxy(client, vigil, provider)


# ---------------------------------------------------------------------------
# Think-step helpers

def _extract_thinking_blocks(resp: Any, usage: Any) -> list[dict]:
    """Return list of {content, tokens} for each thinking block in resp."""
    content = getattr(resp, "content", None) or []
    input_tokens = getattr(usage, "input_tokens", None) if usage else None
    output_tokens = getattr(usage, "output_tokens", None) if usage else None
    total = (input_tokens + output_tokens) if (input_tokens and output_tokens) else None
    return [
        {"content": getattr(block, "thinking", ""), "tokens": total}
        for block in content
        if getattr(block, "type", None) == "thinking"
    ]


def _emit_think_steps_sync(run: Any, resp: Any, usage: Any) -> None:
    for block in _extract_thinking_blocks(resp, usage):
        run.step("think", content=block["content"], tokens=block["tokens"])


async def _emit_think_steps_async(run: Any, resp: Any, usage: Any) -> None:
    for block in _extract_thinking_blocks(resp, usage):
        await run.step("think", content=block["content"], tokens=block["tokens"])


# ---------------------------------------------------------------------------
# Async wrapper — mirrors the sync version

class _AsyncMessagesProxy:
    def __init__(self, real_messages: Any, vigil: AsyncVigil, provider: str) -> None:
        self._real = real_messages
        self._vigil = vigil
        self._provider = provider

    async def create(self, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        status = "success"
        err: str | None = None
        resp: Any = None
        try:
            resp = await self._real.create(**kwargs)
            return resp
        except Exception as e:
            status = "error"
            err = str(e)[:2000]
            raise
        finally:
            await self._record(kwargs, resp, status, err, int((time.monotonic() - t0) * 1000))

    async def _record(self, kwargs: dict, resp: Any, status: str, err: str | None, latency_ms: int) -> None:
        usage = getattr(resp, "usage", None) if resp is not None else None
        run = get_current_run()
        try:
            await self._vigil.ingest_ai(
                model=kwargs.get("model", "unknown"),
                status=status,
                provider=self._provider,
                input_tokens=getattr(usage, "input_tokens", None) if usage else None,
                output_tokens=getattr(usage, "output_tokens", None) if usage else None,
                latency_ms=latency_ms,
                error_message=err,
                request_id=getattr(resp, "id", None) if resp is not None else None,
                agent_run_id=run.id if run is not None else None,
                metadata={
                    "tool_choice": kwargs.get("tool_choice"),
                    "stop_reason": getattr(resp, "stop_reason", None) if resp is not None else None,
                    "num_tools_in_request": len(kwargs.get("tools") or []),
                },
            )
        except Exception:
            pass

        # Auto-emit think steps from thinking blocks — zero user code needed.
        if run is not None and resp is not None:
            try:
                await _emit_think_steps_async(run, resp, usage)
            except Exception:
                pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _AsyncAnthropicProxy:
    def __init__(self, real_client: Any, vigil: AsyncVigil, provider: str) -> None:
        self._real = real_client
        self.messages = _AsyncMessagesProxy(real_client.messages, vigil, provider)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_async_client(client: Any, vigil: AsyncVigil, *, provider: str = "anthropic") -> _AsyncAnthropicProxy:
    """Async sibling of wrap_client."""
    return _AsyncAnthropicProxy(client, vigil, provider)
