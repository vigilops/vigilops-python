"""Helpers + drop-in wrappers for the OpenAI SDK.

Symmetric to `keelwave.adapters.anthropic`. `parse_response` exposes the
universal observability fields (id, tokens, finish_reason) from a
chat.completions response. `wrap_client` returns a proxy that records
every `chat.completions.create` call to keelwave's `ai_traces` table —
auto-linked to the active `Run` if one is open (via ContextVar).

OpenAI is an optional dependency: imports happen inside the wrapper.

Field-name notes (OpenAI vs Anthropic):
- usage.prompt_tokens     <- vs usage.input_tokens
- usage.completion_tokens <- vs usage.output_tokens
- usage.total_tokens      <- already provided by OpenAI
- choices[0].finish_reason <- vs stop_reason
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._context import get_current_run

if TYPE_CHECKING:
    from ..async_client import AsyncKeelwave
    from ..client import Keelwave


@dataclass
class ParsedOpenAIResponse:
    """Universal observability fields extracted from a
    chat.completions.create response.

    Field names normalized to keelwave's internal vocab (input_tokens /
    output_tokens) so dashboards don't have to know which provider sent
    the trace. The full response stays in `raw` for the caller to walk
    `choices[0].message.content` / `.tool_calls` themselves.
    """

    request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    raw: Any = None

    def tokens_total(self) -> int:
        if self.total_tokens is not None:
            return self.total_tokens
        if self.input_tokens is not None and self.output_tokens is not None:
            return self.input_tokens + self.output_tokens
        return 0


def parse_response(resp: Any) -> ParsedOpenAIResponse:
    """Pull the universal observability fields off an OpenAI response.

    Does NOT extract message content or tool_calls. Walk
    `resp.choices[0].message` yourself for those.
    """
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = (
        getattr(usage, "completion_tokens", None) if usage is not None else None
    )
    total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None

    finish_reason: str | None = None
    choices = getattr(resp, "choices", None)
    if choices:
        finish_reason = getattr(choices[0], "finish_reason", None)

    return ParsedOpenAIResponse(
        request_id=getattr(resp, "id", None),
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
        finish_reason=finish_reason,
        raw=resp,
    )


# ---------------------------------------------------------------------------
# Sync wrapper


class _SyncCompletionsProxy:
    """Proxies openai.chat.completions — only `create` is intercepted."""

    def __init__(
        self, real_completions: Any, keelwave: Keelwave, provider: str
    ) -> None:
        self._real = real_completions
        self._keelwave = keelwave
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

    def _record(
        self, kwargs: dict, resp: Any, status: str, err: str | None, latency_ms: int
    ) -> None:
        parsed = parse_response(resp) if resp is not None else None
        run = get_current_run()
        try:
            self._keelwave.ingest_ai(
                model=kwargs.get("model", "unknown"),
                status=status,
                provider=self._provider,
                input_tokens=parsed.input_tokens if parsed else None,
                output_tokens=parsed.output_tokens if parsed else None,
                total_tokens=parsed.total_tokens if parsed else None,
                latency_ms=latency_ms,
                error_message=err,
                request_id=parsed.request_id if parsed else None,
                agent_run_id=run.id if run is not None else None,
                metadata={
                    "tool_choice": kwargs.get("tool_choice"),
                    "finish_reason": parsed.finish_reason if parsed else None,
                    "num_tools_in_request": len(kwargs.get("tools") or []),
                },
            )
        except Exception:
            pass

        if run is not None and resp is not None:
            try:
                _emit_think_steps_sync(run, resp, parsed)
            except Exception:
                pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _SyncChatProxy:
    """Sits between the OpenAI client and its `.completions` attr."""

    def __init__(self, real_chat: Any, keelwave: Keelwave, provider: str) -> None:
        self._real = real_chat
        self.completions = _SyncCompletionsProxy(
            real_chat.completions, keelwave, provider
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _SyncOpenAIProxy:
    """Drop-in wrapper around openai.OpenAI. Only `.chat.completions`
    is intercepted; everything else delegates to the real client."""

    def __init__(self, real_client: Any, keelwave: Keelwave, provider: str) -> None:
        self._real = real_client
        self.chat = _SyncChatProxy(real_client.chat, keelwave, provider)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_client(
    client: Any, keelwave: Keelwave, *, provider: str = "openai"
) -> _SyncOpenAIProxy:
    """Return a proxy that auto-records every chat.completions.create call."""
    return _SyncOpenAIProxy(client, keelwave, provider)


# ---------------------------------------------------------------------------
# Think-step helpers


def _extract_reasoning(resp: Any, parsed: Any) -> str | None:
    """Return reasoning text from o1/o3 (.reasoning) or DeepSeek (.reasoning_content)."""
    choices = getattr(resp, "choices", None)
    if not choices:
        return None
    msg = getattr(choices[0], "message", None)
    if msg is None:
        return None
    reasoning = getattr(msg, "reasoning", None)
    if reasoning:
        return str(reasoning)
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        return str(reasoning)
    return None


def _emit_think_steps_sync(run: Any, resp: Any, parsed: Any) -> None:
    reasoning = _extract_reasoning(resp, parsed)
    if reasoning:
        tokens = parsed.tokens_total() if parsed else None
        run.step("think", content=reasoning, tokens=tokens)


async def _emit_think_steps_async(run: Any, resp: Any, parsed: Any) -> None:
    reasoning = _extract_reasoning(resp, parsed)
    if reasoning:
        tokens = parsed.tokens_total() if parsed else None
        await run.step("think", content=reasoning, tokens=tokens)


# ---------------------------------------------------------------------------
# Async wrapper


class _AsyncCompletionsProxy:
    def __init__(
        self, real_completions: Any, keelwave: AsyncKeelwave, provider: str
    ) -> None:
        self._real = real_completions
        self._keelwave = keelwave
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
            await self._record(
                kwargs, resp, status, err, int((time.monotonic() - t0) * 1000)
            )

    async def _record(
        self, kwargs: dict, resp: Any, status: str, err: str | None, latency_ms: int
    ) -> None:
        parsed = parse_response(resp) if resp is not None else None
        run = get_current_run()
        try:
            await self._keelwave.ingest_ai(
                model=kwargs.get("model", "unknown"),
                status=status,
                provider=self._provider,
                input_tokens=parsed.input_tokens if parsed else None,
                output_tokens=parsed.output_tokens if parsed else None,
                total_tokens=parsed.total_tokens if parsed else None,
                latency_ms=latency_ms,
                error_message=err,
                request_id=parsed.request_id if parsed else None,
                agent_run_id=run.id if run is not None else None,
                metadata={
                    "tool_choice": kwargs.get("tool_choice"),
                    "finish_reason": parsed.finish_reason if parsed else None,
                    "num_tools_in_request": len(kwargs.get("tools") or []),
                },
            )
        except Exception:
            pass

        if run is not None and resp is not None:
            try:
                await _emit_think_steps_async(run, resp, parsed)
            except Exception:
                pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _AsyncChatProxy:
    def __init__(self, real_chat: Any, keelwave: AsyncKeelwave, provider: str) -> None:
        self._real = real_chat
        self.completions = _AsyncCompletionsProxy(
            real_chat.completions, keelwave, provider
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _AsyncOpenAIProxy:
    def __init__(
        self, real_client: Any, keelwave: AsyncKeelwave, provider: str
    ) -> None:
        self._real = real_client
        self.chat = _AsyncChatProxy(real_client.chat, keelwave, provider)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_async_client(
    client: Any, keelwave: AsyncKeelwave, *, provider: str = "openai"
) -> _AsyncOpenAIProxy:
    return _AsyncOpenAIProxy(client, keelwave, provider)
