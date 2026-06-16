from __future__ import annotations

import functools
import time
import warnings
from typing import Any, Callable, TypeVar

from ._context import get_current_run

F = TypeVar("F", bound=Callable[..., Any])

_SENTINEL = object()


class _Span:
    """Sync context manager that emits a step on the active run.

    Usage::

        with client.span("retrieval", name="fetch_docs") as span:
            docs = fetch(query)
            span.set(content=f"fetched {len(docs)} docs")
    """

    def __init__(self, vigil_client: Any, step_type: str, name: str | None) -> None:
        self._vigil = vigil_client
        self._step_type = step_type
        self._name = name
        self._content: str | None = None
        self._tokens: int | None = None
        self._metadata: dict | None = None
        self._t0: float = 0.0

    def set(
        self,
        content: str | None = None,
        *,
        tokens: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        if content is not None:
            self._content = content
        if tokens is not None:
            self._tokens = tokens
        if metadata is not None:
            self._metadata = metadata

    def __enter__(self) -> "_Span":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *_: Any) -> None:
        latency_ms = int((time.monotonic() - self._t0) * 1000)
        try:
            run = get_current_run()
            if run is not None:
                meta = {**(self._metadata or {}), "latency_ms": latency_ms}
                if self._name:
                    meta["span_name"] = self._name
                run.step(self._step_type, content=self._content, tokens=self._tokens, metadata=meta)
        except Exception as e:
            warnings.warn(f"vigil span emit failed: {e}", stacklevel=2)


class _AsyncSpan:
    """Async context manager that emits a step on the active async run.

    Usage::

        async with client.span("retrieval", name="fetch_docs") as span:
            docs = await fetch(query)
            span.set(content=f"fetched {len(docs)} docs")
    """

    def __init__(self, vigil_client: Any, step_type: str, name: str | None) -> None:
        self._vigil = vigil_client
        self._step_type = step_type
        self._name = name
        self._content: str | None = None
        self._tokens: int | None = None
        self._metadata: dict | None = None
        self._t0: float = 0.0

    def set(
        self,
        content: str | None = None,
        *,
        tokens: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        if content is not None:
            self._content = content
        if tokens is not None:
            self._tokens = tokens
        if metadata is not None:
            self._metadata = metadata

    async def __aenter__(self) -> "_AsyncSpan":
        self._t0 = time.monotonic()
        return self

    async def __aexit__(self, *_: Any) -> None:
        latency_ms = int((time.monotonic() - self._t0) * 1000)
        try:
            run = get_current_run()
            if run is not None:
                meta = {**(self._metadata or {}), "latency_ms": latency_ms}
                if self._name:
                    meta["span_name"] = self._name
                await run.step(self._step_type, content=self._content, tokens=self._tokens, metadata=meta)
        except Exception as e:
            warnings.warn(f"vigil span emit failed: {e}", stacklevel=2)


def make_observe(vigil_client: Any, name: str | None, step_type: str) -> Callable[[F], F]:
    """Return a decorator that instruments a sync function."""

    def decorator(fn: F) -> F:
        fn_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic()
            exc_caught: BaseException | None = None
            result: Any = _SENTINEL
            try:
                result = fn(*args, **kwargs)
                return result
            except BaseException as exc:
                exc_caught = exc
                raise
            finally:
                latency_ms = int((time.monotonic() - t0) * 1000)
                ok = exc_caught is None
                _emit_sync(
                    vigil_client,
                    fn_name=fn_name,
                    step_type=step_type,
                    args=args,
                    kwargs=kwargs,
                    result=result if result is not _SENTINEL else None,
                    ok=ok,
                    latency_ms=latency_ms,
                    error=exc_caught,
                )

        return wrapper  # type: ignore[return-value]

    return decorator


def make_observe_async(vigil_client: Any, name: str | None, step_type: str) -> Callable[[F], F]:
    """Return a decorator that instruments an async function."""

    def decorator(fn: F) -> F:
        fn_name = name or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic()
            exc_caught: BaseException | None = None
            result: Any = _SENTINEL
            try:
                result = await fn(*args, **kwargs)
                return result
            except BaseException as exc:
                exc_caught = exc
                raise
            finally:
                latency_ms = int((time.monotonic() - t0) * 1000)
                ok = exc_caught is None
                await _emit_async(
                    vigil_client,
                    fn_name=fn_name,
                    step_type=step_type,
                    args=args,
                    kwargs=kwargs,
                    result=result if result is not _SENTINEL else None,
                    ok=ok,
                    latency_ms=latency_ms,
                    error=exc_caught,
                )

        return wrapper  # type: ignore[return-value]

    return decorator


def make_agent(vigil_client: Any, name: str | None) -> Callable[[F], F]:
    """Return a decorator that wraps a sync function in a vigil Run."""

    def decorator(fn: F) -> F:
        agent_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            input_str: str | None = None
            if args:
                input_str = _safe_str(args[0])
            elif "input" in kwargs:
                input_str = _safe_str(kwargs["input"])

            with vigil_client.run(agent_name, input=input_str) as run:
                result = fn(*args, **kwargs)
                if result is not None:
                    run.set_output(_safe_str(result))
                return result

        return wrapper  # type: ignore[return-value]

    return decorator


def make_agent_async(vigil_client: Any, name: str | None) -> Callable[[F], F]:
    """Return a decorator that wraps an async function in a vigil AsyncRun."""

    def decorator(fn: F) -> F:
        agent_name = name or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            input_str: str | None = None
            if args:
                input_str = _safe_str(args[0])
            elif "input" in kwargs:
                input_str = _safe_str(kwargs["input"])

            async with vigil_client.run(agent_name, input=input_str) as run:
                result = await fn(*args, **kwargs)
                if result is not None:
                    run.set_output(_safe_str(result))
                return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ── emit helpers ──────────────────────────────────────────────────────────────

def _emit_sync(
    vigil_client: Any,
    *,
    fn_name: str,
    step_type: str,
    args: tuple,
    kwargs: dict,
    result: Any,
    ok: bool,
    latency_ms: int,
    error: BaseException | None,
) -> None:
    try:
        run = get_current_run()
        tool_input = _build_input(args, kwargs)
        tool_output = _build_output(result, error)

        if run is not None:
            if step_type == "tool_call":
                run.tool_call(
                    fn_name,
                    input=tool_input,
                    output=tool_output,
                    ok=ok,
                    latency_ms=latency_ms,
                )
                run.check_fingerprint(fn_name, tool_input)
            else:
                run.step(
                    step_type,
                    content=str(result) if result is not None else None,
                    metadata={"fn": fn_name, "latency_ms": latency_ms},
                )
        else:
            vigil_client.ingest_ai(
                model=fn_name,
                provider="observe",
                status="success" if ok else "error",
                latency_ms=latency_ms,
                error_message=str(error)[:500] if error else None,
                metadata={"input": tool_input, "output": tool_output},
            )
    except Exception as emit_err:
        warnings.warn(f"vigil observe emit failed: {emit_err}", stacklevel=3)


async def _emit_async(
    vigil_client: Any,
    *,
    fn_name: str,
    step_type: str,
    args: tuple,
    kwargs: dict,
    result: Any,
    ok: bool,
    latency_ms: int,
    error: BaseException | None,
) -> None:
    try:
        run = get_current_run()
        tool_input = _build_input(args, kwargs)
        tool_output = _build_output(result, error)

        if run is not None:
            if step_type == "tool_call":
                await run.tool_call(
                    fn_name,
                    input=tool_input,
                    output=tool_output,
                    ok=ok,
                    latency_ms=latency_ms,
                )
                run.check_fingerprint(fn_name, tool_input)
            else:
                await run.step(
                    step_type,
                    content=str(result) if result is not None else None,
                    metadata={"fn": fn_name, "latency_ms": latency_ms},
                )
        else:
            await vigil_client.ingest_ai(
                model=fn_name,
                provider="observe",
                status="success" if ok else "error",
                latency_ms=latency_ms,
                error_message=str(error)[:500] if error else None,
                metadata={"input": tool_input, "output": tool_output},
            )
    except Exception as emit_err:
        warnings.warn(f"vigil observe emit failed: {emit_err}", stacklevel=3)


# ── serialisation helpers ─────────────────────────────────────────────────────

def _build_input(args: tuple, kwargs: dict) -> dict:
    result: dict = {}
    if args:
        result["args"] = [_safe_str(a) for a in args]
    if kwargs:
        result.update({k: _safe_str(v) for k, v in kwargs.items()})
    return result


def _build_output(result: Any, error: BaseException | None) -> dict:
    if error is not None:
        return {"error": str(error)[:500]}
    if isinstance(result, dict):
        return result
    if result is None:
        return {}
    return {"value": _safe_str(result)}


def _safe_str(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return str(v)[:500]
    except Exception:
        return "<unserializable>"
