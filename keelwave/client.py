from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from .adapters.anthropic import _SyncAnthropicProxy
    from .adapters.openai import _SyncOpenAIProxy
    from .decorators import _Span

import httpx

from ._client import raise_for_status
from ._exceptions import KeelwaveTransportError
from .run import Run

_F = TypeVar("_F", bound=Callable[..., Any])

class Keelwave:
    def __init__(self, api_key: str, endpoint: str = "http://localhost:8080") -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self._client = httpx.Client(
            base_url=endpoint,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def health(self) -> Any:
        try:
            resp = self._client.get("/v1/health")
        except httpx.RequestError as e:
            raise KeelwaveTransportError(str(e)) from e
        raise_for_status(resp)
        return resp.json()["data"]

    def close(self) -> None:
        self._client.close()
    
    def __enter__(self) -> "Keelwave":
        return self

    def __exit__(self, _exc_type: type[BaseException] | None, _exc: BaseException | None, _tb: TracebackType | None) -> None:
        self.close()

    def ingest_ai(
        self,
        *,
        model: str,
        status: str,
        provider: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
        latency_ms: int | None = None,
        error_message: str | None = None,
        request_id: str | None = None,
        agent_run_id: str | None = None,
        metadata: dict | None = None,
    ) -> Any:
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        payload: dict = {"model": model, "status": status}

        for k, v in (
            ("provider", provider),
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
            ("total_tokens", total_tokens),
            ("cost_usd", cost_usd),
            ("latency_ms", latency_ms),
            ("error_message", error_message),
            ("request_id", request_id),
            ("agent_run_id", agent_run_id),
            ("metadata", metadata),
        ):
            if v is not None:
                payload[k] = v

        try:
            resp = self._client.post("/v1/ingest/ai", json=payload)
        except httpx.RequestError as e:
            raise KeelwaveTransportError(str(e)) from e
        raise_for_status(resp)
        return resp.json()["data"]

    def ingest_agent_run_start(
        self,
        *,
        agent_name: str,
        input: str | None = None,
        metadata: dict | None = None,
    ) -> Any:
        payload: dict[str, Any] = {"agent_name": agent_name}
        if input is not None:
            payload["input"] = input
        if metadata is not None:
            payload["metadata"] = metadata

        try:
            resp = self._client.post("/v1/ingest/agent/runs", json=payload)
        except httpx.RequestError as e:
            raise KeelwaveTransportError(str(e)) from e
        raise_for_status(resp)
        return resp.json()["data"]

    def ingest_agent_run_finish(
        self,
        run_id: str,
        *,
        timestamp: str,
        status: str,
        termination_reason: str | None = None,
        total_steps: int = 0,
        total_tokens: int = 0,
        total_cost_usd: float | None = None,
        duration_ms: int | None = None,
        loop_detected: bool = False,
        loop_step_index: int | None = None,
        output: str | None = None,
    ) -> None:
        payload: dict = {
            "timestamp": timestamp,
            "status": status,
            "total_steps": total_steps,
            "total_tokens": total_tokens,
            "loop_detected": loop_detected,
        }
        for k, v in (
            ("termination_reason", termination_reason),
            ("total_cost_usd", total_cost_usd),
            ("duration_ms", duration_ms),
            ("loop_step_index", loop_step_index),
            ("output", output),
        ):
            if v is not None:
                payload[k] = v
        
        try:
            resp = self._client.post(
                f"/v1/ingest/agent/runs/{run_id}/finish",
                json=payload,
            )
        except httpx.RequestError as e:
            raise KeelwaveTransportError(str(e)) from e
        raise_for_status(resp)

    def ingest_agent_step(
        self,
        *,
        agent_run_id: str,
        step_index: int,
        step_type: str,
        content: str | None = None,
        tool_name: str | None = None,
        tool_input: dict | None = None,
        tool_output: dict | None = None,
        tool_success: bool | None = None,
        tool_latency_ms: int | None = None,
        tokens: int | None = None,
        cost_usd: float | None = None,
        metadata: dict | None = None,
    ) -> Any:
        payload: dict[str, Any] = {
            "agent_run_id": agent_run_id,
            "step_index": step_index,
            "step_type": step_type,
        }

        for k, v in (
            ("content", content),
            ("tool_name", tool_name),
            ("tool_input", tool_input),
            ("tool_output", tool_output),
            ("tool_success", tool_success),
            ("tool_latency_ms", tool_latency_ms),
            ("tokens", tokens),
            ("cost_usd", cost_usd),
            ("metadata", metadata),
        ):
            if v is not None:
                payload[k] = v

        try:
            resp = self._client.post("/v1/ingest/agent/steps", json=payload)
        except httpx.RequestError as e:
            raise KeelwaveTransportError(str(e)) from e
        raise_for_status(resp)
        return resp.json()["data"]

    def run(
        self,
        agent_name: str,
        *,
        input: str | None = None,
        metadata: dict | None = None,
    ) -> "Run":
        return Run(client=self, agent_name=agent_name, input=input, metadata=metadata)

    def wrap_anthropic(self, client: Any, *, provider: str = "anthropic") -> "_SyncAnthropicProxy":
        """Return a drop-in proxy that auto-records every messages.create
        call to ai_traces. If used inside `with keelwave.run(...) as run:`,
        ai_traces.agent_run_id is set automatically via ContextVar.

        Provider defaults to "anthropic". Override when hitting an
        Anthropic-compatible endpoint (e.g. "deepseek", "openrouter").
        """
        from .adapters.anthropic import wrap_client
        return wrap_client(client, self, provider=provider)

    def wrap_openai(self, client: Any, *, provider: str = "openai") -> "_SyncOpenAIProxy":
        """Drop-in proxy for the OpenAI sync client. Every
        chat.completions.create call records to ai_traces and
        auto-links to the active Run via ContextVar.

        Provider defaults to "openai". Override for OpenAI-compatible
        endpoints (e.g. "groq", "together", "azure", "openrouter").
        """
        from .adapters.openai import wrap_client
        return wrap_client(client, self, provider=provider)

    def span(self, step_type: str = "span", *, name: str | None = None) -> "_Span":
        """Sync context manager that emits a custom step on the active run.

        Usage::

            with client.span("retrieval", name="fetch_docs") as span:
                docs = fetch(query)
                span.set(content=f"fetched {len(docs)} docs")
        """
        from .decorators import _Span
        return _Span(self, step_type=step_type, name=name)

    def observe(self, fn: _F | None = None, *, name: str | None = None, step_type: str = "tool_call") -> _F | Callable[[_F], _F]:
        """Decorator that instruments a sync function.

        Inside an active run: emits a tool_call step (or named step type).
        Outside a run: emits a standalone ai_trace row.

        Usage::

            @client.observe
            def web_search(q: str) -> dict: ...

            @client.observe(name="search", step_type="tool_call")
            def web_search(q: str) -> dict: ...
        """
        from .decorators import make_observe
        dec = make_observe(self, name=name, step_type=step_type)
        if fn is not None:
            return dec(fn)
        return dec

    def agent(self, fn: _F | None = None, *, name: str | None = None) -> _F | Callable[[_F], _F]:
        """Decorator that wraps a sync function in a keelwave Run.

        Opens a Run on call, sets _current_run ContextVar so nested
        @observe calls auto-link, closes the Run on return or exception.

        Usage::

            @client.agent
            def my_agent(task: str) -> str: ...

            @client.agent(name="research-agent")
            def my_agent(task: str) -> str: ...
        """
        from .decorators import make_agent
        dec = make_agent(self, name=name)
        if fn is not None:
            return dec(fn)
        return dec
