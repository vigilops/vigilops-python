from __future__ import annotations

import time
from contextvars import Token
from typing import TYPE_CHECKING, Any

from ._context import reset_current_run, set_current_run

if TYPE_CHECKING:
    from .client import Vigil


class Run:
    """An open agent run. Created via Vigil.run() and entered with `with`.

    The context manager POSTs to /v1/ingest/agent/runs on entry, accumulates
    step / token / cost totals across the with-block, and POSTs to
    /v1/ingest/agent/runs/{id}/finish on exit. step() and tool_call() each
    send POST /v1/ingest/agent/steps in between.

    On exception, status is recorded as "failed" with termination_reason="error".
    """

    def __init__(
        self,
        *,
        client: Vigil,
        agent_name: str,
        input: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._client = client
        self._agent_name = agent_name
        self._input = input
        self._metadata = metadata
        self._id: str | None = None
        self._timestamp: str | None = None
        self._t_start: float | None = None
        self._step_index = 0
        self._total_tokens = 0
        self._total_cost_usd = 0.0
        self._output: str | None = None
        self._loop_detected = False
        self._loop_step_index: int | None = None
        self._ctx_token: Token | None = None

    @property
    def id(self) -> str:
        if self._id is None:
            raise RuntimeError("Run not started — use `with client.run(...) as run:`")
        return self._id

    @property
    def timestamp(self) -> str:
        if self._timestamp is None:
            raise RuntimeError("Run not started — use `with client.run(...) as run:`")
        return self._timestamp

    def __enter__(self) -> Run:
        self._t_start = time.monotonic()
        data = self._client.ingest_agent_run_start(
            agent_name=self._agent_name,
            input=self._input,
            metadata=self._metadata,
        )
        self._id = data["id"]
        self._timestamp = data["timestamp"]
        self._ctx_token = set_current_run(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._ctx_token is not None:
            reset_current_run(self._ctx_token)
            self._ctx_token = None
        if self._id is None or self._timestamp is None:
            return
        status = "completed" if exc_type is None else "failed"
        reason = "clean" if exc_type is None else "error"
        duration_ms = (
            int((time.monotonic() - self._t_start) * 1000)
            if self._t_start is not None
            else None
        )
        self._client.ingest_agent_run_finish(
            self._id,
            timestamp=self._timestamp,
            status=status,
            termination_reason=reason,
            total_steps=self._step_index,
            total_tokens=self._total_tokens,
            total_cost_usd=self._total_cost_usd or None,
            duration_ms=duration_ms,
            loop_detected=self._loop_detected,
            loop_step_index=self._loop_step_index,
            output=self._output,
        )

    def step(
        self,
        step_type: str,
        content: str | None = None,
        *,
        tokens: int | None = None,
        cost_usd: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._step_index += 1
        if tokens:
            self._total_tokens += tokens
        if cost_usd:
            self._total_cost_usd += cost_usd
        self._client.ingest_agent_step(
            agent_run_id=self.id,
            step_index=self._step_index,
            step_type=step_type,
            content=content,
            tokens=tokens,
            cost_usd=cost_usd,
            metadata=metadata,
        )

    def tool_call(
        self,
        tool_name: str,
        *,
        input: dict,
        output: Any,
        ok: bool = True,
        tokens: int | None = None,
        cost_usd: float | None = None,
        latency_ms: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._step_index += 1
        if tokens:
            self._total_tokens += tokens
        if cost_usd:
            self._total_cost_usd += cost_usd
        tool_output: dict | None = output if isinstance(output, dict) else {"value": str(output)}
        self._client.ingest_agent_step(
            agent_run_id=self.id,
            step_index=self._step_index,
            step_type="tool_call",
            tool_name=tool_name,
            tool_input=input,
            tool_output=tool_output,
            tool_success=ok,
            tool_latency_ms=latency_ms,
            tokens=tokens,
            cost_usd=cost_usd,
            metadata=metadata,
        )

    def set_output(self, output: str) -> None:
        self._output = output

    def mark_loop(self, step_index: int | None = None) -> None:
        """Mark this run as a loop. step_index = the first repeated step
        if known. Sets loop_detected=true on finish."""
        self._loop_detected = True
        if step_index is not None:
            self._loop_step_index = step_index
