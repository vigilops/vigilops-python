from __future__ import annotations

import hashlib
import json
import time
from contextvars import Token
from types import TracebackType
from typing import TYPE_CHECKING, Any

from ._context import reset_current_run, set_current_run

if TYPE_CHECKING:
    from .async_client import AsyncVigil


class AsyncRun:
    """Async sibling of Run. Same surface, awaited."""

    def __init__(
        self,
        *,
        client: AsyncVigil,
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
        self._seen_fingerprints: dict[str, int] = {}  # fingerprint → first step_index

    @property
    def id(self) -> str:
        if self._id is None:
            raise RuntimeError(
                "AsyncRun not started — use `async with client.run(...) as run:`"
            )
        return self._id

    @property
    def timestamp(self) -> str:
        if self._timestamp is None:
            raise RuntimeError(
                "AsyncRun not started — use `async with client.run(...) as run:`"
            )
        return self._timestamp

    async def __aenter__(self) -> AsyncRun:
        self._t_start = time.monotonic()
        data = await self._client.ingest_agent_run_start(
            agent_name=self._agent_name,
            input=self._input,
            metadata=self._metadata,
        )
        self._id = data["id"]
        self._timestamp = data["timestamp"]
        self._ctx_token = set_current_run(self)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
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
        await self._client.ingest_agent_run_finish(
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

    async def step(
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
        await self._client.ingest_agent_step(
            agent_run_id=self.id,
            step_index=self._step_index,
            step_type=step_type,
            content=content,
            tokens=tokens,
            cost_usd=cost_usd,
            metadata=metadata,
        )

    async def tool_call(
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
        tool_output: dict | None = (
            output if isinstance(output, dict) else {"value": str(output)}
        )
        await self._client.ingest_agent_step(
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

    def check_fingerprint(self, tool_name: str, tool_input: dict) -> None:
        """Compute SHA-256 of tool_name+input, mark loop on first duplicate."""
        fp = hashlib.sha256(
            (
                tool_name
                + json.dumps(tool_input, sort_keys=True, separators=(",", ":"))
            ).encode()
        ).hexdigest()
        if fp in self._seen_fingerprints:
            if not self._loop_detected:
                self.mark_loop(step_index=self._seen_fingerprints[fp])
        else:
            self._seen_fingerprints[fp] = self._step_index

    def set_output(self, output: str) -> None:
        self._output = output

    def mark_loop(self, step_index: int | None = None) -> None:
        self._loop_detected = True
        if step_index is not None:
            self._loop_step_index = step_index
