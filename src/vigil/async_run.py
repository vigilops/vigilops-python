from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .async_client import AsyncVigil


class AsyncRun:
    """An open agent run. Created via AsyncVigil.run() and entered with `with`.

    The context manager calls POST /v1/ingest/agent/runs on entry and
    POST /v1/ingest/agent/runs/{id}/finish on exit. step() and
    tool_call() each send POST /v1/ingest/agent/steps in between.

    The run is marked failed automatically if the `with` block raises.
    """

    def __init__(
        self,
        *,
        client: AsyncVigil,
        agent_name: str,
        input: str | None = None,
    ) -> None:
        self._client = client
        self._agent_name = agent_name
        self._input = input
        self._id: str | None = None
        self._timestamp: str | None = None
        self._step_index = 0
        self._total_tokens = 0
        self._output: str | None = None

    @property
    def id(self) -> str:
        if self._id is None:
            raise RuntimeError("AsyncRun not started — use `with client.run(...) as run:`")
        return self._id

    @property
    def timestamp(self) -> str:
        if self._timestamp is None:
            raise RuntimeError("AsyncRun not started — use `with client.run(...) as run:`")
        return self._timestamp

    async def __aenter__(self) -> "AsyncRun":
        data = await self._client.ingest_agent_run_start(
            agent_name=self._agent_name,
            input=self._input,
        )
        self._id = data["id"]
        self._timestamp = data["timestamp"]
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._id is None or self._timestamp is None:
            return
        status = "completed" if exc_type is None else "failed"
        reason = "clean" if exc_type is None else "error"
        await self._client.ingest_agent_run_finish(
            self._id,
            timestamp=self._timestamp,
            status=status,
            termination_reason=reason,
            total_steps=self._step_index,
            total_tokens=self._total_tokens,
            output=self._output,
        )

    async def step(
        self,
        step_type: str,
        content: str | None = None,
        *,
        tokens: int | None = None,
    ) -> None:
        self._step_index += 1
        if tokens:
            self._total_tokens += tokens
        await self._client.ingest_agent_step(
            agent_run_id=self.id,
            step_index=self._step_index,
            step_type=step_type,
            content=content,
            tokens=tokens,
        )

    async def tool_call(
        self,
        tool_name: str,
        *,
        input: dict,
        output: Any,
        ok: bool = True,
        tokens: int | None = None,
    ) -> None:
        self._step_index += 1
        if tokens:
            self._total_tokens += tokens
        tool_output: dict | None = output if isinstance(output, dict) else {"value": str(output)}
        await self._client.ingest_agent_step(
            agent_run_id=self.id,
            step_index=self._step_index,
            step_type="tool_call",
            tool_name=tool_name,
            tool_input=input,
            tool_output=tool_output,
            tool_success=ok,
            tokens=tokens,
        )

    async def set_output(self, output: str) -> None:
        self._output = output
