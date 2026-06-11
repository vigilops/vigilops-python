import httpx

from .async_run import AsyncRun
from ._client import raise_for_status
from ._exceptions import VigilTransportError


class AsyncVigil:
    def __init__(self, api_key: str, endpoint: str = "http://localhost:8080") -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self._client = httpx.AsyncClient(
            base_url=endpoint,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    async def health(self) -> dict:
        try:
            resp = await self._client.get("/v1/health")
        except httpx.RequestError as e:
            raise VigilTransportError(str(e)) from e
        raise_for_status(resp)
        return resp.json()["data"]

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncVigil":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    async def ingest_ai(
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
    ) -> dict:
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
            resp = await self._client.post("/v1/ingest/ai", json=payload)
        except httpx.RequestError as e:
            raise VigilTransportError(str(e)) from e
        raise_for_status(resp)
        return resp.json()["data"]
    
    async def ingest_agent_run_start(
        self,
        *,
        agent_name: str,
        input: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        payload: dict = {"agent_name": agent_name}
        
        if input is not None:
            payload["input"] = input
        if metadata is not None:
            payload["metadata"] = metadata

        try:
            resp = await self._client.post("/v1/ingest/agent/runs", json=payload)
        except httpx.RequestError as e:
            raise VigilTransportError(str(e)) from e
        raise_for_status(resp)
        return resp.json()["data"]

    async def ingest_agent_run_finish(
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
            resp = await self._client.post(f"/v1/ingest/agent/runs/{run_id}/finish", json=payload)
        except httpx.RequestError as e:
            raise VigilTransportError(str(e)) from e
        raise_for_status(resp)

    async def ingest_agent_step(
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
    ) -> dict:
        payload: dict = {
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
            resp = await self._client.post("/v1/ingest/agent/steps", json=payload)
        except httpx.RequestError as e:
            raise VigilTransportError(str(e)) from e
        raise_for_status(resp)
        return resp.json()["data"]
    
    def run(
        self,
        agent_name: str,
        *,
        input: str | None = None,
        metadata: dict | None = None,
    ) -> "AsyncRun":
        return AsyncRun(client=self, agent_name=agent_name, input=input, metadata=metadata)

    def wrap_anthropic(self, client, *, provider: str = "anthropic"):
        """Async sibling of Vigil.wrap_anthropic. Wraps an
        anthropic.AsyncAnthropic client — every `await
        client.messages.create(...)` records to ai_traces and auto-links
        to the current AsyncRun via ContextVar.
        """
        from .adapters.anthropic import wrap_async_client
        return wrap_async_client(client, self, provider=provider)

    def wrap_openai(self, client, *, provider: str = "openai"):
        """Async sibling of Vigil.wrap_openai. Wraps an openai.AsyncOpenAI
        client — every `await client.chat.completions.create(...)` records
        to ai_traces and auto-links to the current AsyncRun.
        """
        from .adapters.openai import wrap_async_client
        return wrap_async_client(client, self, provider=provider)
