import httpx

from ._client import raise_for_status
from ._exceptions import VigilTransportError

class Vigil:
    def __init__(self, api_key: str, endpoint: str = "http://localhost:8080") -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self._client = httpx.Client(
            base_url=endpoint,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def health(self) -> dict:
        try:
            resp = self._client.get("/v1/health")
        except httpx.RequestError as e:
            raise VigilTransportError(str(e)) from e
        raise_for_status(resp)
        return resp.json()["data"]

    def close(self) -> None:
        self._client.close()

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
    ) -> dict:
        payload: dict = {"model": model, "status": status}
        
        for k, v in (
            ("provider", provider),
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
            ("total_tokens", total_tokens),
            ("cost_usd", cost_usd),
            ("latency_ms", latency_ms),
        ):
            if v is not None:
                payload[k] = v

        try:
            resp = self._client.post("/v1/ingest/ai", json=payload)
        except httpx.RequestError as e:
            raise VigilTransportError(str(e)) from e
        raise_for_status(resp)
        return resp.json()["data"]

    def ingest_agent_run_start(
        self,
        *,
        agent_name: str,
        input: str | None = None,
    ) -> dict:
        payload: dict = {"agent_name": agent_name}
        if input is not None:
            payload["input"] = input
        try:
            resp = self._client.post("/v1/ingest/agent/runs", json=payload)
        except httpx.RequestError as e:
            raise VigilTransportError(str(e)) from e
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
        output: str | None = None,
    ) -> None:
        payload: dict = {
            "timestamp": timestamp,
            "status": status,
            "total_steps": total_steps,
            "total_tokens": total_tokens,
        }
        if termination_reason is not None:
            payload["termination_reason"] = termination_reason
        if output is not None:
            payload["output"] = output
        try:
            resp = self._client.post(
                f"/v1/ingest/agent/runs/{run_id}/finish",
                json=payload,
            )
        except httpx.RequestError as e:
            raise VigilTransportError(str(e)) from e
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
        tokens: int | None = None,
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
            ("tokens", tokens),
        ):
            if v is not None:
                payload[k] = v
        try:
            resp = self._client.post("/v1/ingest/agent/steps", json=payload)
        except httpx.RequestError as e:
            raise VigilTransportError(str(e)) from e
        raise_for_status(resp)
        return resp.json()["data"]

    def run(self, agent_name: str, *, input: str | None = None) -> "Run":
        from .run import Run

        return Run(client=self, agent_name=agent_name, input=input)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()