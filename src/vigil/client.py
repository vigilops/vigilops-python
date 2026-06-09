import httpx

class Vigil :
    def __init__(self, api_key:str, endpoint:str = "http://localhost:8080") -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self._client = httpx.Client(
            base_url=endpoint,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0),
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

    def health(self) -> dict:
        response = self._client.get("/v1/health")
        response.raise_for_status()
        return response.json()["data"]
        
    def close(self) -> None:
        self._client.close()

    def ingest_ai(
        self,
        *,                       # force keyword-only args
        model: str,
        status: str,
        provider: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
        latency_ms: int | None = None,
    ) -> dict:
        payload = {"model": model, "status": status}
       
        if provider is not None:
            payload["provider"] = provider
        if input_tokens is not None:
            payload["input_tokens"] = input_tokens
        if output_tokens is not None:
            payload["output_tokens"] = output_tokens
        if total_tokens is not None:
            payload["total_tokens"] = total_tokens
        if cost_usd is not None:
            payload["cost_usd"] = cost_usd
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        
        resp = self._client.post("/v1/ingest/ai", json=payload)
        if resp.status_code != 201:
            raise RuntimeError(f"ingest_ai failed: {resp.status_code} {resp.text}")
        
        return resp.json()["data"]
    
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()