import asyncio
import os

import pytest

from keelwave._exceptions import (
    KeelwaveAuthError,
    KeelwaveTransportError,
    KeelwaveValidationError,
)
from keelwave.async_client import AsyncKeelwave


@pytest.mark.asyncio
async def test_async_health_returns_ok(async_client):
    body = await async_client.health()
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_async_ingest_ai_minimum_payload(async_client):
    body = await async_client.ingest_ai(model="claude-opus-4-7", status="success")
    assert "id" in body
    assert "timestamp" in body


@pytest.mark.asyncio
async def test_ingest_ai_conncurent_calls_parallelize(async_client):
    tasks = [
        async_client.ingest_ai(model=f"model-{i}", status="success") for i in range(10)
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == 10
    for body in results:
        assert "id" in body


@pytest.mark.asyncio
async def test_agent_lifecycle_via_low_level_methods(async_client):
    run = await async_client.ingest_agent_run_start(agent_name="async-low", input="x")
    await async_client.ingest_agent_step(
        agent_run_id=run["id"],
        step_index=1,
        step_type="think",
        content="hello",
    )
    await async_client.ingest_agent_run_finish(
        run["id"],
        timestamp=run["timestamp"],
        status="completed",
        termination_reason="clean",
        total_steps=1,
        total_tokens=0,
    )


@pytest.mark.asyncio
async def test_ingest_ai_raises_auth_error_on_bad_key():
    endpoint = os.getenv("KEELWAVE_ENDPOINT", "http://localhost:8080")
    async with AsyncKeelwave(api_key="vop_obviously_wrong", endpoint=endpoint) as c:
        with pytest.raises(KeelwaveAuthError):
            await c.ingest_ai(model="m", status="success")


@pytest.mark.asyncio
async def test_ingest_ai_raises_validation_error_on_bad_status(async_client):
    with pytest.raises(KeelwaveValidationError):
        await async_client.ingest_ai(model="m", status="not-a-valid-status")


@pytest.mark.asyncio
async def test_health_raises_transport_error_on_unreachable_host():
    async with AsyncKeelwave(api_key="vop_x", endpoint="http://127.0.0.1:1") as c:
        with pytest.raises(KeelwaveTransportError):
            await c.health()
