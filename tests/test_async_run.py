import asyncio

import httpx
import pytest

@pytest.mark.asyncio
async def test_async_run_lifecycle_completes_with_steps(async_client):
    from vigil import AsyncRun
    async with async_client.run("async-test-agent", input="hi") as run:
        assert isinstance(run, AsyncRun)
        await run.step("think", "I should respond")
        await run.tool_call("echo", input={"msg": "hi"}, output="hi", ok=True)
        await run.set_output("done")

@pytest.mark.asyncio
async def test_async_run_marks_failed_on_exception(async_client):
    class BoomError(Exception):
        pass

    with pytest.raises(BoomError):
        async with async_client.run("async-test-agent") as run:
            await run.step("think", "about to fail")
            raise BoomError("simulated")


@pytest.mark.asyncio
async def test_loop_detection_via_repeated_tool_call(async_client):
    async with async_client.run("async-looper") as run:
        for _ in range(3):
            await run.tool_call("search", input={"q": "same"}, output={"results": []})

    # agent_steps batched server-side; let the 500ms flush fire.
    await asyncio.sleep(0.7)

    async with httpx.AsyncClient() as h:
        resp = await h.get(
            f"{async_client.endpoint}/v1/agent/runs/{run.id}/loops",
            params={"at": run.timestamp},
            headers={"Authorization": f"Bearer {async_client.api_key}"},
        )
    body = resp.json()["data"]
    assert body is not None, "expected loops payload, got null"
    assert any(hit["hits"] >= 2 for hit in body), f"expected loop hits: {body}"
