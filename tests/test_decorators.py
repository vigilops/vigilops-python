import os
import unittest.mock as mock

import pytest
import pytest_asyncio

from vigil import Vigil, AsyncVigil
from vigil._context import get_current_run


@pytest.fixture
def client():
    api_key = os.getenv("VIGIL_API_KEY")
    if not api_key:
        pytest.skip("VIGIL_API_KEY not set — run `make seed` and export it")
    with Vigil(api_key=api_key, endpoint=os.getenv("VIGIL_ENDPOINT", "http://localhost:8080")) as c:
        yield c


@pytest_asyncio.fixture
async def async_client():
    api_key = os.getenv("VIGIL_API_KEY")
    if not api_key:
        pytest.skip("VIGIL_API_KEY not set")
    async with AsyncVigil(api_key=api_key, endpoint=os.getenv("VIGIL_ENDPOINT", "http://localhost:8080")) as c:
        yield c


# ── @observe sync ─────────────────────────────────────────────────────────────

def test_observe_returns_function_value(client):
    @client.observe
    def add(x: int, y: int) -> int:
        return x + y

    assert add(2, 3) == 5


def test_observe_noparen_form(client):
    @client.observe
    def noop() -> str:
        return "hi"

    assert noop() == "hi"


def test_observe_paren_form(client):
    @client.observe(name="my_tool", step_type="tool_call")
    def noop() -> str:
        return "hi"

    assert noop() == "hi"


def test_observe_emits_tool_call_inside_run(client):
    @client.observe
    def fake_search(q: str) -> dict:
        return {"results": ["a"]}

    with client.run("test-agent", input="test") as run:
        with mock.patch.object(run, "tool_call", wraps=run.tool_call) as mock_tc:
            result = fake_search(q="hello")
            assert result == {"results": ["a"]}
            mock_tc.assert_called_once()
            args, kwargs = mock_tc.call_args
            assert args[0] == "fake_search"
            assert kwargs["ok"] is True
            assert "latency_ms" in kwargs


def test_observe_emits_ai_trace_outside_run(client):
    @client.observe(name="standalone_fn")
    def standalone(x: int) -> int:
        return x * 2

    with mock.patch.object(client, "ingest_ai", wraps=client.ingest_ai) as mock_ai:
        result = standalone(5)
        assert result == 10
        mock_ai.assert_called_once()
        kw = mock_ai.call_args[1]
        assert kw["model"] == "standalone_fn"
        assert kw["status"] == "success"
        assert "latency_ms" in kw


def test_observe_fail_soft_on_server_down(client):
    """observe() must not crash user code even when server is unreachable."""
    @client.observe
    def my_tool(x: int) -> int:
        return x + 1

    with mock.patch.object(client, "ingest_ai", side_effect=Exception("network down")):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = my_tool(4)
            assert result == 5
            assert any("vigil observe emit failed" in str(warning.message) for warning in w)


# ── @observe async ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_observe_async_returns_value(async_client):
    @async_client.observe
    async def async_add(x: int, y: int) -> int:
        return x + y

    assert await async_add(3, 4) == 7


@pytest.mark.asyncio
async def test_observe_async_emits_tool_call_inside_run(async_client):
    @async_client.observe
    async def async_tool(q: str) -> dict:
        return {"result": q}

    async with async_client.run("async-test-agent", input="q") as run:
        with mock.patch.object(run, "tool_call", wraps=run.tool_call) as mock_tc:
            result = await async_tool(q="test")
            assert result == {"result": "test"}
            mock_tc.assert_called_once()
            args, kwargs = mock_tc.call_args
            assert args[0] == "async_tool"
            assert kwargs["ok"] is True


# ── @agent sync ───────────────────────────────────────────────────────────────

def test_agent_returns_function_value(client):
    @client.agent(name="test-sync-agent")
    def my_agent(task: str) -> str:
        return f"done: {task}"

    assert my_agent("hello") == "done: hello"


def test_agent_noparen_form(client):
    @client.agent
    def my_agent(task: str) -> str:
        return "ok"

    assert my_agent("x") == "ok"


def test_agent_sets_current_run(client):
    captured: list = []

    @client.observe
    def inner_tool(x: int) -> int:
        captured.append(get_current_run())
        return x + 1

    @client.agent(name="test-context-agent")
    def my_agent(task: str) -> str:
        inner_tool(1)
        return "ok"

    my_agent("task")
    assert len(captured) == 1
    assert captured[0] is not None


def test_agent_observe_tool_call_linked(client):
    """@observe inside @agent must complete without error."""
    @client.observe
    def search(q: str) -> dict:
        return {"r": q}

    @client.agent(name="linked-agent")
    def my_agent(task: str) -> str:
        search(q=task)
        return "done"

    result = my_agent("find me")
    assert result == "done"


# ── @agent async ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_async_returns_value(async_client):
    @async_client.agent(name="test-async-agent")
    async def my_agent(task: str) -> str:
        return f"async done: {task}"

    assert await my_agent("hello") == "async done: hello"


@pytest.mark.asyncio
async def test_agent_async_sets_current_run(async_client):
    captured: list = []

    @async_client.observe
    async def inner_tool(x: int) -> int:
        captured.append(get_current_run())
        return x + 1

    @async_client.agent(name="test-async-context-agent")
    async def my_agent(task: str) -> str:
        await inner_tool(1)
        return "ok"

    await my_agent("task")
    assert len(captured) == 1
    assert captured[0] is not None
