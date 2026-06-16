"""Tests for the pydantic-ai adapter.

Uses pydantic-ai's built-in TestModel so no real LLM calls are made.
"""
import os
import pytest
import pytest_asyncio

from vigilops import Vigil
from vigilops.adapters.pydantic_ai import instrument, run_with_steps


@pytest.fixture
def client(_sync_project):
    with Vigil(api_key=_sync_project, endpoint=os.getenv("VIGILOPS_ENDPOINT", "http://localhost:8080")) as c:
        yield c


@pytest.mark.asyncio
async def test_instrument_opens_run_and_returns_result(client):
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel(), name="test-agent")
    result = await instrument(client, agent, "say hi")
    assert result is not None
    assert result.output is not None


@pytest.mark.asyncio
async def test_instrument_with_tool_emits_steps(client):
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    calls: list[str] = []

    agent = Agent(TestModel(), name="tool-agent")

    @agent.tool_plain
    def greet(name: str) -> str:
        calls.append(name)
        return f"Hello, {name}!"

    result = await instrument(client, agent, "greet Alice")
    assert result is not None


@pytest.mark.asyncio
async def test_run_with_steps_reuses_existing_run(client):
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel(), name="reuse-agent")

    with client.run("outer-run", input="task") as run:
        result = await run_with_steps(run, agent, "inner task")
        assert result is not None
        run_id = run.id

    assert run_id  # run was properly opened


@pytest.mark.asyncio
async def test_instrument_reuses_ambient_run(client):
    """When called inside @agent, instrument() reuses the ambient run."""
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    inner_agent = Agent(TestModel(), name="inner")

    @client.agent(name="outer-agent")
    async def outer(task: str) -> str:
        result = await instrument(client, inner_agent, task)
        return str(result.output)

    result = await outer("test task")
    assert result is not None
