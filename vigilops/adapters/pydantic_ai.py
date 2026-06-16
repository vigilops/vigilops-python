"""pydantic-ai integration for vigil.

Usage — wrap an agent run so vigil records the full trace::

    from vigilops import Vigil
    from vigilops.adapters.pydantic_ai import instrument

    client = Vigil(api_key=...)
    agent = Agent("openai:gpt-4o", ...)

    result = await instrument(client, agent, "find bugs in this file", deps=deps)
    # → opens Run, emits think/tool_call/tool_result steps, closes Run

Or use the low-level helper when you already have a Run open::

    with client.run("my-agent") as run:
        result = await run_with_steps(run, agent, "task", deps=deps)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, AsyncIterable

if TYPE_CHECKING:
    from pydantic_ai.agent import Agent
    from pydantic_ai.tools import RunContext

from .._context import get_current_run
from ..client import Vigil
from ..run import Run
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
)


async def run_with_steps(
    run: Run,
    agent: "Agent[Any, Any]",
    user_prompt: str | None = None,
    *,
    deps: Any = None,
    **run_kwargs: Any,
) -> Any:
    """Run a pydantic-ai agent inside an already-open vigil Run.

    Emits tool_call + tool_result steps for each tool invocation.
    Stores token usage on the run when available.
    Returns the AgentRunResult.
    """
    

    pending_calls: dict[str, tuple[str, dict[str, Any], int]] = {}

    async def _handler(
        ctx: "RunContext[Any]",
        stream: AsyncIterable[Any],
    ) -> None:
        async for event in stream:
            if isinstance(event, FunctionToolCallEvent):
                tool_name = event.part.tool_name
                try:
                    args: dict[str, Any] = (
                        event.part.args
                        if isinstance(event.part.args, dict)
                        else {}
                    )
                except Exception:
                    args = {}
                pending_calls[event.tool_call_id] = (tool_name, args, run._step_index + 1)

            elif isinstance(event, FunctionToolResultEvent):
                call_id = event.tool_call_id
                entry = pending_calls.pop(call_id, None)
                if entry is None:
                    continue
                tool_name, tool_input, _step = entry

                part = event.part
                try:
                    output_str: str | None = getattr(part, "content", None)
                    if output_str is None:
                        output_str = str(getattr(part, "return_value", ""))
                except Exception:
                    output_str = ""

                ok = not isinstance(part, type) and getattr(part, "is_error", False) is False
                try:
                    output = {"result": output_str[:2000]} if output_str else {}
                    run.tool_call(tool_name, input=tool_input, output=output, ok=ok)
                except Exception as e:
                    warnings.warn(f"vigil pydantic-ai step emit failed: {e}", stacklevel=2)

    result = await agent.run(
        user_prompt,
        deps=deps,
        event_stream_handler=_handler,
        **run_kwargs,
    )

    try:
        usage = result.usage
        total = (usage.input_tokens or 0) + (usage.output_tokens or 0)
        if total:
            run._total_tokens += total
    except Exception:
        pass

    return result


async def instrument(
    vigil_client: Vigil,
    agent: "Agent[Any, Any]",
    user_prompt: str | None = None,
    *,
    agent_name: str | None = None,
    deps: Any = None,
    **run_kwargs: Any,
) -> Any:
    """Run a pydantic-ai agent, opening a vigil Run automatically.

    If there's already an active run (from @agent decorator), reuses it.
    Otherwise opens a new one named after the agent (or agent_name).

    Returns the AgentRunResult.
    """
    existing = get_current_run()

    if existing is not None:
        return await run_with_steps(existing, agent, user_prompt, deps=deps, **run_kwargs)

    name = agent_name or getattr(agent, "name", None) or "pydantic-ai-agent"
    with vigil_client.run(name, input=user_prompt) as run:
        return await run_with_steps(run, agent, user_prompt, deps=deps, **run_kwargs)
