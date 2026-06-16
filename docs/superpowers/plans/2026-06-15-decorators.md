# Decorators (`@observe` + `@agent`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `@vigil_client.observe` and `@vigil_client.agent` decorators — sync + async — to the vigil Python SDK.

**Architecture:** A single new `src/vigil/decorators.py` module holds the two decorator factories (`make_observe`, `make_agent`). Both `Vigil` and `AsyncVigil` grow `.observe()` and `.agent()` methods that bind `self` to those factories and return the ready-to-use decorator. `@observe` reads the active run from the `_current_run` ContextVar; if none is active it emits a standalone `ai_trace`. `@agent` opens/closes a `Run` or `AsyncRun` around the decorated function's execution and sets `_current_run` so nested `@observe` calls auto-link.

**Tech Stack:** Python 3.10+, `functools.wraps`, `inspect.iscoroutinefunction`, existing `vigil._context`, `vigil.run.Run`, `vigil.async_run.AsyncRun`, `pytest` + `pytest-asyncio` for tests.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| **Create** | `src/vigil/decorators.py` | `make_observe()` + `make_agent()` factories, sync + async variants |
| **Modify** | `src/vigil/client.py` | Add `.observe()` + `.agent()` methods to `Vigil` |
| **Modify** | `src/vigil/async_client.py` | Add `.observe()` + `.agent()` methods to `AsyncVigil` |
| **Modify** | `src/vigil/__init__.py` | No change needed — decorators accessed via client instance |
| **Create** | `tests/test_decorators.py` | Integration tests for both decorators, sync + async |

---

## Task 1: Create `decorators.py` — `make_observe` sync

**Files:**
- Create: `src/vigil/decorators.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decorators.py
import os
import pytest
from vigil import Vigil

@pytest.fixture
def client():
    api_key = os.getenv("VIGIL_API_KEY")
    if not api_key:
        pytest.skip("VIGIL_API_KEY not set")
    with Vigil(api_key=api_key, endpoint=os.getenv("VIGIL_ENDPOINT", "http://localhost:8080")) as c:
        yield c

def test_observe_decorator_exists(client):
    """observe() method exists on Vigil and returns a callable."""
    dec = client.observe
    assert callable(dec)

def test_observe_wraps_function_sync(client):
    """@observe wraps a sync function and still returns its value."""
    @client.observe
    def add(x: int, y: int) -> int:
        return x + y

    result = add(2, 3)
    assert result == 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
uv run pytest tests/test_decorators.py::test_observe_decorator_exists tests/test_decorators.py::test_observe_wraps_function_sync -v
```

Expected: `FAILED` — `AttributeError: 'Vigil' object has no attribute 'observe'`

- [ ] **Step 3: Create `src/vigil/decorators.py` with sync `make_observe`**

```python
# src/vigil/decorators.py
from __future__ import annotations

import functools
import inspect
import time
import warnings
from typing import Any, Callable, TypeVar

from ._context import get_current_run

F = TypeVar("F", bound=Callable[..., Any])

_SENTINEL = object()


def make_observe(vigil_client: Any, name: str | None, step_type: str) -> Callable[[F], F]:
    """Return a decorator that instruments a sync function."""

    def decorator(fn: F) -> F:
        fn_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic()
            exc_caught: BaseException | None = None
            result: Any = _SENTINEL
            try:
                result = fn(*args, **kwargs)
                return result
            except BaseException as exc:
                exc_caught = exc
                raise
            finally:
                latency_ms = int((time.monotonic() - t0) * 1000)
                ok = exc_caught is None
                _emit_sync(
                    vigil_client,
                    fn_name=fn_name,
                    step_type=step_type,
                    args=args,
                    kwargs=kwargs,
                    result=result if result is not _SENTINEL else None,
                    ok=ok,
                    latency_ms=latency_ms,
                    error=exc_caught,
                )

        return wrapper  # type: ignore[return-value]

    return decorator


def _emit_sync(
    vigil_client: Any,
    *,
    fn_name: str,
    step_type: str,
    args: tuple,
    kwargs: dict,
    result: Any,
    ok: bool,
    latency_ms: int,
    error: BaseException | None,
) -> None:
    try:
        run = get_current_run()
        tool_input = _build_input(args, kwargs)
        tool_output = _build_output(result, error)

        if run is not None:
            if step_type == "tool_call":
                run.tool_call(
                    fn_name,
                    input=tool_input,
                    output=tool_output,
                    ok=ok,
                    latency_ms=latency_ms,
                )
            else:
                run.step(
                    step_type,
                    content=str(result) if result is not None else None,
                    metadata={"fn": fn_name, "latency_ms": latency_ms},
                )
        else:
            # No active run — emit a standalone ai_trace
            vigil_client.ingest_ai(
                model=fn_name,
                provider="observe",
                status="success" if ok else "error",
                latency_ms=latency_ms,
                error_message=str(error)[:500] if error else None,
                metadata={"input": tool_input, "output": tool_output},
            )
    except Exception as emit_err:
        warnings.warn(f"vigil observe emit failed: {emit_err}", stacklevel=3)


def _build_input(args: tuple, kwargs: dict) -> dict:
    result: dict = {}
    if args:
        result["args"] = [_safe_str(a) for a in args]
    if kwargs:
        result.update({k: _safe_str(v) for k, v in kwargs.items()})
    return result


def _build_output(result: Any, error: BaseException | None) -> dict:
    if error is not None:
        return {"error": str(error)[:500]}
    if isinstance(result, dict):
        return result
    if result is None:
        return {}
    return {"value": _safe_str(result)}


def _safe_str(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return str(v)[:500]
    except Exception:
        return "<unserializable>"
```

- [ ] **Step 4: Add `.observe()` to `Vigil` in `client.py`**

Add this method to the `Vigil` class (after `wrap_openai`):

```python
def observe(
    self,
    fn=None,
    *,
    name: str | None = None,
    step_type: str = "tool_call",
):
    """Decorator that instruments a sync function.

    Usage:
        @client.observe
        def my_tool(q: str) -> dict: ...

        @client.observe(name="search", step_type="tool_call")
        def my_tool(q: str) -> dict: ...
    """
    from .decorators import make_observe
    dec = make_observe(self, name=name, step_type=step_type)
    if fn is not None:
        # called as @client.observe (no parens)
        return dec(fn)
    return dec
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
uv run pytest tests/test_decorators.py::test_observe_decorator_exists tests/test_decorators.py::test_observe_wraps_function_sync -v
```

Expected: `PASSED` for both.

- [ ] **Step 6: Commit**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
git add src/vigil/decorators.py src/vigil/client.py tests/test_decorators.py
git commit -m "feat(sdk): add make_observe factory + Vigil.observe() sync decorator"
```

---

## Task 2: `@observe` emits tool_call step when inside a run

**Files:**
- Modify: `tests/test_decorators.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_decorators.py`:

```python
def test_observe_emits_tool_call_inside_run(client):
    """@observe inside an active run calls run.tool_call() with correct args."""
    calls = []

    # Patch run.tool_call to capture the call without hitting the server
    import unittest.mock as mock

    @client.observe
    def fake_search(q: str) -> dict:
        return {"results": ["a", "b"]}

    with client.run("test-agent", input="test") as run:
        with mock.patch.object(run, "tool_call", wraps=run.tool_call) as mock_tc:
            result = fake_search(q="hello")
            assert result == {"results": ["a", "b"]}
            mock_tc.assert_called_once()
            call_kwargs = mock_tc.call_args
            assert call_kwargs[0][0] == "fake_search"  # positional tool_name
            assert call_kwargs[1]["ok"] is True
            assert "latency_ms" in call_kwargs[1]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
uv run pytest tests/test_decorators.py::test_observe_emits_tool_call_inside_run -v
```

Expected: `FAILED` — the ContextVar integration isn't tested yet; verify it fails for the right reason (assertion fails or import error).

- [ ] **Step 3: Run test to verify it passes**

No code change needed — the implementation in Task 1 already handles this via `_context.get_current_run()`. If the test fails, check that `_emit_sync` correctly reads `get_current_run()`.

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
uv run pytest tests/test_decorators.py::test_observe_emits_tool_call_inside_run -v
```

Expected: `PASSED`

- [ ] **Step 4: Commit**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
git add tests/test_decorators.py
git commit -m "test(sdk): verify @observe links tool_call to active run via ContextVar"
```

---

## Task 3: `@observe` emits standalone `ai_trace` outside a run

**Files:**
- Modify: `tests/test_decorators.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_decorators.py`:

```python
def test_observe_emits_ai_trace_outside_run(client):
    """@observe with no active run emits a standalone ai_trace row."""
    import unittest.mock as mock

    @client.observe(name="standalone_fn", step_type="tool_call")
    def standalone(x: int) -> int:
        return x * 2

    with mock.patch.object(client, "ingest_ai", wraps=client.ingest_ai) as mock_ai:
        result = standalone(5)
        assert result == 10
        mock_ai.assert_called_once()
        call_kwargs = mock_ai.call_args[1]
        assert call_kwargs["model"] == "standalone_fn"
        assert call_kwargs["status"] == "success"
        assert "latency_ms" in call_kwargs
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
uv run pytest tests/test_decorators.py::test_observe_emits_ai_trace_outside_run -v
```

Expected: `FAILED`

- [ ] **Step 3: Run test to verify it passes**

The `_emit_sync` function already falls through to `ingest_ai` when `run is None`. No code change needed.

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
uv run pytest tests/test_decorators.py::test_observe_emits_ai_trace_outside_run -v
```

Expected: `PASSED`

- [ ] **Step 4: Commit**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
git add tests/test_decorators.py
git commit -m "test(sdk): verify @observe fallback to ai_trace when no active run"
```

---

## Task 4: Async `@observe` via `make_observe_async`

**Files:**
- Modify: `src/vigil/decorators.py`
- Modify: `src/vigil/async_client.py`
- Modify: `tests/test_decorators.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_decorators.py`:

```python
import pytest_asyncio
import pytest

@pytest.mark.asyncio
async def test_observe_wraps_async_function(async_client):
    """@observe wraps an async function and returns its value."""
    @async_client.observe
    async def async_add(x: int, y: int) -> int:
        return x + y

    result = await async_add(3, 4)
    assert result == 7

@pytest.mark.asyncio
async def test_observe_async_emits_tool_call_inside_run(async_client):
    """@observe on async fn links to active async run via ContextVar."""
    import unittest.mock as mock

    @async_client.observe
    async def async_tool(q: str) -> dict:
        return {"result": q}

    async with async_client.run("async-test-agent", input="q") as run:
        with mock.patch.object(run, "tool_call", wraps=run.tool_call) as mock_tc:
            result = await async_tool(q="test")
            assert result == {"result": "test"}
            mock_tc.assert_called_once()
            call_kwargs = mock_tc.call_args
            assert call_kwargs[0][0] == "async_tool"
            assert call_kwargs[1]["ok"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
uv run pytest tests/test_decorators.py::test_observe_wraps_async_function tests/test_decorators.py::test_observe_async_emits_tool_call_inside_run -v
```

Expected: `FAILED` — `AsyncVigil` has no `.observe()` yet.

- [ ] **Step 3: Add `make_observe_async` to `decorators.py`**

Add after `make_observe` in `src/vigil/decorators.py`:

```python
def make_observe_async(vigil_client: Any, name: str | None, step_type: str) -> Callable[[F], F]:
    """Return a decorator that instruments an async function."""

    def decorator(fn: F) -> F:
        fn_name = name or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic()
            exc_caught: BaseException | None = None
            result: Any = _SENTINEL
            try:
                result = await fn(*args, **kwargs)
                return result
            except BaseException as exc:
                exc_caught = exc
                raise
            finally:
                latency_ms = int((time.monotonic() - t0) * 1000)
                ok = exc_caught is None
                await _emit_async(
                    vigil_client,
                    fn_name=fn_name,
                    step_type=step_type,
                    args=args,
                    kwargs=kwargs,
                    result=result if result is not _SENTINEL else None,
                    ok=ok,
                    latency_ms=latency_ms,
                    error=exc_caught,
                )

        return wrapper  # type: ignore[return-value]

    return decorator


async def _emit_async(
    vigil_client: Any,
    *,
    fn_name: str,
    step_type: str,
    args: tuple,
    kwargs: dict,
    result: Any,
    ok: bool,
    latency_ms: int,
    error: BaseException | None,
) -> None:
    try:
        run = get_current_run()
        tool_input = _build_input(args, kwargs)
        tool_output = _build_output(result, error)

        if run is not None:
            if step_type == "tool_call":
                await run.tool_call(
                    fn_name,
                    input=tool_input,
                    output=tool_output,
                    ok=ok,
                    latency_ms=latency_ms,
                )
            else:
                await run.step(
                    step_type,
                    content=str(result) if result is not None else None,
                    metadata={"fn": fn_name, "latency_ms": latency_ms},
                )
        else:
            await vigil_client.ingest_ai(
                model=fn_name,
                provider="observe",
                status="success" if ok else "error",
                latency_ms=latency_ms,
                error_message=str(error)[:500] if error else None,
                metadata={"input": tool_input, "output": tool_output},
            )
    except Exception as emit_err:
        warnings.warn(f"vigil observe emit failed: {emit_err}", stacklevel=3)
```

- [ ] **Step 4: Add `.observe()` to `AsyncVigil` in `async_client.py`**

Add this method to the `AsyncVigil` class (after `wrap_openai`):

```python
def observe(
    self,
    fn=None,
    *,
    name: str | None = None,
    step_type: str = "tool_call",
):
    """Decorator that instruments an async function.

    Usage:
        @async_client.observe
        async def my_tool(q: str) -> dict: ...

        @async_client.observe(name="search", step_type="tool_call")
        async def my_tool(q: str) -> dict: ...
    """
    from .decorators import make_observe_async
    dec = make_observe_async(self, name=name, step_type=step_type)
    if fn is not None:
        return dec(fn)
    return dec
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
uv run pytest tests/test_decorators.py::test_observe_wraps_async_function tests/test_decorators.py::test_observe_async_emits_tool_call_inside_run -v
```

Expected: `PASSED`

- [ ] **Step 6: Commit**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
git add src/vigil/decorators.py src/vigil/async_client.py tests/test_decorators.py
git commit -m "feat(sdk): add async @observe decorator via make_observe_async"
```

---

## Task 5: `@agent` sync decorator

**Files:**
- Modify: `src/vigil/decorators.py`
- Modify: `src/vigil/client.py`
- Modify: `tests/test_decorators.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_decorators.py`:

```python
def test_agent_decorator_exists(client):
    """agent() method exists on Vigil and returns a callable."""
    assert callable(client.agent)

def test_agent_wraps_function_sync(client):
    """@agent wraps a sync function, opens a run, returns the fn's value."""
    @client.agent(name="test-sync-agent")
    def my_agent(task: str) -> str:
        return f"done: {task}"

    result = my_agent("hello")
    assert result == "done: hello"

def test_agent_sets_current_run(client):
    """@agent sets _current_run so nested @observe calls auto-link."""
    from vigil._context import get_current_run
    import unittest.mock as mock

    captured_run = []

    @client.observe
    def inner_tool(x: int) -> int:
        captured_run.append(get_current_run())
        return x + 1

    @client.agent(name="test-context-agent")
    def my_agent(task: str) -> str:
        inner_tool(1)
        return "ok"

    my_agent("task")
    assert len(captured_run) == 1
    assert captured_run[0] is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
uv run pytest tests/test_decorators.py::test_agent_decorator_exists tests/test_decorators.py::test_agent_wraps_function_sync tests/test_decorators.py::test_agent_sets_current_run -v
```

Expected: `FAILED` — `Vigil` has no `.agent()` yet.

- [ ] **Step 3: Add `make_agent` to `decorators.py`**

Add after `_safe_str` in `src/vigil/decorators.py`:

```python
def make_agent(vigil_client: Any, name: str | None) -> Callable[[F], F]:
    """Return a decorator that wraps a sync function in a vigil Run."""

    def decorator(fn: F) -> F:
        agent_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Use first positional arg as run input if available
            input_str: str | None = None
            if args:
                input_str = _safe_str(args[0])
            elif "input" in kwargs:
                input_str = _safe_str(kwargs["input"])

            try:
                with vigil_client.run(agent_name, input=input_str) as run:
                    result = fn(*args, **kwargs)
                    if isinstance(result, str):
                        run.set_output(result)
                    elif result is not None:
                        run.set_output(_safe_str(result))
                    return result
            except Exception:
                raise

        return wrapper  # type: ignore[return-value]

    return decorator
```

- [ ] **Step 4: Add `.agent()` to `Vigil` in `client.py`**

Add this method to the `Vigil` class (after `.observe()`):

```python
def agent(
    self,
    fn=None,
    *,
    name: str | None = None,
):
    """Decorator that wraps a sync function in a vigil Run.

    Usage:
        @client.agent
        def my_agent(task: str) -> str: ...

        @client.agent(name="research-agent")
        def my_agent(task: str) -> str: ...
    """
    from .decorators import make_agent
    dec = make_agent(self, name=name)
    if fn is not None:
        return dec(fn)
    return dec
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
uv run pytest tests/test_decorators.py::test_agent_decorator_exists tests/test_decorators.py::test_agent_wraps_function_sync tests/test_decorators.py::test_agent_sets_current_run -v
```

Expected: `PASSED`

- [ ] **Step 6: Commit**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
git add src/vigil/decorators.py src/vigil/client.py tests/test_decorators.py
git commit -m "feat(sdk): add @agent sync decorator via make_agent"
```

---

## Task 6: `@agent` async decorator

**Files:**
- Modify: `src/vigil/decorators.py`
- Modify: `src/vigil/async_client.py`
- Modify: `tests/test_decorators.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_decorators.py`:

```python
@pytest.mark.asyncio
async def test_agent_wraps_async_function(async_client):
    """@agent wraps an async function, opens an AsyncRun, returns the fn's value."""
    @async_client.agent(name="test-async-agent")
    async def my_agent(task: str) -> str:
        return f"async done: {task}"

    result = await my_agent("hello")
    assert result == "async done: hello"

@pytest.mark.asyncio
async def test_agent_async_sets_current_run(async_client):
    """@agent async sets _current_run so nested @observe calls auto-link."""
    from vigil._context import get_current_run

    captured_run = []

    @async_client.observe
    async def inner_tool(x: int) -> int:
        captured_run.append(get_current_run())
        return x + 1

    @async_client.agent(name="test-async-context-agent")
    async def my_agent(task: str) -> str:
        await inner_tool(1)
        return "ok"

    await my_agent("task")
    assert len(captured_run) == 1
    assert captured_run[0] is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
uv run pytest tests/test_decorators.py::test_agent_wraps_async_function tests/test_decorators.py::test_agent_async_sets_current_run -v
```

Expected: `FAILED` — `AsyncVigil` has no `.agent()` yet.

- [ ] **Step 3: Add `make_agent_async` to `decorators.py`**

Add after `make_agent` in `src/vigil/decorators.py`:

```python
def make_agent_async(vigil_client: Any, name: str | None) -> Callable[[F], F]:
    """Return a decorator that wraps an async function in a vigil AsyncRun."""

    def decorator(fn: F) -> F:
        agent_name = name or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            input_str: str | None = None
            if args:
                input_str = _safe_str(args[0])
            elif "input" in kwargs:
                input_str = _safe_str(kwargs["input"])

            try:
                async with vigil_client.run(agent_name, input=input_str) as run:
                    result = await fn(*args, **kwargs)
                    if isinstance(result, str):
                        run.set_output(result)
                    elif result is not None:
                        run.set_output(_safe_str(result))
                    return result
            except Exception:
                raise

        return wrapper  # type: ignore[return-value]

    return decorator
```

- [ ] **Step 4: Add `.agent()` to `AsyncVigil` in `async_client.py`**

Add after `.observe()` in `AsyncVigil`:

```python
def agent(
    self,
    fn=None,
    *,
    name: str | None = None,
):
    """Decorator that wraps an async function in a vigil AsyncRun.

    Usage:
        @async_client.agent
        async def my_agent(task: str) -> str: ...

        @async_client.agent(name="research-agent")
        async def my_agent(task: str) -> str: ...
    """
    from .decorators import make_agent_async
    dec = make_agent_async(self, name=name)
    if fn is not None:
        return dec(fn)
    return dec
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
uv run pytest tests/test_decorators.py::test_agent_wraps_async_function tests/test_decorators.py::test_agent_async_sets_current_run -v
```

Expected: `PASSED`

- [ ] **Step 6: Commit**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
git add src/vigil/decorators.py src/vigil/async_client.py tests/test_decorators.py
git commit -m "feat(sdk): add @agent async decorator via make_agent_async"
```

---

## Task 7: Run full decorator test suite + example

**Files:**
- Modify: `tests/test_decorators.py`
- Create: `examples/decorator_agent.py`

- [ ] **Step 1: Run all decorator tests**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
uv run pytest tests/test_decorators.py -v
```

Expected: all tests `PASSED`, no warnings about unrecognised markers.

- [ ] **Step 2: Write the example**

Create `examples/decorator_agent.py`:

```python
"""Minimal decorator example — shows @agent + @observe in 20 lines.

Run:
    VIGIL_API_KEY=vgl_... uv run python examples/decorator_agent.py
"""
import os
from vigil import Vigil

client = Vigil(
    api_key=os.environ["VIGIL_API_KEY"],
    endpoint=os.environ.get("VIGIL_ENDPOINT", "http://localhost:8080"),
)


@client.observe(name="web_search", step_type="tool_call")
def web_search(q: str) -> dict:
    # Fake search — replace with real implementation
    return {"results": [f"result for: {q}"]}


@client.agent(name="decorator-demo-agent")
def run_agent(task: str) -> str:
    results = web_search(q=task)
    return f"Found: {results['results'][0]}"


if __name__ == "__main__":
    answer = run_agent("what is vigilops")
    print(answer)
```

- [ ] **Step 3: Run the example (requires live server)**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python/examples
VIGIL_API_KEY=$VIGIL_API_KEY uv run python decorator_agent.py
```

Expected output: `Found: result for: what is vigilops`

- [ ] **Step 4: Commit**

```bash
cd /home/theyusuf/Desktop/saas/vigilops/python
git add examples/decorator_agent.py
git commit -m "docs(sdk): add decorator_agent.py example for @observe + @agent"
```

---

## Self-Review

**Spec coverage:**
- [x] `@observe` sync — Tasks 1–3
- [x] `@observe` async — Task 4
- [x] `@agent` sync — Task 5
- [x] `@agent` async — Task 6
- [x] Implicit ContextVar (option C) — `_emit_sync`/`_emit_async` read `get_current_run()`
- [x] Fallback to `ai_trace` outside run — Task 3
- [x] `step_type` explicit param defaulting to `"tool_call"` — `make_observe` signature
- [x] Fail-soft: `warnings.warn` on emit error — `_emit_sync` / `_emit_async` try/except
- [x] `name` defaults to `fn.__name__` — `fn_name = name or fn.__name__`
- [x] Both no-paren (`@client.observe`) and parens (`@client.observe(...)`) forms — sentinel pattern in `.observe()` method

**Placeholder scan:** None found.

**Type consistency:**
- `make_observe` → `Vigil.observe()` → calls `make_observe(self, name=name, step_type=step_type)` ✓
- `make_observe_async` → `AsyncVigil.observe()` → calls `make_observe_async(self, name=name, step_type=step_type)` ✓
- `make_agent` → `Vigil.agent()` → calls `make_agent(self, name=name)` ✓
- `make_agent_async` → `AsyncVigil.agent()` → calls `make_agent_async(self, name=name)` ✓
- `run.tool_call(fn_name, input=..., output=..., ok=..., latency_ms=...)` matches `Run.tool_call` signature ✓
- `run.step(step_type, content=..., metadata=...)` matches `Run.step` signature ✓
- `async run.tool_call(...)` / `async run.step(...)` match `AsyncRun` signatures ✓
