"""Tracks the currently-open Run so adapter wrappers (wrap_anthropic etc.)
can auto-link ai_traces to the right agent_run_id without the user
threading it manually.

The ContextVar is per-task in asyncio — `asyncio.gather` calls each get
their own snapshot, so nested concurrent runs don't leak parent state.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .async_run import AsyncRun
    from .run import Run

    AnyRun = Union[Run, AsyncRun]


_current_run: ContextVar[AnyRun | None] = ContextVar(
    "keelwave_current_run", default=None
)


def get_current_run() -> AnyRun | None:
    """Return the Run/AsyncRun currently inside its with-block, or None."""
    return _current_run.get()


def set_current_run(run: AnyRun) -> Token:
    """Set the active Run. Returns a token to pass to reset_current_run()."""
    return _current_run.set(run)


def reset_current_run(token: Token) -> None:
    _current_run.reset(token)
