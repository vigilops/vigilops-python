import time

import httpx
import pytest

from vigilops import Run


def test_run_lifecycle_completes_with_steps(client):
    with client.run("test-agent", input="hi") as run:
        assert isinstance(run, Run)
        run.step("think", "I should respond")
        run.tool_call("echo", input={"msg": "hi"}, output="hi", ok=True)
        run.set_output("done")


def test_run_marks_failed_on_exception(client):
    class BoomError(Exception):
        pass

    with pytest.raises(BoomError):
        with client.run("test-agent") as run:
            run.step("think", "about to fail")
            raise BoomError("simulated")


def test_loop_detection_via_repeated_tool_call(client):
    with client.run("looper") as run:
        for _ in range(3):
            run.tool_call("search", input={"q": "same"}, output={"results": []})

    # agent_steps are batched server-side (default 500ms flush); wait so
    # the loops query sees the rows once they hit the hypertable.
    time.sleep(2.5)

    resp = httpx.get(
        f"{client.endpoint}/v1/agent/runs/{run.id}/loops",
        params={"at": run.timestamp},
        headers={"Authorization": f"Bearer {client.api_key}"},
    )
    body = resp.json()["data"]
    assert body is not None, "expected loops payload, got null"
    assert any(h["hits"] >= 2 for h in body), f"expected loop hits: {body}"
