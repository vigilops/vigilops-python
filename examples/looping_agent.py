"""Deliberately broken agent that runs the same web search forever.

Demonstrates vigil's headline differentiator: server-side SHA-256 fingerprint
on each step lets the loop view spot repeated tool calls — exactly the kind
of silent failure that costs real money in production.

What this proves
- Agent picks the SAME query every turn (`target_query`).
- vigil computes input_fingerprint server-side, so the SDK doesn't have to
  cooperate for detection to work.
- `/v1/agent/runs/{id}/loops?at=...` returns hit groups; we print them.
- `run.mark_loop(step_index)` flips `agent_runs.loop_detected = true` so
  dashboards/alerts can filter without joining on the loop view.

Run it
    uv run --env-file .env python examples/looping_agent.py
"""

import os
import time

import httpx

from vigil import Vigil

from tools import web_search


TARGET_QUERY = "vigil observability platform"
LOOP_TURNS = 5


def main() -> None:
    vigil_key = os.environ["VIGIL_API_KEY"]
    vigil_endpoint = os.environ.get("VIGIL_ENDPOINT", "http://localhost:8080")

    metadata = {
        "purpose": "loop-detection-demo",
        "target_query": TARGET_QUERY,
        "expected_loop_turns": LOOP_TURNS,
    }

    with Vigil(api_key=vigil_key, endpoint=vigil_endpoint) as vigil_client:
        with vigil_client.run("looping-agent",
                              input=f"answer using query: {TARGET_QUERY}",
                              metadata=metadata) as run:

            first_repeat_step: int | None = None
            for turn in range(LOOP_TURNS):
                run.step("think",
                         content=f"turn {turn}: I will search '{TARGET_QUERY}' again "
                                 f"(this agent never updates its plan)",
                         metadata={"turn": turn})

                t_tool = time.monotonic()
                results = web_search(TARGET_QUERY, max_results=3)
                lat = int((time.monotonic() - t_tool) * 1000)

                run.tool_call(
                    "web_search",
                    input={"q": TARGET_QUERY},
                    output={"results": results},
                    ok=True,
                    latency_ms=lat,
                    metadata={"turn": turn, "result_count": len(results)},
                )

                # First duplicate appears on turn 1 (steps 2, 4, 6, ... are tool_calls).
                if turn == 1 and first_repeat_step is None:
                    first_repeat_step = run._step_index  # tool_call step we just sent

            run.set_output(
                f"agent looped {LOOP_TURNS} times searching '{TARGET_QUERY}' without progress"
            )
            if first_repeat_step is not None:
                run.mark_loop(step_index=first_repeat_step)

    # Wait for agent_steps to flush from the server-side batch buffer.
    time.sleep(0.7)

    # No `at` param — server uses its 30-day default window, which covers
    # the full run regardless of duration. The ±1s window the server uses
    # when `at` IS supplied is too narrow for a multi-step agent run.
    loops_url = f"{vigil_endpoint}/v1/agent/runs/{run.id}/loops"
    resp = httpx.get(
        loops_url,
        headers={"Authorization": f"Bearer {vigil_key}"},
        timeout=5.0,
    )
    resp.raise_for_status()
    hits = resp.json()["data"] or []

    print(f"\nrun_id: {run.id}")
    print(f"loops endpoint returned {len(hits)} fingerprint group(s):")
    for h in hits:
        print(f"  hits={h['hits']:>3}  tool={h.get('tool_name')!r:<20} "
              f"first_step={h['step_indices'][0]}  later_steps={h['step_indices'][1:]}")

    if not any(h["hits"] >= 2 for h in hits):
        raise SystemExit("expected at least one fingerprint group with hits >= 2")


if __name__ == "__main__":
    main()
