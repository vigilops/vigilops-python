"""Deliberately broken agent that runs the same web search forever.

Demonstrates vigil's headline differentiator: @observe tracks tool_call
fingerprints client-side — duplicate SHA-256 → run.mark_loop() fires
automatically. Zero user code for loop detection.

Run it:
    uv run --env-file .env python examples/looping_agent.py
"""

import os
import time

import httpx

from vigilops import Vigil, get_current_run

from tools import web_search as _web_search


TARGET_QUERY = "vigil observability platform"
LOOP_TURNS = 5

vigil_key = os.environ["VIGILOPS_API_KEY"]
vigil_endpoint = os.environ.get("VIGILOPS_ENDPOINT", "http://localhost:8080")

vigil_client = Vigil(api_key=vigil_key, endpoint=vigil_endpoint)


@vigil_client.observe(name="web_search", step_type="tool_call")
def web_search(q: str, max_results: int = 3) -> dict:
    return {"results": _web_search(q, max_results=max_results)}


@vigil_client.agent(name="looping-agent")
def run_loop(question: str) -> str:
    run = get_current_run()

    for _ in range(LOOP_TURNS):
        # @observe emits tool_call step + checks fingerprint automatically
        # duplicate on turn 1 → run.mark_loop() fires, no user code needed
        web_search(q=question, max_results=3)

    return run.id if run else ""


def main() -> None:
    run_id = run_loop(TARGET_QUERY)

    # Wait for batch buffer flush
    time.sleep(0.7)

    loops_url = f"{vigil_endpoint}/v1/agent/runs/{run_id}/loops"
    resp = httpx.get(
        loops_url,
        headers={"Authorization": f"Bearer {vigil_key}"},
        timeout=5.0,
    )
    resp.raise_for_status()
    hits = resp.json()["data"] or []

    print(f"\nrun_id: {run_id}")
    print(f"loops endpoint returned {len(hits)} fingerprint group(s):")
    for h in hits:
        print(
            f"  hits={h['hits']:>3}  tool={h.get('tool_name')!r:<20} "
            f"first_step={h['step_indices'][0]}  later_steps={h['step_indices'][1:]}"
        )

    if not any(h["hits"] >= 2 for h in hits):
        raise SystemExit("expected at least one fingerprint group with hits >= 2")


if __name__ == "__main__":
    main()
