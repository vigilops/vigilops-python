"""Minimal @observe + @agent example.

Run:
    VIGILOPS_API_KEY=vgl_... uv run python examples/decorator_agent.py
"""

import os
from vigilops import Vigil

client = Vigil(
    api_key=os.environ["VIGILOPS_API_KEY"],
    endpoint=os.environ.get("VIGILOPS_ENDPOINT", "http://localhost:8080"),
)


@client.observe(name="web_search", step_type="tool_call")
def web_search(q: str) -> dict:
    # Replace with real search implementation
    return {"results": [f"result for: {q}"]}


@client.agent(name="decorator-demo-agent")
def run_agent(task: str) -> str:
    results = web_search(q=task)
    return f"Found: {results['results'][0]}"


if __name__ == "__main__":
    answer = run_agent("what is vigilops")
    print(answer)
