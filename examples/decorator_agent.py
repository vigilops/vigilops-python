"""Minimal @observe + @agent example.

Run:
    KEELWAVE_API_KEY=kw_... uv run python examples/decorator_agent.py
"""
import os
from keelwave import Keelwave

client = Keelwave(
    api_key=os.environ["KEELWAVE_API_KEY"],
    endpoint=os.environ.get("KEELWAVE_ENDPOINT", "http://localhost:8080"),
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
    answer = run_agent("what is keelwave")
    print(answer)
