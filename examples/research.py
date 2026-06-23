"""Multi-turn Claude research agent, traced by keelwave.

Demonstrates @observe + @agent decorators:
- `@client.observe` on `web_search` auto-records every tool call as an
  agent_step and links it to the active run via ContextVar.
- `@client.agent` on `run_agent` opens/closes the Run automatically.
- `client.wrap_anthropic` still handles ai_traces for every LLM call.
"""

import json
import os

import anthropic

from keelwave import Keelwave

from tools import web_search as _web_search


TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web. Returns title/url/snippet for top results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "search query"},
            },
            "required": ["q"],
        },
    },
]

keelwave_client = Keelwave(
    api_key=os.environ["KEELWAVE_API_KEY"],
    endpoint=os.environ.get("KEELWAVE_ENDPOINT", "http://localhost:8080"),
)

model = os.environ.get("CLAUDE_MODEL", "deepseek-v4-pro")
provider = os.environ.get("LLM_PROVIDER", "deepseek")

claude = keelwave_client.wrap_anthropic(
    anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.environ.get(
            "ANTHROPIC_ENDPOINT", "https://api.deepseek.com/anthropic"
        ),
    ),
    provider=provider,
)


@keelwave_client.observe(name="web_search", step_type="tool_call")
def web_search(q: str, max_results: int = 5) -> dict:
    return {"results": _web_search(q, max_results=max_results)}


@keelwave_client.agent(name="research-agent")
def run_agent(question: str) -> str:
    MAX_TURNS = 8
    messages = [{"role": "user", "content": question}]

    for turn in range(MAX_TURNS):
        kwargs: dict = {
            "model": model,
            "max_tokens": 2048,
            "tools": TOOLS,
            "messages": messages,
        }
        if turn == 0:
            kwargs["tool_choice"] = {"type": "any"}

        resp = claude.messages.create(**kwargs)

        text = "\n".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]

        # thinking blocks auto-emitted as think steps by wrap_anthropic

        if not tool_uses:
            if not text:
                raise RuntimeError(f"empty final response: {resp.content!r}")
            return text

        tool_results = []
        for tu in tool_uses:
            # web_search is @observe — tool_call step emitted automatically
            result = web_search(q=tu.input["q"], max_results=5)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result),
                }
            )

        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": tool_results})
    else:
        raise RuntimeError(f"agent exceeded {MAX_TURNS} turns")


def main() -> None:
    question = (
        "what is observability platform? and what are the top 3 providers in 2026? "
        "use web_search tool to find out, and cite your sources"
    )
    answer = run_agent(question)
    print("Final answer:\n", answer)


if __name__ == "__main__":
    main()
