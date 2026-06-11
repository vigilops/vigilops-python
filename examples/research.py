"""Multi-turn Claude research agent, traced by vigil.

Uses the v0.0.2 sugar layer:
- `vigil_client.wrap_anthropic(claude)` auto-records every messages.create
  call to ai_traces and links agent_run_id via ContextVar.
- `parse_anthropic_response(resp)` exposes the universal observability
  fields (id, tokens, stop_reason). Block-type walking (text / thinking
  / tool_uses) is inline in user code below — Anthropic's content shape
  evolves and is not vigil's job to track.

What still lives in user code: the agent loop, tool execution, messages
stitching, and content-block extraction. SDK handles all recording.
"""

import json
import os
import time

import anthropic

from vigil import Vigil, parse_anthropic_response

from tools import web_search


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


def main() -> None:
    question = (
        "what is observability platform? and what are the top 3 providers in 2026? use web_search tool to find out, and cite your sources"
    )

    vigil_key = os.environ["VIGIL_API_KEY"]
    vigil_endpoint = os.environ.get("VIGIL_ENDPOINT", "http://localhost:8080")
    model = os.environ.get("CLAUDE_MODEL", "deepseek-v4-pro")
    provider = os.environ.get("LLM_PROVIDER", "deepseek")

    real_claude = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.environ.get("ANTHROPIC_ENDPOINT", "https://api.deepseek.com/anthropic"),
    )

    MAX_TURNS = 8
    final: str | None = None

    run_metadata = {
        "model": model,
        "provider": provider,
        "max_turns": MAX_TURNS,
        "tools": [t["name"] for t in TOOLS],
    }

    with Vigil(api_key=vigil_key, endpoint=vigil_endpoint) as vigil_client:
        # Wrap once — every .messages.create now auto-records to ai_traces.
        claude = vigil_client.wrap_anthropic(real_claude, provider=provider)

        with vigil_client.run("research-agent", input=question, metadata=run_metadata) as run:
            messages = [{"role": "user", "content": question}]

            for turn in range(MAX_TURNS):
                tool_choice = {"type": "any"} if turn == 0 else None

                kwargs: dict = {
                    "model": model,
                    "max_tokens": 2048,
                    "tools": TOOLS,
                    "messages": messages,
                }
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice

                resp = claude.messages.create(**kwargs)
                parsed = parse_anthropic_response(resp)

                # Block-type walking lives in user code — Anthropic's
                # content shape evolves and isn't vigil's job to track.
                text = "\n".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
                thinking = "\n".join(b.thinking for b in resp.content if getattr(b, "type", None) == "thinking")
                tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]

                if thinking:
                    run.step(
                        "think",
                        content=thinking,
                        tokens=parsed.tokens_total(),
                        metadata={"turn": turn, "claude_id": parsed.request_id},
                    )

                if not tool_uses:
                    if not text:
                        run.step(
                            "think",
                            content=f"turn {turn}: no tool_use and no text",
                            metadata={"turn": turn, "error": "empty_response"},
                        )
                        raise RuntimeError(f"empty final response: {resp.content!r}")
                    final = text
                    break

                tool_results = []
                for tu in tool_uses:
                    t_tool = time.monotonic()
                    try:
                        search_results = web_search(tu.input["q"], max_results=5)
                        lat = int((time.monotonic() - t_tool) * 1000)
                        run.tool_call(
                            tu.name,
                            input=tu.input,
                            output={"results": search_results},
                            ok=True,
                            latency_ms=lat,
                            metadata={"turn": turn, "claude_tool_use_id": tu.id,
                                      "result_count": len(search_results)},
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": json.dumps(search_results),
                        })
                    except Exception as e:
                        lat = int((time.monotonic() - t_tool) * 1000)
                        run.tool_call(
                            tu.name,
                            input=tu.input,
                            output={"error": str(e)[:500]},
                            ok=False,
                            latency_ms=lat,
                            metadata={"turn": turn, "claude_tool_use_id": tu.id,
                                      "error_type": type(e).__name__},
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": json.dumps({"error": str(e)[:500]}),
                            "is_error": True,
                        })

                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                run.step("replan",
                         content=f"hit MAX_TURNS={MAX_TURNS} without final answer")
                raise RuntimeError(f"agent exceeded {MAX_TURNS} turns")

            run.set_output(final)

    print("Final answer:\n", final)
    print(f"\nrun_id: {run.id}")


if __name__ == "__main__":
    main()
