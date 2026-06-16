"""Minimum vigil + claude example. One LLM call, one vigil run, one step."""

import os

import anthropic
from vigilops import Vigil


def main():
    vigil_key = os.environ["VIGILOPS_API_KEY"]
    vigil_endpoint = os.environ.get("VIGILOPS_ENDPOINT", "http://localhost:8080")

    claude = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.environ.get("ANTHROPIC_ENDPOINT", "https://api.deepseek.com/anthropic"),
    )

    with Vigil(api_key=vigil_key, endpoint=vigil_endpoint) as vigil_client:
        with vigil_client.run("hello-agent", input="say salam in 5 words") as run:
            resp = claude.messages.create(
                model=os.environ.get("CLAUDE_MODEL", "deepseek-v4-pro"),
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": "say salam in exactly 5 words"},
                ],
            )

            text_blocks = [b for b in resp.content if getattr(b, "type", None) == "text"]
            if not text_blocks:
                run.step(
                    "think",
                    content=f"no text block from LLM; raw blocks: {[type(b).__name__ for b in resp.content]}",
                )
                raise RuntimeError(f"no text block in response: {resp.content!r}")

            answer = text_blocks[0].text
            run.step(
                "think",
                content=answer,
                tokens=resp.usage.input_tokens + resp.usage.output_tokens,
            )
            run.set_output(answer)

    print("Claude said:", answer)
    print(f"Vigil recorded run_id: {run.id}")
    print(f"View in DB: SELECT * FROM agent_runs WHERE id = '{run.id}';")


if __name__ == "__main__":
    main()
