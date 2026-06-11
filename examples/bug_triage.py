"""Bug-triage agent demo. Uses OpenAI + vigil.wrap_openai.

The agent is given a failing-test report and three tools (`list_dir`,
`read_file`, `grep`). It walks the small fixture package in
`examples/_fixture_buggy/` and reports the actual buggy line.

What this demo proves vs research.py
- a DIFFERENT shape of agent — filesystem tools instead of network
- the OpenAI adapter records `ai_traces` exactly like Anthropic's
- the same `Run` + step + tool_call surface fits both providers
"""

import json
import os
import time
from pathlib import Path

import openai

from vigil import Vigil, parse_openai_response


FIXTURE_DIR = Path(__file__).parent / "_fixture_buggy"

FAILING_REPORT = """
Bug report from QA:

Customer complaint: after applying a $10-off coupon at checkout, the
order total went UP, not down. Refund issued. Need root cause.

Repro (pytest):

    cart = [(10.0, 2), (4.0, 1)]   # 2 x $10 + 1 x $4 = $24 subtotal
    # tax 8% = $1.92
    # expected with $10 coupon = $24 + $1.92 - $10 = $15.92
    quote(cart, coupon_value=10.0)
    # actual: $35.92  ← wrong

Source lives in `examples/_fixture_buggy/`. Find the bug, name the
file and line, suggest the one-line fix.
""".strip()


# ---------------------------------------------------------------------------
# Tools

def _safe_path(rel: str) -> Path:
    """Resolve `rel` strictly under FIXTURE_DIR. Prevents path-escape."""
    p = (FIXTURE_DIR / rel).resolve()
    if FIXTURE_DIR.resolve() not in p.parents and p != FIXTURE_DIR.resolve():
        raise ValueError(f"path escapes fixture: {rel}")
    return p


def list_dir(rel: str = ".") -> list[str]:
    """List files + subdirs relative to the fixture root."""
    base = _safe_path(rel)
    return sorted(
        str(p.relative_to(FIXTURE_DIR))
        for p in base.iterdir()
        if not p.name.startswith("_") or p.name == "__init__.py"
    )


def read_file(rel: str) -> str:
    """Return the contents of a file under the fixture root."""
    return _safe_path(rel).read_text(encoding="utf-8")


def grep(pattern: str, rel: str = ".") -> list[str]:
    """Recursive grep under fixture root. Returns `file:line:text` matches."""
    base = _safe_path(rel)
    out: list[str] = []
    targets = [base] if base.is_file() else list(base.rglob("*.py"))
    for f in targets:
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
                if pattern in line:
                    out.append(f"{f.relative_to(FIXTURE_DIR)}:{i}:{line}")
        except (OSError, UnicodeDecodeError):
            continue
    return out


TOOL_IMPL = {"list_dir": list_dir, "read_file": read_file, "grep": grep}


# ---------------------------------------------------------------------------
# Tool spec for OpenAI

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files + subdirs under a relative path (fixture root).",
            "parameters": {
                "type": "object",
                "properties": {
                    "rel": {"type": "string", "description": "relative path; '.' for root"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a Python file's full text by relative path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rel": {"type": "string", "description": "relative path to a file"},
                },
                "required": ["rel"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a substring in .py files; returns file:line:text matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "rel": {"type": "string", "description": "subdir or file; '.' for all"},
                },
                "required": ["pattern"],
            },
        },
    },
]


SYSTEM_PROMPT = (
    "You are a senior engineer triaging a production bug. "
    "Use the provided tools to inspect the codebase. "
    "When you are sure, reply with: "
    "1. the file and line number of the bug, "
    "2. a one-sentence explanation of what's wrong, "
    "3. the corrected code (one line). "
    "Do not call tools after you have your answer."
)


def main() -> None:
    vigil_key = os.environ["VIGIL_API_KEY"]
    vigil_endpoint = os.environ.get("VIGIL_ENDPOINT", "http://localhost:8080")
    model = os.environ.get("OPENAI_MODEL", "deepseek-v4-flash")
    provider = os.environ.get("LLM_PROVIDER", "deepseek")

    real_openai = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_ENDPOINT", "https://api.deepseek.com"),
    )

    MAX_TURNS = 12
    final: str | None = None

    run_metadata = {
        "model": model,
        "provider": provider,
        "max_turns": MAX_TURNS,
        "tools": [t["function"]["name"] for t in TOOLS],
        "task": "bug-triage",
    }

    with Vigil(api_key=vigil_key, endpoint=vigil_endpoint) as vigil_client:
        client = vigil_client.wrap_openai(real_openai, provider=provider)

        with vigil_client.run("bug-triage-agent",
                              input=FAILING_REPORT,
                              metadata=run_metadata) as run:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": FAILING_REPORT},
            ]

            for turn in range(MAX_TURNS):
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=2048,
                    tools=TOOLS,
                    tool_choice="auto",
                    messages=messages,
                )
                parsed = parse_openai_response(resp)

                msg = resp.choices[0].message
                text = msg.content or ""
                tool_calls = msg.tool_calls or []

                if text:
                    run.step(
                        "think",
                        content=text,
                        tokens=parsed.tokens_total(),
                        metadata={"turn": turn, "openai_id": parsed.request_id,
                                  "finish_reason": parsed.finish_reason},
                    )

                if not tool_calls:
                    if not text:
                        run.step(
                            "think",
                            content=f"turn {turn}: no tool_calls and no content",
                            metadata={"turn": turn, "error": "empty_response"},
                        )
                        raise RuntimeError(f"empty final response: {msg!r}")
                    final = text
                    break

                tool_results = []
                for tc in tool_calls:
                    name = tc.function.name
                    impl = TOOL_IMPL.get(name)
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {"_raw": tc.function.arguments}

                    if impl is None:
                        out_payload = {"error": f"unknown tool: {name}"}
                        run.tool_call(name, input=args, output=out_payload, ok=False,
                                      metadata={"turn": turn, "openai_tool_call_id": tc.id})
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(out_payload),
                        })
                        continue

                    t_tool = time.monotonic()
                    try:
                        result = impl(**args)
                        lat = int((time.monotonic() - t_tool) * 1000)
                        run.tool_call(
                            name,
                            input=args,
                            output={"result": result} if not isinstance(result, dict) else result,
                            ok=True,
                            latency_ms=lat,
                            metadata={"turn": turn, "openai_tool_call_id": tc.id},
                        )
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"result": result}),
                        })
                    except Exception as e:
                        lat = int((time.monotonic() - t_tool) * 1000)
                        run.tool_call(
                            name,
                            input=args,
                            output={"error": str(e)[:500]},
                            ok=False,
                            latency_ms=lat,
                            metadata={"turn": turn, "openai_tool_call_id": tc.id,
                                      "error_type": type(e).__name__},
                        )
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"error": str(e)[:500]}),
                        })

                messages.append({
                    "role": "assistant",
                    "content": text,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ],
                })
                messages.extend(tool_results)
            else:
                run.step("replan",
                         content=f"hit MAX_TURNS={MAX_TURNS} without final answer")
                raise RuntimeError(f"agent exceeded {MAX_TURNS} turns")

            run.set_output(final)

    print("Triage report:\n" + "-" * 60)
    print(final)
    print("-" * 60)
    print(f"\nrun_id: {run.id}")


if __name__ == "__main__":
    main()
