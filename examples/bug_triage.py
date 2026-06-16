"""Bug-triage agent demo. Uses OpenAI + vigil.wrap_openai.

Demonstrates @observe + @agent decorators:
- `@client.observe` on each tool (`list_dir`, `read_file`, `grep`) auto-records
  tool_call steps and links them to the active run via ContextVar.
- `@client.agent` on `run_agent` opens/closes the Run automatically.
- `client.wrap_openai` still handles ai_traces for every LLM call.
"""

import json
import os
from pathlib import Path

import openai

from vigilops import Vigil


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


vigil_client = Vigil(
    api_key=os.environ["VIGILOPS_API_KEY"],
    endpoint=os.environ.get("VIGILOPS_ENDPOINT", "http://localhost:8080"),
)

model = os.environ.get("OPENAI_MODEL", "deepseek-v4-flash")
provider = os.environ.get("LLM_PROVIDER", "deepseek")

openai_client = vigil_client.wrap_openai(
    openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_ENDPOINT", "https://api.deepseek.com"),
    ),
    provider=provider,
)


# ── tools ─────────────────────────────────────────────────────────────────────

def _safe_path(rel: str) -> Path:
    p = (FIXTURE_DIR / rel).resolve()
    if FIXTURE_DIR.resolve() not in p.parents and p != FIXTURE_DIR.resolve():
        raise ValueError(f"path escapes fixture: {rel}")
    return p


@vigil_client.observe(name="list_dir", step_type="tool_call")
def list_dir(rel: str = ".") -> list:
    base = _safe_path(rel)
    return sorted(
        str(p.relative_to(FIXTURE_DIR))
        for p in base.iterdir()
        if not p.name.startswith("_") or p.name == "__init__.py"
    )


@vigil_client.observe(name="read_file", step_type="tool_call")
def read_file(rel: str) -> str:
    return _safe_path(rel).read_text(encoding="utf-8")


@vigil_client.observe(name="grep", step_type="tool_call")
def grep(pattern: str, rel: str = ".") -> list:
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


# ── agent ─────────────────────────────────────────────────────────────────────

@vigil_client.agent(name="bug-triage-agent")
def run_agent(report: str) -> str:
    MAX_TURNS = 12
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": report},
    ]

    for turn in range(MAX_TURNS):
        resp = openai_client.chat.completions.create(
            model=model,
            max_tokens=2048,
            tools=TOOLS,
            tool_choice="auto",
            messages=messages,
        )

        msg = resp.choices[0].message
        text = msg.content or ""
        tool_calls = msg.tool_calls or []

        # reasoning_content (DeepSeek) / reasoning (o1/o3) auto-emitted
        # as think steps by wrap_openai

        if not tool_calls:
            if not text:
                raise RuntimeError(f"empty final response: {msg!r}")
            return text

        tool_results = []
        for tc in tool_calls:
            name = tc.function.name
            impl = TOOL_IMPL.get(name)
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}

            if impl is None:
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"error": f"unknown tool: {name}"}),
                })
                continue

            # impl is @observe-decorated — tool_call step emitted automatically
            try:
                result = impl(**args)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"result": result}),
                })
            except Exception as e:
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
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ],
        })
        messages.extend(tool_results)
    else:
        raise RuntimeError(f"agent exceeded {MAX_TURNS} turns")


def main() -> None:
    answer = run_agent(FAILING_REPORT)
    print("Triage report:\n" + "-" * 60)
    print(answer)
    print("-" * 60)


if __name__ == "__main__":
    main()
