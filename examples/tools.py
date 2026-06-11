"""Real tools for the research agent demo.

`web_search` tries multiple ddgs backends with bounded retries so the demo
survives the rate-limit / decode failures that hit a single backend.
"""

import time

from ddgs import DDGS
from ddgs.exceptions import DDGSException


# Order matters — start with the one historically most reliable for English text.
_BACKENDS = ("duckduckgo", "brave", "bing", "google")


def web_search(q: str, max_results: int = 10) -> list[dict]:
    """Search the web. Returns a list of {title, url, snippet} dicts.

    Tries each backend up to two times before giving up. Raises the last
    exception so the caller (the agent) can record a failed tool_call.
    """
    last_err: Exception | None = None
    for backend in _BACKENDS:
        for attempt in range(2):
            try:
                with DDGS(timeout=10) as ddg:
                    rows = list(ddg.text(q, max_results=max_results, backend=backend))
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                    for r in rows
                ]
            except DDGSException as e:
                last_err = e
                if attempt == 0:
                    time.sleep(0.5)
                continue
    raise RuntimeError(f"web_search failed across all backends; last: {last_err}")
