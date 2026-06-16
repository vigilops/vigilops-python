import os

import httpx
import pytest
import pytest_asyncio

from vigilops import AsyncVigil, Vigil

_ENDPOINT = os.getenv("VIGILOPS_ENDPOINT", "http://localhost:8080")
_ADMIN_BASE = f"{_ENDPOINT}/v1/admin"


def _provision_key(project_name: str) -> tuple[str, str]:
    """Create a throwaway project + key. Returns (project_id, api_key)."""
    with httpx.Client(timeout=5.0) as h:
        proj = h.post(f"{_ADMIN_BASE}/projects", json={"name": project_name})
        proj.raise_for_status()
        project_id = proj.json()["data"]["id"]

        key_resp = h.post(
            f"{_ADMIN_BASE}/projects/{project_id}/keys",
            json={"name": "test"},
        )
        key_resp.raise_for_status()
        api_key = key_resp.json()["data"]["key"]

    return project_id, api_key


def _delete_project(project_id: str) -> None:
    with httpx.Client(timeout=5.0) as h:
        h.delete(f"{_ADMIN_BASE}/projects/{project_id}")


def _server_reachable() -> bool:
    try:
        r = httpx.get(f"{_ENDPOINT}/v1/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


# ── session-scoped project fixtures ───────────────────────────────────────────
# One project per test session to avoid FK-violation batch failures that occur
# when per-test project deletion races with the server's 500ms batch flush.


@pytest.fixture(scope="session")
def _sync_project():
    if not _server_reachable():
        pytest.skip(
            "vigil server not reachable — run `make db-up && make run` in core/"
        )
    project_id, api_key = _provision_key("pytest-sync-session")
    yield api_key
    _delete_project(project_id)


@pytest.fixture(scope="session")
def _async_project():
    if not _server_reachable():
        pytest.skip(
            "vigil server not reachable — run `make db-up && make run` in core/"
        )
    project_id, api_key = _provision_key("pytest-async-session")
    yield api_key
    _delete_project(project_id)


# ── per-test client fixtures (function scope, reuse session project) ──────────


@pytest.fixture
def client(_sync_project):
    with Vigil(api_key=_sync_project, endpoint=_ENDPOINT) as c:
        yield c


@pytest_asyncio.fixture
async def async_client(_async_project):
    async with AsyncVigil(api_key=_async_project, endpoint=_ENDPOINT) as c:
        yield c
