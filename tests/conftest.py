import os
import pytest
import pytest_asyncio

from vigil import (Vigil, AsyncVigil)

@pytest.fixture
def client():
    api_key = os.getenv("VIGIL_API_KEY")
    if not api_key:
        pytest.skip("VIGIL_API_KEY not set — run `make seed` in the vigil repo and export it")
    with Vigil(api_key=api_key, endpoint=os.getenv("VIGIL_ENDPOINT", "http://localhost:8080")) as c:
        yield c

    

@pytest_asyncio.fixture
async def async_client():
    api_key = os.getenv("VIGIL_API_KEY")
    if not api_key:
        pytest.skip("VIGIL_API_KEY not set")
    endpoint = os.getenv("VIGIL_ENDPOINT", "http://localhost:8080")
    async with AsyncVigil(api_key=api_key, endpoint=endpoint) as c:
        yield c
