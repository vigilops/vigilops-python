import os

import pytest

from vigil import Vigil, VigilAuthError, VigilTransportError, VigilValidationError


def test_health_returns_ok(client):
    body = client.health()
    assert body["status"] == "ok"


def test_ingest_ai_minimum_payload(client):
    body = client.ingest_ai(model="claude-opus-4-7", status="success")
    assert "id" in body
    assert "timestamp" in body


def test_ingest_ai_with_all_optional_fields(client):
    body = client.ingest_ai(
        model="gpt-4o",
        status="success",
        provider="openai",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cost_usd=0.0023,
        latency_ms=420,
    )
    assert "id" in body


def test_ingest_ai_raises_auth_error_on_bad_key():
    endpoint = os.getenv("VIGIL_ENDPOINT", "http://localhost:8080")
    with Vigil(api_key="vgl_obviously_wrong", endpoint=endpoint) as c:
        with pytest.raises(VigilAuthError):
            c.ingest_ai(model="m", status="success")


def test_ingest_ai_raises_validation_error_on_bad_status(client):
    with pytest.raises(VigilValidationError):
        client.ingest_ai(model="m", status="not-a-valid-status")


def test_health_raises_transport_error_on_unreachable_host():
    with Vigil(api_key="vgl_x", endpoint="http://127.0.0.1:1") as c:
        with pytest.raises(VigilTransportError):
            c.health()
