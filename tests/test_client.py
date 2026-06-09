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
