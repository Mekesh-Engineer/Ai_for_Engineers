import pytest

def test_models_endpoint(client):
    res = client.get("/api/models/")
    assert res.status_code == 200
    data = res.get_json()
    assert "default_model" in data
    assert "local_models" in data

def test_models_health_endpoint(client):
    res = client.get("/api/models/health")
    assert res.status_code in [200, 503]
    data = res.get_json()
    assert "status" in data

def test_chat_stream_endpoint(client):
    # Test streaming SSE initiation
    res = client.post("/api/chat/stream", json={"prompt": "Hello VOLTIX", "enable_rag": False})
    assert res.status_code == 200
    assert "text/event-stream" in res.content_type
