import os

import pytest
from fastapi.testclient import TestClient

from fastrag.server import app


@pytest.fixture
def test_model_dir():
    return os.path.join(os.path.dirname(__file__), "dummy_model")


@pytest.fixture
def client(monkeypatch, test_model_dir):
    monkeypatch.setenv("CODE_DIR", test_model_dir)
    monkeypatch.setenv("RUNTIME_PARAMS_FILE", os.path.join(test_model_dir, "model-metadata.yaml"))

    with TestClient(app) as c:
        yield c


def test_info(client):
    response = client.get("/info/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid_csv(client):
    csv_content = b"a,promptText\n1,foo"
    files = {"X": ("test.csv", csv_content, "text/csv")}
    response = client.post("/predict/", files=files)
    assert response.status_code == 200
    json_resp = response.json()
    assert "predictions" in json_resp
    assert json_resp["predictions"][0]["predictions"] == "score: foo"


def test_predict_empty_csv(client):
    csv_content = b""
    files = {"X": ("test.csv", csv_content, "text/csv")}
    response = client.post("/predict/", files=files)
    assert response.status_code == 400
    assert "Invalid CSV file" in response.json()["detail"]


def test_chat_valid_json(client):
    response = client.post("/chat/completions", json={"model": "test-model", "messages": []})
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hello form fastrag!"


def test_chat_invalid_json(client):
    response = client.post(
        "/chat/completions", content=b"{invalid_json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


def test_chat_streaming_response(client):
    with client.stream(
        "POST",
        "/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "data: [DONE]" in body
        assert "chat.completion.chunk" in body


def test_chat_async_streaming_response(client):
    async def fake_chat(payload, **kwargs):
        async def _agen():
            yield {"object": "chat.completion.chunk", "choices": [{"delta": {"content": "Echo:"}}]}
            yield {"object": "chat.completion.chunk", "choices": [{"delta": {"content": "hello"}}]}

        return _agen()

    client.app.state.model_adapter.chat = fake_chat

    with client.stream(
        "POST",
        "/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "data: [DONE]" in body
        assert "chat.completion.chunk" in body


def test_predict_contract_violation_returns_422(monkeypatch, test_model_dir):
    monkeypatch.setenv("CODE_DIR", test_model_dir)
    monkeypatch.setenv("TARGET_TYPE", "binary")
    monkeypatch.setenv("POSITIVE_CLASS_LABEL", "Yes")
    monkeypatch.setenv("NEGATIVE_CLASS_LABEL", "No")

    with TestClient(app) as c:
        csv_content = b"a,promptText\n1,foo"
        files = {"X": ("test.csv", csv_content, "text/csv")}
        response = c.post("/predict/", files=files)

    assert response.status_code == 422
    assert response.json()["detail"]
