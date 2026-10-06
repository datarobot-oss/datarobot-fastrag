import json
import os

import pandas as pd
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


def test_chat_forwards_request_headers_to_hook(client):
    captured_kwargs = {}

    async def fake_chat(payload, **kwargs):
        captured_kwargs.update(kwargs)
        return {
            "id": "association_id",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
        }

    client.app.state.model_adapter.chat = fake_chat

    response = client.post(
        "/chat/completions",
        json={"model": "test-model", "messages": []},
        headers={"X-DataRobot-Identity-Token": "test-token"},
    )
    assert response.status_code == 200
    assert captured_kwargs["headers"]["X-DataRobot-Identity-Token"] == "test-token"
    assert captured_kwargs["headers"]["x-datarobot-identity-token"] == "test-token"


def test_predict_forwards_request_headers_to_hook(client):
    captured_kwargs = {}

    async def fake_score(data, **kwargs):
        captured_kwargs.update(kwargs)
        return pd.DataFrame({"predictions": ["score: foo"]})

    client.app.state.model_adapter.score = fake_score

    csv_content = b"a,promptText\n1,foo"
    files = {"X": ("test.csv", csv_content, "text/csv")}
    response = client.post(
        "/predict/", files=files, headers={"X-DataRobot-Identity-Token": "test-token"}
    )
    assert response.status_code == 200
    assert captured_kwargs["headers"]["X-DataRobot-Identity-Token"] == "test-token"
    assert captured_kwargs["headers"]["x-datarobot-identity-token"] == "test-token"


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


def _stream_chunk(content):
    return {"object": "chat.completion.chunk", "choices": [{"delta": {"content": content}}]}


def _read_stream(client):
    with client.stream(
        "POST",
        "/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        return response, "".join(response.iter_text())


def _assert_error_event_then_done(body):
    events = [e for e in body.split("\n\n") if e]
    assert events[-1] == "data: [DONE]"
    error = json.loads(events[-2].removeprefix("data: "))["error"]
    assert error["type"] == "server_error"
    assert "boom" not in body  # real cause is logged, not leaked to the client
    return events


@pytest.mark.parametrize("is_async", [False, True])
def test_chat_stream_failure_emits_error_event_and_done(client, caplog, is_async):
    async def fake_chat(payload, **kwargs):
        if is_async:

            async def agen():
                yield _stream_chunk("Echo:")
                yield _stream_chunk("hi")
                raise RuntimeError("boom")

            return agen()

        def gen():
            yield _stream_chunk("Echo:")
            yield _stream_chunk("hi")
            raise RuntimeError("boom")

        return gen()

    client.app.state.model_adapter.chat = fake_chat

    with caplog.at_level("ERROR", logger="fastrag.server"):
        response, body = _read_stream(client)

    assert response.status_code == 200
    events = _assert_error_event_then_done(body)
    assert len(events) == 4  # two chunks, error event, [DONE]
    assert "Echo:" in events[0]
    assert any("failed mid-stream" in r.getMessage() and r.exc_info for r in caplog.records)


def test_chat_stream_failure_before_first_chunk_emits_error_event(client):
    async def fake_chat(payload, **kwargs):
        def gen():
            raise RuntimeError("boom")
            yield

        return gen()

    client.app.state.model_adapter.chat = fake_chat

    response, body = _read_stream(client)
    assert response.status_code == 200
    events = _assert_error_event_then_done(body)
    assert len(events) == 2


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
