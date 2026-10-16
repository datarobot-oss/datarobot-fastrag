import asyncio
import json
import os
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from fastrag import server as server_module
from fastrag.prediction_data import PredictionDataReporter
from fastrag.prediction_data import chat_prompt_text
from fastrag.prediction_stats import StatsConfig
from fastrag.server import app

BASE = "https://app.datarobot.test/api/v2/deployments/deployment-1"
DATA_URL = f"{BASE}/predictionInputs/fromJSON/"


class FakeDataRobot:
    """Serves the deployment and settings lookups and records prediction data posts."""

    def __init__(self, collection=True, drift=False, prompt="question", statuses=()):
        self.collection = collection
        self.drift = drift
        self.prompt = prompt
        self.statuses = list(statuses)
        self.posts = []

    def handler(self, request):
        url = str(request.url)
        if request.method == "GET" and url == f"{BASE}/settings/":
            return httpx.Response(
                200,
                json={
                    "predictionsDataCollection": {"enabled": self.collection},
                    "featureDrift": {"enabled": self.drift},
                },
            )
        if request.method == "GET" and url == f"{BASE}/":
            return httpx.Response(200, json={"model": {"id": "model-x", "prompt": self.prompt}})
        if request.method == "POST" and url == DATA_URL:
            status = self.statuses.pop(0) if self.statuses else 202
            if status in (200, 202):
                self.posts.append(json.loads(request.content))
            return httpx.Response(status, text="nope" if status >= 400 else "")
        return httpx.Response(404)

    @property
    def rows(self):
        return [item for post in self.posts for item in post["data"]]


def make_reporter(fake, model_id="model-1", **overrides):
    config = StatsConfig(
        endpoint="https://app.datarobot.test",
        api_token="token-abc",
        deployment_id="deployment-1",
        model_id=model_id,
        **{"flush_interval_s": 0.05, "retry_backoff_s": 0.0, **overrides},
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return PredictionDataReporter(config, client=client)


def wait_for_rows(fake, count=1, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(fake.rows) >= count:
            return fake.rows
        time.sleep(0.01)
    raise AssertionError(f"expected {count} row(s), got {fake.rows}")


def test_chat_prompt_text_uses_last_message():
    params = {"messages": [{"role": "user", "content": "first"}, {"role": "user", "content": "hi"}]}
    assert chat_prompt_text(params) == "hi"
    assert chat_prompt_text({"messages": []}) is None


def test_chat_prompt_text_flattens_content_parts():
    content = [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "https://x/img.png"}},
    ]
    params = {"messages": [{"role": "user", "content": content}]}
    assert chat_prompt_text(params) == "describe\nImage URL: https://x/img.png"


async def test_reports_row_in_drum_payload_format():
    fake = FakeDataRobot()
    reporter = make_reporter(fake)
    await reporter.start()
    reporter.report(prompt="What is the policy?", response="30 days", association_id="a-1")
    await reporter.aclose()

    (post,) = fake.posts
    assert post["skipDriftTracking"] is False
    assert post["skipAccuracyTracking"] is False
    (row,) = post["data"]
    assert row["modelId"] == "model-1"
    assert row["features"] == [{"name": "question", "values": ["What is the policy?"]}]
    assert row["predictions"] == ["30 days"]
    assert row["associationIds"] == ["a-1"]
    assert row["timestamp"]


async def test_falls_back_to_deployment_model_id_and_default_prompt_column():
    fake = FakeDataRobot(prompt=None)
    reporter = make_reporter(fake, model_id=None)
    await reporter.start()
    reporter.report(prompt="hi", response="hello", association_id="a-1")
    await reporter.aclose()

    (row,) = fake.rows
    assert row["modelId"] == "model-x"
    assert row["features"][0]["name"] == "promptText"


async def test_feature_drift_alone_enables_reporting():
    fake = FakeDataRobot(collection=False, drift=True)
    reporter = make_reporter(fake)
    await reporter.start()
    assert reporter.enabled


async def test_nothing_is_reported_when_collection_and_drift_are_off():
    fake = FakeDataRobot(collection=False, drift=False)
    reporter = make_reporter(fake)
    await reporter.start()
    reporter.report(prompt="hi", response="hello", association_id="a-1")
    await reporter.aclose()

    assert not reporter.enabled
    assert fake.posts == []


async def test_settings_lookup_failure_keeps_reporting_off():
    def handler(request):
        return httpx.Response(500)

    config = StatsConfig(endpoint="https://app.datarobot.test", api_token="t", deployment_id="d")
    reporter = PredictionDataReporter(
        config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    await reporter.start()
    assert not reporter.enabled
    await reporter.aclose()


async def test_settings_are_refreshed():
    fake = FakeDataRobot(collection=False)
    reporter = make_reporter(fake)
    reporter._settings_refresh_s = 0.01
    await reporter.start()
    assert not reporter.enabled

    fake.collection = True
    for _ in range(100):
        if reporter.enabled:
            break
        await asyncio.sleep(0.01)
    assert reporter.enabled
    await reporter.aclose()


async def test_retries_then_posts():
    fake = FakeDataRobot(statuses=[503])
    reporter = make_reporter(fake)
    await reporter.start()
    reporter.report(prompt="hi", response="hello", association_id="a-1")
    await reporter.aclose()

    assert len(fake.rows) == 1


@pytest.fixture
def data_client(monkeypatch):
    test_model_dir = os.path.join(os.path.dirname(__file__), "dummy_model")
    monkeypatch.setenv("CODE_DIR", test_model_dir)
    monkeypatch.setenv("RUNTIME_PARAMS_FILE", os.path.join(test_model_dir, "model-metadata.yaml"))
    fake = FakeDataRobot()

    async def fake_start_data_reporter():
        reporter = make_reporter(fake)
        await reporter.start()
        return reporter

    monkeypatch.setattr(server_module, "start_data_reporter", fake_start_data_reporter)
    with TestClient(app) as client:
        yield client, fake


def test_chat_reports_prompt_response_and_association_id(data_client):
    client, fake = data_client
    response = client.post(
        "/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200

    (row,) = wait_for_rows(fake)
    assert row["features"][0]["values"] == ["hello"]
    assert row["predictions"] == [response.json()["choices"][0]["message"]["content"]]
    assert row["associationIds"] == [response.json()["datarobot_association_id"]]


def test_streamed_chat_reports_joined_response(data_client):
    client, fake = data_client

    async def fake_chat(payload, **kwargs):
        async def gen():
            for text in ("Hel", "lo"):
                yield {"object": "chat.completion.chunk", "choices": [{"delta": {"content": text}}]}

        return gen()

    client.app.state.model_adapter.chat = fake_chat
    with client.stream(
        "POST",
        "/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "datarobot_association_id": "caller-id",
        },
    ) as response:
        "".join(response.iter_text())

    (row,) = wait_for_rows(fake)
    assert row["predictions"] == ["Hello"]
    assert row["associationIds"] == ["caller-id"]


def test_sync_streamed_chat_is_reported_from_worker_thread(data_client):
    client, fake = data_client
    # dummy_model's chat returns a sync generator, which Starlette iterates in a worker thread.
    with client.stream(
        "POST",
        "/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        "".join(response.iter_text())

    (row,) = wait_for_rows(fake)
    assert row["features"][0]["values"] == ["hi"]
    assert row["predictions"] == ["my.chat with prompt: hi"]


def test_failed_stream_is_not_reported(data_client):
    client, fake = data_client

    async def fake_chat(payload, **kwargs):
        async def gen():
            yield {"object": "chat.completion.chunk", "choices": [{"delta": {"content": "x"}}]}
            raise RuntimeError("boom")

        return gen()

    client.app.state.model_adapter.chat = fake_chat
    with client.stream(
        "POST",
        "/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        "".join(response.iter_text())

    time.sleep(0.2)
    assert fake.rows == []
