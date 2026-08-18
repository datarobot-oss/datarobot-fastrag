import asyncio
import json
import os
import time

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from fastrag import server as server_module
from fastrag.prediction_stats import PredictionStatsReporter
from fastrag.prediction_stats import StatsConfig
from fastrag.server import PredictionStatsMiddleware
from fastrag.server import app

DRUM_ENV = {
    "MONITOR": "1",
    "EXTERNAL_WEB_SERVER_URL": "https://app.datarobot.test",
    "API_TOKEN": "token-abc",
    "DEPLOYMENT_ID": "deployment-1",
    "MODEL_ID": "model-1",
}


def wait_for_records(collector, count=1, timeout=2.0):
    """Wait for the reporter's drain task to post, without depending on wall-clock luck."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(collector.records) >= count:
            return collector.records
        time.sleep(0.01)
    raise AssertionError(f"expected {count} record(s), got {collector.records}")


class Collector:
    """Captures the batches a reporter posts."""

    def __init__(self):
        self.batches = []

    def transport(self, *statuses):
        """A transport returning the given statuses in order, then 202 forever."""
        remaining = list(statuses)

        def handler(request):
            status = remaining.pop(0) if remaining else 202
            if status in (200, 202):
                self.batches.append(json.loads(request.content)["data"])
            return httpx.Response(status, text="nope" if status >= 400 else "")

        return httpx.MockTransport(handler)

    @property
    def records(self):
        return [record for batch in self.batches for record in batch]


def make_reporter(collector, *statuses, **overrides):
    config = StatsConfig(
        endpoint="https://app.datarobot.test",
        api_token="token-abc",
        deployment_id="deployment-1",
        model_id="model-1",
        **{"flush_interval_s": 0.05, "retry_backoff_s": 0.0, **overrides},
    )
    client = httpx.AsyncClient(transport=collector.transport(*statuses))
    return PredictionStatsReporter(config, client=client)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_config_is_off_without_monitor(monkeypatch):
    for name in DRUM_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in DRUM_ENV.items():
        if name != "MONITOR":
            monkeypatch.setenv(name, value)

    assert StatsConfig.from_env() is None


def test_config_is_off_when_credentials_are_missing(monkeypatch):
    for name, value in DRUM_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("API_TOKEN")

    assert StatsConfig.from_env() is None


def test_config_reads_the_drum_environment(monkeypatch):
    for name, value in DRUM_ENV.items():
        monkeypatch.setenv(name, value)

    config = StatsConfig.from_env()

    assert config is not None
    assert config.deployment_id == "deployment-1"
    assert config.model_id == "model-1"
    assert config.api_token == "token-abc"
    assert config.url == (
        "https://app.datarobot.test/api/v2/deployments/deployment-1/predictionRequests/fromJSON/"
    )


def test_config_accepts_mlops_prefixed_names(monkeypatch):
    for name in DRUM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MONITOR", "true")
    monkeypatch.setenv("MLOPS_SERVICE_URL", "https://app.datarobot.test/")
    monkeypatch.setenv("MLOPS_API_TOKEN", "token-abc")
    monkeypatch.setenv("MLOPS_DEPLOYMENT_ID", "deployment-2")

    config = StatsConfig.from_env()

    assert config is not None
    assert config.deployment_id == "deployment-2"
    assert config.model_id is None
    # A trailing slash on the endpoint must not double up in the URL.
    assert "datarobot.test/api/v2" in config.url


# --------------------------------------------------------------------------
# reporter
# --------------------------------------------------------------------------


async def test_records_are_batched_and_all_delivered():
    collector = Collector()
    reporter = make_reporter(collector, max_batch=3)
    await reporter.start()

    for _ in range(7):
        reporter.report(num_predictions=1, execution_time_ms=12.5)
    await reporter.aclose()

    assert [len(batch) for batch in collector.batches] == [3, 3, 1]
    assert len(collector.records) == 7
    assert collector.records[0] == {
        "timestamp": collector.records[0]["timestamp"],
        "numPredictions": 1,
        "executionTime": 12.5,
        "userError": False,
        "systemError": False,
        "modelId": "model-1",
    }


async def test_each_record_keeps_the_time_it_was_reported():
    collector = Collector()
    reporter = make_reporter(collector, max_batch=500, flush_interval_s=0.3)
    await reporter.start()

    reporter.report(num_predictions=1, execution_time_ms=1.0)
    await asyncio.sleep(0.05)
    reporter.report(num_predictions=1, execution_time_ms=1.0)
    await reporter.aclose()

    # Both records ship in one batch, but they must not share the flush timestamp.
    assert len(collector.batches) == 1
    first, second = collector.records
    assert first["timestamp"] != second["timestamp"]


async def test_reporting_is_dropped_rather_than_blocking_when_the_queue_is_full():
    collector = Collector()
    reporter = make_reporter(collector, queue_max_size=2)

    for _ in range(5):
        reporter.report(num_predictions=1, execution_time_ms=1.0)

    assert reporter.dropped == 3


async def test_a_failed_post_is_retried():
    collector = Collector()
    reporter = make_reporter(collector, 503, 503, max_batch=1)
    await reporter.start()

    reporter.report(num_predictions=1, execution_time_ms=1.0)
    await reporter.aclose()

    assert len(collector.records) == 1


async def test_records_are_discarded_after_the_last_attempt():
    collector = Collector()
    reporter = make_reporter(collector, 500, 500, 500, max_batch=1)
    await reporter.start()

    reporter.report(num_predictions=1, execution_time_ms=1.0)
    await reporter.aclose()

    assert collector.records == []


async def test_report_after_close_is_ignored():
    collector = Collector()
    reporter = make_reporter(collector)
    await reporter.start()
    await reporter.aclose()

    reporter.report(num_predictions=1, execution_time_ms=1.0)

    assert collector.records == []


# --------------------------------------------------------------------------
# middleware
# --------------------------------------------------------------------------


async def test_streaming_response_is_timed_until_the_last_chunk():
    collector = Collector()
    reporter = make_reporter(collector, max_batch=1)
    await reporter.start()

    streaming_app = FastAPI()

    @streaming_app.post("/chat/completions")
    async def stream():
        async def chunks():
            for index in range(3):
                await asyncio.sleep(0.05)
                yield f"data: chunk-{index}\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")

    streaming_app.state.prediction_stats = reporter
    streaming_app.add_middleware(PredictionStatsMiddleware, endpoints=frozenset({stream}))
    transport = httpx.ASGITransport(app=streaming_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat/completions")

    assert response.text.count("chunk-") == 3
    await reporter.aclose()

    (record,) = collector.records
    # Reporting at response start would have recorded roughly nothing; the whole stream
    # takes about 150ms.
    assert record["executionTime"] > 100
    assert record["numPredictions"] == 1


async def test_reported_when_served_behind_a_proxy_root_path():
    """A deployment can mount the app under a root_path.

    Starlette routes on the path with root_path removed, so a matcher that looks at the
    raw scope path sees /lrs/deployment-1/chat/completions and misses the request even
    though the handler ran. That silently zeroes out Total Predictions.
    """
    collector = Collector()
    reporter = make_reporter(collector, max_batch=1)
    await reporter.start()

    proxied_app = FastAPI()

    @proxied_app.post("/chat/completions")
    async def chat():
        return {"ok": True}

    proxied_app.state.prediction_stats = reporter
    proxied_app.add_middleware(PredictionStatsMiddleware, endpoints=frozenset({chat}))
    transport = httpx.ASGITransport(app=proxied_app, root_path="/lrs/deployment-1")

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/lrs/deployment-1/chat/completions")

    assert response.status_code == 200, response.text
    await reporter.aclose()

    (record,) = collector.records
    assert record["numPredictions"] == 1


@pytest.fixture
def test_model_dir():
    return os.path.join(os.path.dirname(__file__), "dummy_model")


@pytest.fixture
def stats_client(monkeypatch, test_model_dir):
    """The real app, with a reporter whose HTTP calls are captured."""
    monkeypatch.setenv("CODE_DIR", test_model_dir)
    monkeypatch.setenv("RUNTIME_PARAMS_FILE", os.path.join(test_model_dir, "model-metadata.yaml"))

    collector = Collector()

    async def fake_start_reporter():
        reporter = make_reporter(collector, max_batch=500)
        await reporter.start()
        return reporter

    monkeypatch.setattr(server_module, "start_reporter", fake_start_reporter)

    with TestClient(app) as client:
        yield client, collector


def test_chat_request_reports_one_prediction(stats_client):
    client, collector = stats_client

    response = client.post("/chat/completions", json={"model": "test-model", "messages": []})
    assert response.status_code == 200

    (record,) = wait_for_records(collector)
    assert record["numPredictions"] == 1
    assert record["userError"] is False
    assert record["systemError"] is False
    assert record["executionTime"] > 0


def test_predict_reports_one_prediction_per_row(stats_client):
    client, collector = stats_client

    files = {"X": ("test.csv", b"a,promptText\n1,foo\n2,bar\n3,baz", "text/csv")}
    response = client.post("/predict/", files=files)
    assert response.status_code == 200

    (record,) = wait_for_records(collector)
    assert record["numPredictions"] == 3


def test_client_error_is_reported_as_a_user_error(stats_client):
    client, collector = stats_client

    files = {"X": ("test.csv", b"", "text/csv")}
    response = client.post("/predict/", files=files)
    assert response.status_code == 400

    (record,) = wait_for_records(collector)
    assert record["numPredictions"] == 0
    assert record["userError"] is True
    assert record["systemError"] is False


def test_health_checks_are_not_reported(stats_client):
    client, collector = stats_client

    assert client.get("/health/").status_code == 200
    assert client.get("/info/").status_code == 200

    time.sleep(0.2)

    assert collector.records == []
