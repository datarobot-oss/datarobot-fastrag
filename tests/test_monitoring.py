import asyncio
import os
import sys
import time
import types

import pytest
from fastapi.testclient import TestClient

from fastrag.monitoring import MLOpsReporter
from fastrag.monitoring import elapsed_ms
from fastrag.server import _format_chat_response
from fastrag.server import _meter_async
from fastrag.server import _meter_sync
from fastrag.server import app


@pytest.fixture
def test_model_dir():
    return os.path.join(os.path.dirname(__file__), "dummy_model")


class _RecordingReporter:
    """Stand-in for MLOpsReporter that records what the server reports."""

    def __init__(self):
        self.chat = 0
        self.predictions: list[int] = []
        self.errors = 0

    def report_chat(self, execution_time_ms):
        self.chat += 1

    def report_predictions(self, num_predictions, execution_time_ms):
        self.predictions.append(num_predictions)

    def report_error(self, execution_time_ms):
        self.errors += 1


@pytest.fixture
def client_and_reporter(monkeypatch, test_model_dir):
    monkeypatch.setenv("CODE_DIR", test_model_dir)
    monkeypatch.setenv("RUNTIME_PARAMS_FILE", os.path.join(test_model_dir, "model-metadata.yaml"))
    reporter = _RecordingReporter()
    with TestClient(app) as c:
        # Lifespan installs a (disabled) real reporter; swap in the recorder.
        c.app.state.mlops_reporter = reporter
        yield c, reporter


# --------------------------------------------------------------------------- #
# Server integration: predictions are counted for chat + structured scoring.   #
# --------------------------------------------------------------------------- #


def test_chat_non_streaming_reports_one_prediction(client_and_reporter):
    client, reporter = client_and_reporter
    resp = client.post("/chat/completions", json={"model": "m", "messages": []})
    assert resp.status_code == 200
    assert reporter.chat == 1
    assert reporter.errors == 0


def test_chat_streaming_reports_one_after_drain(client_and_reporter):
    client, reporter = client_and_reporter
    with client.stream(
        "POST",
        "/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as resp:
        body = "".join(resp.iter_text())
        assert resp.status_code == 200
    assert "data: [DONE]" in body
    # Reported only after the stream is fully consumed.
    assert reporter.chat == 1
    assert reporter.errors == 0


def test_predict_reports_row_count(client_and_reporter):
    client, reporter = client_and_reporter
    csv_content = b"a,promptText\n1,foo\n2,bar"
    files = {"X": ("test.csv", csv_content, "text/csv")}
    resp = client.post("/predict/", files=files)
    assert resp.status_code == 200
    assert reporter.predictions == [2]
    assert reporter.errors == 0


def test_chat_error_reports_error(client_and_reporter):
    client, reporter = client_and_reporter

    async def boom(payload, **kwargs):
        raise RuntimeError("upstream exploded")

    client.app.state.model_adapter.chat = boom
    resp = client.post("/chat/completions", json={"model": "m", "messages": []})
    assert resp.status_code == 500
    assert reporter.errors == 1
    assert reporter.chat == 0


def test_predict_error_reports_error(client_and_reporter):
    client, reporter = client_and_reporter

    async def boom(data, **kwargs):
        raise RuntimeError("scoring failed")

    client.app.state.model_adapter.score = boom
    files = {"X": ("test.csv", b"a,promptText\n1,foo", "text/csv")}
    resp = client.post("/predict/", files=files)
    assert resp.status_code == 500
    assert reporter.errors == 1
    assert reporter.predictions == []


# --------------------------------------------------------------------------- #
# Metering wrappers fire the right callback on drain vs. mid-stream failure.     #
# --------------------------------------------------------------------------- #


def test_meter_sync_reports_on_complete():
    calls = {"complete": 0, "error": 0}

    def lines():
        yield "data: a\n"
        yield "data: b\n"

    out = list(
        _meter_sync(
            lines(),
            lambda: calls.__setitem__("complete", calls["complete"] + 1),
            lambda: calls.__setitem__("error", calls["error"] + 1),
        )
    )
    assert out == ["data: a\n", "data: b\n"]  # passed through unchanged
    assert calls == {"complete": 1, "error": 0}


def test_meter_sync_reports_on_error():
    calls = {"complete": 0, "error": 0}

    def lines():
        yield "data: a\n"
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        list(
            _meter_sync(
                lines(),
                lambda: calls.__setitem__("complete", calls["complete"] + 1),
                lambda: calls.__setitem__("error", calls["error"] + 1),
            )
        )
    assert calls == {"complete": 0, "error": 1}


def test_meter_async_reports_on_complete():
    calls = {"complete": 0, "error": 0}

    async def lines():
        yield "data: a\n"

    async def drain():
        return [
            line
            async for line in _meter_async(
                lines(),
                lambda: calls.__setitem__("complete", calls["complete"] + 1),
                lambda: calls.__setitem__("error", calls["error"] + 1),
            )
        ]

    out = asyncio.run(drain())
    assert out == ["data: a\n"]
    assert calls == {"complete": 1, "error": 0}


def test_non_streaming_reports_error_when_serialization_fails():
    calls = {"complete": 0, "error": 0}

    class _BadResponse:
        object = "chat.completion"  # routes to the non-streaming path

        def model_dump(self):
            raise RuntimeError("cannot serialize")

    with pytest.raises(RuntimeError):
        _format_chat_response(
            _BadResponse(),
            on_complete=lambda: calls.__setitem__("complete", calls["complete"] + 1),
            on_error=lambda: calls.__setitem__("error", calls["error"] + 1),
        )
    # Failed conversion must count as an error, never a success.
    assert calls == {"complete": 0, "error": 1}


def test_non_streaming_reports_success_after_serialization():
    calls = {"complete": 0, "error": 0}
    payload = _format_chat_response(
        {"object": "chat.completion", "choices": []},
        on_complete=lambda: calls.__setitem__("complete", calls["complete"] + 1),
        on_error=lambda: calls.__setitem__("error", calls["error"] + 1),
    )
    assert payload == {"object": "chat.completion", "choices": []}
    assert calls == {"complete": 1, "error": 0}


def test_meter_async_reports_on_error():
    calls = {"complete": 0, "error": 0}

    async def lines():
        yield "data: a\n"
        raise RuntimeError("boom")

    async def drain():
        return [
            line
            async for line in _meter_async(
                lines(),
                lambda: calls.__setitem__("complete", calls["complete"] + 1),
                lambda: calls.__setitem__("error", calls["error"] + 1),
            )
        ]

    with pytest.raises(RuntimeError):
        asyncio.run(drain())
    assert calls == {"complete": 0, "error": 1}


# --------------------------------------------------------------------------- #
# MLOpsReporter lifecycle: enable gating, spooler wiring, graceful disable.     #
# --------------------------------------------------------------------------- #

_MLOPS_ENV = [
    "MLOPS_DEPLOYMENT_ID",
    "DEPLOYMENT_ID",
    "MLOPS_MODEL_ID",
    "MODEL_ID",
    "EXTERNAL_WEB_SERVER_URL",
    "MLOPS_SERVICE_URL",
    "DATAROBOT_ENDPOINT",
    "API_TOKEN",
    "MLOPS_API_TOKEN",
    "DATAROBOT_API_TOKEN",
    "MLOPS_SPOOLER_TYPE",
    "MLOPS_MONITORING_DISABLED",
    "MONITOR",
]


@pytest.fixture
def clean_mlops_env(monkeypatch):
    for name in _MLOPS_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _install_fake_mlops(monkeypatch, recorder):
    fake_pkg = types.ModuleType("datarobot_mlops")
    fake_mod = types.ModuleType("datarobot_mlops.mlops")

    class FakeMLOps:
        def __init__(self):
            recorder["created"] = True

        def set_deployment_id(self, deployment_id):
            recorder["deployment_id"] = deployment_id
            return self

        def set_model_id(self, model_id):
            recorder["model_id"] = model_id
            return self

        def set_api_spooler(self, mlops_service_url=None, mlops_api_token=None):
            recorder["spooler"] = ("api", mlops_service_url, mlops_api_token)
            return self

        def set_async_reporting(self, async_reporting=True):
            recorder["async"] = async_reporting
            return self

        def init(self):
            recorder["init"] = True
            return self

        def report_deployment_stats(self, num_predictions, execution_time_ms, **kwargs):
            assert isinstance(num_predictions, int)
            assert isinstance(execution_time_ms, float)
            recorder["stats"].append((num_predictions, execution_time_ms))

        def shutdown(self, timeout_sec=0):
            recorder["shutdown"] = True

    fake_mod.MLOps = FakeMLOps
    fake_pkg.mlops = fake_mod
    monkeypatch.setitem(sys.modules, "datarobot_mlops", fake_pkg)
    monkeypatch.setitem(sys.modules, "datarobot_mlops.mlops", fake_mod)


def test_disabled_without_deployment_id(clean_mlops_env):
    reporter = MLOpsReporter()
    reporter.initialize()
    assert reporter.enabled is False
    # Reporting on a disabled reporter is a silent no-op.
    reporter.report_chat(1.0)
    reporter.report_predictions(5, 1.0)
    reporter.report_error(1.0)
    reporter.shutdown()


def test_disabled_by_kill_switch(clean_mlops_env):
    clean_mlops_env.setenv("MLOPS_DEPLOYMENT_ID", "dep123")
    clean_mlops_env.setenv("MLOPS_MONITORING_DISABLED", "true")
    reporter = MLOpsReporter()
    reporter.initialize()
    assert reporter.enabled is False


def test_disabled_by_monitor_false(clean_mlops_env):
    clean_mlops_env.setenv("MLOPS_DEPLOYMENT_ID", "dep123")
    clean_mlops_env.setenv("MONITOR", "false")
    reporter = MLOpsReporter()
    reporter.initialize()
    assert reporter.enabled is False


def test_initialize_wires_api_spooler_and_reports(clean_mlops_env):
    recorder = {"stats": []}
    _install_fake_mlops(clean_mlops_env, recorder)
    clean_mlops_env.setenv("MLOPS_DEPLOYMENT_ID", "dep123")
    clean_mlops_env.setenv("MLOPS_MODEL_ID", "mod456")
    clean_mlops_env.setenv("EXTERNAL_WEB_SERVER_URL", "https://app.datarobot.com")
    clean_mlops_env.setenv("API_TOKEN", "secret-token")

    reporter = MLOpsReporter()
    reporter.initialize()
    assert reporter.enabled is True
    assert recorder["deployment_id"] == "dep123"
    assert recorder["model_id"] == "mod456"
    assert recorder["spooler"] == ("api", "https://app.datarobot.com", "secret-token")
    assert recorder["async"] is True
    assert recorder["init"] is True

    reporter.report_chat(12.5)
    reporter.report_predictions(3, 4.0)
    reporter.report_error(1.0)
    assert recorder["stats"] == [(1, 12.5), (3, 4.0), (0, 1.0)]

    reporter.shutdown()
    assert recorder["shutdown"] is True
    assert reporter.enabled is False


def test_initialize_accepts_env_spooler(clean_mlops_env):
    recorder = {"stats": []}
    _install_fake_mlops(clean_mlops_env, recorder)
    clean_mlops_env.setenv("MLOPS_DEPLOYMENT_ID", "dep123")
    clean_mlops_env.setenv("MLOPS_SPOOLER_TYPE", "api")

    reporter = MLOpsReporter()
    reporter.initialize()
    assert reporter.enabled is True
    assert recorder.get("spooler") is None  # env-configured, set_api_spooler not called
    assert recorder["async"] is True


def test_initialize_disables_without_spooler_config(clean_mlops_env):
    recorder = {"stats": []}
    _install_fake_mlops(clean_mlops_env, recorder)
    clean_mlops_env.setenv("MLOPS_DEPLOYMENT_ID", "dep123")
    # No spooler env at all -> initialize must disable gracefully, not crash.

    reporter = MLOpsReporter()
    reporter.initialize()
    assert reporter.enabled is False


def test_elapsed_ms_is_nonnegative_float():
    start = time.monotonic()
    value = elapsed_ms(start)
    assert isinstance(value, float)
    assert value >= 0.0
