"""GET /models target-type gating.

The chat route serves both textgeneration and agenticworkflow models, so the
models route that supports it (OpenAI ``models.list()``) must accept the same
target types. Anything else stays a 404.
"""

import os

import pytest
from fastapi.testclient import TestClient

from fastrag.server import app

DUMMY_MODEL_DIR = os.path.join(os.path.dirname(__file__), "dummy_model")


def _client(monkeypatch, target_type: str) -> TestClient:
    monkeypatch.setenv("CODE_DIR", DUMMY_MODEL_DIR)
    monkeypatch.setenv("TARGET_TYPE", target_type)
    return TestClient(app)


@pytest.mark.parametrize("target_type", ["textgeneration", "agenticworkflow"])
@pytest.mark.parametrize("path", ["/models", "/v1/models"])
def test_models_served_for_llm_target_types(monkeypatch, target_type, path):
    with _client(monkeypatch, target_type) as c:
        response = c.get(path)

    assert response.status_code == 200, response.text
    # fastrag returns the hook's list as-is (see _to_jsonable), not a wrapped object.
    assert [m["id"] for m in response.json()] == ["datarobot_llm_id"]


def test_models_rejected_for_other_target_types(monkeypatch):
    with _client(monkeypatch, "regression") as c:
        response = c.get("/models")

    assert response.status_code == 404
    assert "supported only for" in response.json()["detail"]
