"""GET /models: target-type gating and the OpenAI list shape.

The chat route serves both textgeneration and agenticworkflow models, so the
models route that supports it (OpenAI ``models.list()``) must accept the same
target types. Anything else stays a 404.
"""

import json
import os

import openai
import pytest
from fastapi.testclient import TestClient

from fastrag.runtime_parameters import RuntimeParameters
from fastrag.server import app

DUMMY_MODEL_DIR = os.path.join(os.path.dirname(__file__), "dummy_model")

MODEL = {"id": "m1", "object": "model", "created": 1, "owned_by": "me"}

HOOK_RESULTS = {
    "async-page": """
from openai.pagination import AsyncPage
from openai.types.model import Model

async def get_supported_llm_models(model):
    return AsyncPage[Model](data=[Model(**MODEL)], object="list")
""",
    "sync-page": """
from openai.pagination import SyncPage
from openai.types.model import Model

def get_supported_llm_models(model):
    return SyncPage[Model](data=[Model(**MODEL)], object="list")
""",
    "list-of-dicts": """
def get_supported_llm_models(model):
    return [MODEL]
""",
    "pre-wrapped-dict": """
def get_supported_llm_models(model):
    return {"object": "list", "data": [MODEL]}
""",
}

NO_MODELS_HOOK = """
def load_model(code_dir):
    return None
"""


def _client(monkeypatch, target_type: str, code_dir: str = DUMMY_MODEL_DIR) -> TestClient:
    monkeypatch.setenv("CODE_DIR", code_dir)
    monkeypatch.setenv("TARGET_TYPE", target_type)
    return TestClient(app)


def _code_dir(tmp_path, custom_py: str) -> str:
    (tmp_path / "custom.py").write_text(custom_py)
    return str(tmp_path)


@pytest.mark.parametrize("target_type", ["textgeneration", "agenticworkflow"])
@pytest.mark.parametrize("path", ["/models", "/v1/models"])
def test_models_served_for_llm_target_types(monkeypatch, target_type, path):
    with _client(monkeypatch, target_type) as c:
        response = c.get(path)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["datarobot_llm_id"]


def test_openai_client_lists_models(monkeypatch):
    with _client(monkeypatch, "textgeneration") as c:
        client = openai.OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=c)
        models = client.models.list()

    assert [m.id for m in models] == ["datarobot_llm_id"]


@pytest.mark.parametrize("hook", HOOK_RESULTS.values(), ids=HOOK_RESULTS.keys())
def test_hook_result_shapes_become_models_list(monkeypatch, tmp_path, hook):
    custom_py = f"MODEL = {MODEL!r}\n{hook}"
    with _client(monkeypatch, "textgeneration", _code_dir(tmp_path, custom_py)) as c:
        response = c.get("/models")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["m1"]


@pytest.mark.parametrize("result", [None, "m1", {"object": "list"}], ids=["none", "str", "no-data"])
def test_unsupported_hook_result_is_500(monkeypatch, tmp_path, result):
    custom_py = f"def get_supported_llm_models(model):\n    return {result!r}\n"
    with _client(monkeypatch, "textgeneration", _code_dir(tmp_path, custom_py)) as c:
        response = c.get("/models")

    assert response.status_code == 500
    assert response.json() == {"detail": "Models endpoint failed."}


@pytest.mark.parametrize("llm_id", ["my-llm", None])
def test_models_without_hook_fall_back_to_llm_id(monkeypatch, tmp_path, llm_id):
    if llm_id is not None:
        monkeypatch.setenv(
            RuntimeParameters.namespaced_param_name("LLM_ID"),
            json.dumps({"type": "string", "payload": llm_id}),
        )
    expected = []
    if llm_id is not None:
        expected = [
            {"id": llm_id, "object": "model", "created": 1735689600, "owned_by": "DataRobot"}
        ]

    with _client(monkeypatch, "textgeneration", _code_dir(tmp_path, NO_MODELS_HOOK)) as c:
        response = c.get("/models")

    assert response.status_code == 200, response.text
    assert response.json() == {"object": "list", "data": expected}


@pytest.mark.parametrize("payload", [None, ""])
def test_models_without_hook_skip_empty_llm_id(monkeypatch, tmp_path, payload):
    monkeypatch.setenv(
        RuntimeParameters.namespaced_param_name("LLM_ID"),
        json.dumps({"type": "string", "payload": payload}),
    )

    with _client(monkeypatch, "textgeneration", _code_dir(tmp_path, NO_MODELS_HOOK)) as c:
        response = c.get("/models")

    assert response.status_code == 200, response.text
    assert response.json() == {"object": "list", "data": []}


def test_models_rejected_for_other_target_types(monkeypatch):
    with _client(monkeypatch, "regression") as c:
        response = c.get("/models")

    assert response.status_code == 404
    assert "supported only for" in response.json()["detail"]
