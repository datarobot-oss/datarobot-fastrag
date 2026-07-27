import os

import pytest
from fastapi.testclient import TestClient

from fastrag.server import app

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

_SCORE_CASES = [
    {
        "target": "anomaly",
        "env": {"TARGET_TYPE": "anomaly"},
        "csv": b"a,b\n1,2\n3,4",
    },
    {
        "target": "binary",
        "env": {
            "TARGET_TYPE": "binary",
            "POSITIVE_CLASS_LABEL": "yes",
            "NEGATIVE_CLASS_LABEL": "no",
        },
        "csv": b"a,b\n1,2",
    },
    {
        "target": "multiclass",
        "env": {
            "TARGET_TYPE": "multiclass",
            "CLASS_LABELS": "a,b,c",
        },
        "csv": b"a,b\n1,2",
    },
    {
        "target": "regression",
        "env": {"TARGET_TYPE": "regression"},
        "csv": b"a,b\n1,2",
    },
    {
        "target": "textgen",
        "env": {"TARGET_TYPE": "textgeneration"},
        "csv": b"input\nhello",
    },
]

MODEL_CASES = []
for case in _SCORE_CASES:
    for suffix, label in [("", "sync"), ("_async", "async")]:
        MODEL_CASES.append(
            {
                "id": f"{case['target']}_{label}",
                "dir": f"python3_dummy_{case['target']}{suffix}",
                "env": case["env"],
                "csv": case["csv"],
            }
        )


@pytest.fixture(params=MODEL_CASES, ids=lambda c: c["id"])
def model_client(request, monkeypatch):
    case = request.param
    model_dir = os.path.join(MODELS_DIR, case["dir"])
    monkeypatch.setenv("CODE_DIR", model_dir)
    for k, v in case["env"].items():
        monkeypatch.setenv(k, v)

    with TestClient(app) as c:
        yield c, case


def test_info_per_model(model_client):
    client, _ = model_client
    response = client.get("/info/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_per_model(model_client):
    client, case = model_client
    files = {"X": ("test.csv", case["csv"], "text/csv")}
    response = client.post("/predict/", files=files)
    assert response.status_code == 200
    json_resp = response.json()
    assert "predictions" in json_resp
    assert len(json_resp["predictions"]) > 0


CHAT_CASES = [
    {"id": "chat_sync", "dir": "python3_dummy_chat"},
    {"id": "chat_async", "dir": "python3_dummy_chat_async"},
]


@pytest.fixture(params=CHAT_CASES, ids=lambda c: c["id"])
def chat_client(request, monkeypatch):
    model_dir = os.path.join(MODELS_DIR, request.param["dir"])
    monkeypatch.setenv("CODE_DIR", model_dir)
    monkeypatch.setenv("TARGET_TYPE", "textgeneration")
    with TestClient(app) as c:
        yield c


def test_chat_per_model(chat_client):
    response = chat_client.post(
        "/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"]
