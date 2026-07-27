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
    with TestClient(app) as c:
        yield c


def test_health_check_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "ok"}


def test_health_check_ping(client):
    response = client.get("/ping/")
    assert response.status_code == 200
    assert response.json() == {"message": "ok"}


def test_health_check_health(client):
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {"message": "ok"}
