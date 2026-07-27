import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from fastrag.loader import HookName
from fastrag.server import _ensure_hook_available
from fastrag.server import app
from fastrag.validation import ApiError


# Test the helper function directly
def test_ensure_hook_available_raises():
    mock_loader = MagicMock()
    mock_loader.hooks.has.return_value = False

    with pytest.raises(ApiError) as exc:
        _ensure_hook_available(mock_loader, HookName.SCORE)
    assert exc.value.status_code == 501
    assert exc.value.detail == "score hook not implemented."

    with pytest.raises(ApiError) as exc:
        _ensure_hook_available(mock_loader, HookName.CHAT)
    assert exc.value.status_code == 501
    assert exc.value.detail == "chat hook not implemented."


def test_ensure_hook_available_passes():
    mock_loader = MagicMock()
    mock_loader.hooks.has.return_value = True
    # Should not raise
    _ensure_hook_available(mock_loader, HookName.SCORE)


# Integration test with mocked settings
@pytest.fixture
def client_unstructured(monkeypatch):
    # Mock settings to return target_type='unstructured'

    # We need to mock load_settings to return a settings object with target_type='unstructured'
    # and code_dir pointing to dummy_model (which lacks score_unstructured)

    # We can use the existing load_settings but modify the return value
    # OR simpler: use monkeypatch.setenv

    monkeypatch.setenv("CODE_DIR", os.path.abspath("tests/dummy_model"))
    monkeypatch.setenv("TARGET_TYPE", "unstructured")

    with TestClient(app) as c:
        yield c


def test_predict_unstructured_missing_hook(client_unstructured):
    # The dummy model does not have score_unstructured
    # But we force target_type='unstructured' so it passes the first check in predict_unstructured
    # Then it should fail at _ensure_hook_available(HookName.SCORE_UNSTRUCTURED)

    response = client_unstructured.post(
        "/predictUnstructured/", content=b"test", headers={"Content-Type": "text/plain"}
    )
    assert "score_unstructured hook not implemented" in response.json()["detail"]
    assert response.status_code == 501
