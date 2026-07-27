import os

import pytest

from fastrag.settings import Settings
from fastrag.settings import load_settings


def test_settings_defaults():
    s = Settings()
    assert s.code_dir == "."
    assert s.address == "0.0.0.0:8080"
    assert s.workers == 1
    assert s.thread_pool_workers == 2
    assert s.runtime_params_file is None
    assert s.verbose is False
    assert s.allow_dr_api_access is False


def test_setup_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE_DIR", str(tmp_path))
    monkeypatch.setenv("ADDRESS", "127.0.0.1:9000")
    monkeypatch.setenv("MAX_WORKERS", "4")
    monkeypatch.setenv("THREAD_POOL_WORKERS", "8")
    monkeypatch.setenv("VERBOSE", "true")
    monkeypatch.setenv("ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS", "true")
    monkeypatch.delenv("RUNTIME_PARAMS_FILE", raising=False)

    s = Settings()
    s.setup_env()

    assert os.environ["CODE_DIR"] == os.path.abspath(str(tmp_path))
    assert os.environ["ADDRESS"] == "127.0.0.1:9000"
    assert os.environ["MAX_WORKERS"] == "4"
    assert os.environ["THREAD_POOL_WORKERS"] == "8"
    assert os.environ["VERBOSE"] == "true"
    assert os.environ["ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS"] == "true"


def test_setup_env_verbose_false(monkeypatch):
    monkeypatch.delenv("VERBOSE", raising=False)
    s = Settings(verbose=False)
    s.setup_env()
    assert os.environ["VERBOSE"] == "false"


def test_setup_env_with_runtime_params_file(tmp_path, monkeypatch):
    params_file = tmp_path / "params.yaml"
    params_file.touch()

    monkeypatch.setenv("CODE_DIR", str(tmp_path))
    monkeypatch.setenv("RUNTIME_PARAMS_FILE", str(params_file))

    s = Settings()
    s.setup_env()

    assert os.environ["RUNTIME_PARAMS_FILE"] == os.path.abspath(str(params_file))


def test_setup_env_without_runtime_params_file(monkeypatch):
    monkeypatch.delenv("RUNTIME_PARAMS_FILE", raising=False)
    s = Settings()
    s.setup_env()
    assert "RUNTIME_PARAMS_FILE" not in os.environ


def test_merge_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("CODE_DIR", str(tmp_path))
    monkeypatch.setenv("ADDRESS", "192.168.1.1:7000")
    monkeypatch.setenv("MAX_WORKERS", "3")
    monkeypatch.setenv("THREAD_POOL_WORKERS", "6")
    monkeypatch.setenv("VERBOSE", "true")
    monkeypatch.setenv("ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS", "yes")

    s = Settings()
    s._merge_env_overrides()

    assert s.code_dir == str(tmp_path)
    assert s.address == "192.168.1.1:7000"
    assert s.workers == 3
    assert s.thread_pool_workers == 6
    assert s.verbose is True
    assert s.allow_dr_api_access is True


@pytest.mark.parametrize("val", ["false", "0", "no", "off"])
def test_merge_env_overrides_verbose_falsy(monkeypatch, val):
    monkeypatch.setenv("VERBOSE", val)
    s = Settings()
    s._merge_env_overrides()
    assert s.verbose is False


@pytest.mark.parametrize("val", ["true", "1", "yes", "on"])
def test_merge_env_overrides_allow_dr_api_truthy(monkeypatch, val):
    monkeypatch.setenv("ALLOW_DR_API_ACCESS_FOR_ALL_CUSTOM_MODELS", val)
    s = Settings()
    s._merge_env_overrides()
    assert s.allow_dr_api_access is True


def test_merge_env_overrides_with_runtime_params_file(monkeypatch, tmp_path):
    params_file = tmp_path / "params.yaml"
    params_file.touch()
    monkeypatch.setenv("RUNTIME_PARAMS_FILE", str(params_file))

    s = Settings()
    s._merge_env_overrides()

    assert s.runtime_params_file == str(params_file)


def test_load_settings_uses_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CODE_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_WORKERS", "5")

    settings = load_settings()
    assert settings.code_dir == str(tmp_path)
    assert settings.workers == 5


def test_load_settings_defaults(monkeypatch):
    for key in ["CODE_DIR", "ADDRESS", "MAX_WORKERS", "THREAD_POOL_WORKERS"]:
        monkeypatch.delenv(key, raising=False)

    settings = load_settings()
    assert settings.address == "0.0.0.0:8080"
    assert settings.thread_pool_workers == 2
