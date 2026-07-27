import json

import pytest
import yaml

from fastrag.runtime_parameters import RuntimeParameters
from fastrag.runtime_parameters import RuntimeParametersLoader
from fastrag.runtime_parameters_schema import RuntimeParameterBooleanPayload
from fastrag.runtime_parameters_schema import RuntimeParameterCredentialPayload
from fastrag.runtime_parameters_schema import RuntimeParameterDeploymentPayload
from fastrag.runtime_parameters_schema import RuntimeParameterNumericPayload
from fastrag.runtime_parameters_schema import RuntimeParameterStringPayload


class TestRuntimeParameters:
    def test_get_string_param(self, monkeypatch):
        payload = RuntimeParameterStringPayload(payload="hello")
        monkeypatch.setenv("MLOPS_RUNTIME_PARAM_my_key", payload.model_dump_json())
        assert RuntimeParameters.get("my_key") == "hello"

    def test_get_boolean_param(self, monkeypatch):
        payload = RuntimeParameterBooleanPayload(payload=True)
        monkeypatch.setenv("MLOPS_RUNTIME_PARAM_flag", payload.model_dump_json())
        assert RuntimeParameters.get("flag") is True

    def test_get_numeric_param(self, monkeypatch):
        payload = RuntimeParameterNumericPayload(payload=3.14)
        monkeypatch.setenv("MLOPS_RUNTIME_PARAM_threshold", payload.model_dump_json())
        assert RuntimeParameters.get("threshold") == 3.14

    def test_get_credential_param(self, monkeypatch):
        cred = {"apiToken": "secret123", "apiTokenType": "bearer"}
        payload = RuntimeParameterCredentialPayload(payload=cred)
        monkeypatch.setenv("MLOPS_RUNTIME_PARAM_cred", payload.model_dump_json())
        assert RuntimeParameters.get("cred") == cred

    def test_get_deployment_param(self, monkeypatch):
        payload = RuntimeParameterDeploymentPayload(payload="deploy-id-123")
        monkeypatch.setenv("MLOPS_RUNTIME_PARAM_dep", payload.model_dump_json())
        assert RuntimeParameters.get("dep") == "deploy-id-123"

    def test_get_missing_param_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            RuntimeParameters.get("nonexistent_key")

    def test_get_invalid_json_raises(self, monkeypatch):
        monkeypatch.setenv("MLOPS_RUNTIME_PARAM_bad", "not-json")
        with pytest.raises(ValueError, match="Failed to parse"):
            RuntimeParameters.get("bad")

    def test_get_invalid_payload_raises(self, monkeypatch):
        monkeypatch.setenv("MLOPS_RUNTIME_PARAM_bad", json.dumps({"wrong": "schema"}))
        with pytest.raises(ValueError, match="Failed to parse"):
            RuntimeParameters.get("bad")

    def test_namespaced_param_name(self):
        assert RuntimeParameters.namespaced_param_name("foo") == "MLOPS_RUNTIME_PARAM_foo"

    def test_has_returns_true(self, monkeypatch):
        monkeypatch.setenv("MLOPS_RUNTIME_PARAM_exists", "anything")
        assert RuntimeParameters.has("exists") is True

    def test_has_returns_false(self):
        assert RuntimeParameters.has("definitely_not_set_xyz") is False


def _make_metadata_yaml(tmp_path, params):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    metadata = {
        "targetType": "regression",
        "runtimeParameters": params,
    }
    with open(code_dir / "model-metadata.yaml", "w") as f:
        yaml.dump(metadata, f)
    return str(code_dir)


class TestRuntimeParametersLoader:
    def test_loads_string_param(self, tmp_path, monkeypatch):
        code_dir = _make_metadata_yaml(
            tmp_path,
            [
                {"name": "greeting", "type": "string", "defaultValue": "hi"},
            ],
        )
        values_file = tmp_path / "values.yaml"
        values_file.write_text(yaml.dump({"greeting": "hello"}))

        loader = RuntimeParametersLoader(str(values_file), code_dir)
        loader.setup_environment_variables()

        assert RuntimeParameters.get("greeting") == "hello"
        monkeypatch.delenv("MLOPS_RUNTIME_PARAM_greeting", raising=False)

    def test_uses_default_when_value_missing(self, tmp_path, monkeypatch):
        code_dir = _make_metadata_yaml(
            tmp_path,
            [
                {"name": "level", "type": "numeric", "defaultValue": 42},
            ],
        )
        values_file = tmp_path / "values.yaml"
        values_file.write_text(yaml.dump({}))

        loader = RuntimeParametersLoader(str(values_file), code_dir)
        loader.setup_environment_variables()

        assert RuntimeParameters.get("level") == 42
        monkeypatch.delenv("MLOPS_RUNTIME_PARAM_level", raising=False)

    def test_boolean_param(self, tmp_path, monkeypatch):
        code_dir = _make_metadata_yaml(
            tmp_path,
            [
                {"name": "enabled", "type": "boolean", "defaultValue": False},
            ],
        )
        values_file = tmp_path / "values.yaml"
        values_file.write_text(yaml.dump({"enabled": True}))

        loader = RuntimeParametersLoader(str(values_file), code_dir)
        loader.setup_environment_variables()

        assert RuntimeParameters.get("enabled") is True
        monkeypatch.delenv("MLOPS_RUNTIME_PARAM_enabled", raising=False)

    def test_credential_param(self, tmp_path, monkeypatch):
        code_dir = _make_metadata_yaml(
            tmp_path,
            [
                {"name": "api_cred", "type": "credential"},
            ],
        )
        cred_value = {"apiToken": "tok123"}
        values_file = tmp_path / "values.yaml"
        values_file.write_text(yaml.dump({"api_cred": cred_value}))

        loader = RuntimeParametersLoader(str(values_file), code_dir)
        loader.setup_environment_variables()

        assert RuntimeParameters.get("api_cred") == cred_value
        monkeypatch.delenv("MLOPS_RUNTIME_PARAM_api_cred", raising=False)

    def test_deployment_param(self, tmp_path, monkeypatch):
        code_dir = _make_metadata_yaml(
            tmp_path,
            [
                {"name": "dep_id", "type": "deployment"},
            ],
        )
        values_file = tmp_path / "values.yaml"
        values_file.write_text(yaml.dump({"dep_id": "d-abc"}))

        loader = RuntimeParametersLoader(str(values_file), code_dir)
        loader.setup_environment_variables()

        assert RuntimeParameters.get("dep_id") == "d-abc"
        monkeypatch.delenv("MLOPS_RUNTIME_PARAM_dep_id", raising=False)

    def test_values_file_not_found(self, tmp_path):
        code_dir = _make_metadata_yaml(tmp_path, [])
        with pytest.raises(FileNotFoundError):
            RuntimeParametersLoader(str(tmp_path / "missing.yaml"), code_dir)

    def test_values_file_not_a_dict(self, tmp_path):
        code_dir = _make_metadata_yaml(tmp_path, [])
        values_file = tmp_path / "values.yaml"
        values_file.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            RuntimeParametersLoader(str(values_file), code_dir)

    def test_empty_values_file(self, tmp_path):
        code_dir = _make_metadata_yaml(
            tmp_path,
            [
                {"name": "x", "type": "string", "defaultValue": "default_val"},
            ],
        )
        values_file = tmp_path / "values.yaml"
        values_file.write_text("")

        loader = RuntimeParametersLoader(str(values_file), code_dir)
        loader.setup_environment_variables()

    def test_no_model_metadata(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        values_file = tmp_path / "values.yaml"
        values_file.write_text(yaml.dump({"some": "value"}))

        loader = RuntimeParametersLoader(str(values_file), str(code_dir))
        assert loader.parameter_definitions == {}

    def test_multiple_params(self, tmp_path, monkeypatch):
        code_dir = _make_metadata_yaml(
            tmp_path,
            [
                {"name": "s", "type": "string", "defaultValue": "a"},
                {"name": "n", "type": "numeric", "defaultValue": 1},
                {"name": "b", "type": "boolean", "defaultValue": True},
            ],
        )
        values_file = tmp_path / "values.yaml"
        values_file.write_text(yaml.dump({"s": "z", "n": 99}))

        loader = RuntimeParametersLoader(str(values_file), code_dir)
        loader.setup_environment_variables()

        assert RuntimeParameters.get("s") == "z"
        assert RuntimeParameters.get("n") == 99
        assert RuntimeParameters.get("b") is True

        for k in ["s", "n", "b"]:
            monkeypatch.delenv(f"MLOPS_RUNTIME_PARAM_{k}", raising=False)
