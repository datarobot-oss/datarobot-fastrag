import pytest
import yaml

from fastrag.model_metadata import ModelMetadata
from fastrag.model_metadata import TargetType
from fastrag.runtime_parameters_schema import RuntimeParameterTypes


def test_model_metadata_from_yaml(tmp_path):
    yaml_content = {
        "name": "My Custom Model",
        "type": "inference",
        "targetType": "binary",
        "inferenceModel": {
            "targetName": "is_fraud",
            "positiveClassLabel": "Yes",
            "negativeClassLabel": "No",
            "predictionThreshold": 0.5,
        },
        "runtimeParameters": [
            {
                "name": "confidence_level",
                "type": "numeric",
                "defaultValue": 0.8,
                "minValue": 0.0,
                "maxValue": 1.0,
            },
            {"name": "api_key", "type": "credential", "allowEmpty": False},
        ],
    }

    metadata_file = tmp_path / "model-metadata.yaml"
    with open(metadata_file, "w") as f:
        yaml.dump(yaml_content, f)

    metadata = ModelMetadata.from_yaml(str(metadata_file))

    assert metadata.name == "My Custom Model"
    assert metadata.target_type == TargetType.BINARY
    assert metadata.inference_model.target_name == "is_fraud"
    assert metadata.inference_model.positive_class_label == "Yes"
    assert metadata.inference_model.prediction_threshold == 0.5

    assert len(metadata.runtime_parameters) == 2
    param1 = next(p for p in metadata.runtime_parameters if p.name == "confidence_level")
    assert param1.type == RuntimeParameterTypes.NUMERIC
    assert param1.default == 0.8
    assert param1.min_value == 0.0
    assert param1.max_value == 1.0

    param2 = next(p for p in metadata.runtime_parameters if p.name == "api_key")
    assert param2.type == RuntimeParameterTypes.CREDENTIAL
    assert param2.allow_empty is False


def test_model_metadata_minimal(tmp_path):
    yaml_content = {"targetType": "regression"}
    metadata_file = tmp_path / "model-metadata.yaml"
    with open(metadata_file, "w") as f:
        yaml.dump(yaml_content, f)

    metadata = ModelMetadata.from_yaml(str(metadata_file))
    assert metadata.target_type == TargetType.REGRESSION
    assert metadata.name is None
    assert metadata.runtime_parameters == []


def test_model_metadata_alias_definitions(tmp_path):
    yaml_content = {
        "targetType": "regression",
        "runtimeParameterDefinitions": [{"name": "param1", "type": "string"}],
    }
    metadata_file = tmp_path / "model-metadata.yaml"
    with open(metadata_file, "w") as f:
        yaml.dump(yaml_content, f)

    metadata = ModelMetadata.from_yaml(str(metadata_file))
    assert len(metadata.runtime_parameters) == 1
    assert metadata.runtime_parameters[0].name == "param1"


def test_model_metadata_env_overrides(tmp_path, monkeypatch):
    yaml_content = {
        "targetType": "regression",
        "inferenceModel": {
            "positiveClassLabel": "True",
            "negativeClassLabel": "False",
        },
    }
    metadata_file = tmp_path / "model-metadata.yaml"
    with open(metadata_file, "w") as f:
        yaml.dump(yaml_content, f)

    # Set environment variables to override YAML values
    monkeypatch.setenv("TARGET_TYPE", "binary")
    monkeypatch.setenv("POSITIVE_CLASS_LABEL", "Yes")
    monkeypatch.setenv("NEGATIVE_CLASS_LABEL", "No")
    monkeypatch.setenv("CLASS_LABELS", "A, B, C")

    metadata = ModelMetadata.from_yaml(str(metadata_file))
    metadata.merge_env_overrides()

    assert metadata.target_type == TargetType.BINARY
    assert metadata.inference_model.positive_class_label == "Yes"
    assert metadata.inference_model.negative_class_label == "No"
    assert metadata.inference_model.class_labels == ["A", "B", "C"]


def test_target_name_env_override_strips_quotes(monkeypatch):
    """DataRobot exports TARGET_NAME wrapped in double quotes."""
    monkeypatch.setenv("TARGET_NAME", '"relevant"')

    metadata = ModelMetadata(target_type=TargetType.VECTOR_DATABASE)
    metadata.merge_env_overrides()

    assert metadata.inference_model.target_name == "relevant"


def test_target_name_env_override_without_quotes(monkeypatch):
    monkeypatch.setenv("TARGET_NAME", "relevant")

    metadata = ModelMetadata(target_type=TargetType.VECTOR_DATABASE)
    metadata.merge_env_overrides()

    assert metadata.inference_model.target_name == "relevant"


def test_target_name_env_override_strips_quotes_greedily(monkeypatch):
    """`.strip('"')` removes every surrounding quote, not just one balanced pair.

    This deliberately diverges from DRUM, which strips a single pair only when the raw
    value does not match a column. A target name that itself starts or ends with a quote
    is not supported.
    """
    monkeypatch.setenv("TARGET_NAME", '""relevant""')

    metadata = ModelMetadata(target_type=TargetType.VECTOR_DATABASE)
    metadata.merge_env_overrides()

    assert metadata.inference_model.target_name == "relevant"


def _write_metadata(path, content):
    with open(path, "w") as f:
        yaml.dump(content, f)


@pytest.fixture
def no_metadata_env(monkeypatch):
    for name in ("TARGET_TYPE", "TARGET_NAME", "CLASS_LABELS"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("no_metadata_env")
def test_for_server_reads_code_dir_metadata(tmp_path):
    _write_metadata(
        tmp_path / "model-metadata.yaml",
        {"targetType": "textgeneration", "inferenceModel": {"targetName": "answer"}},
    )

    metadata = ModelMetadata.for_server(str(tmp_path), runtime_params_file=None)

    assert metadata.target_type == TargetType.TEXT_GENERATION
    assert metadata.inference_model.target_name == "answer"


@pytest.mark.usefixtures("no_metadata_env")
def test_for_server_code_dir_metadata_wins_over_runtime_params_file(tmp_path):
    _write_metadata(tmp_path / "model-metadata.yaml", {"targetType": "textgeneration"})
    runtime_params_file = tmp_path / "runtime-params.yaml"
    _write_metadata(runtime_params_file, {"targetType": "binary"})

    metadata = ModelMetadata.for_server(str(tmp_path), runtime_params_file=str(runtime_params_file))

    assert metadata.target_type == TargetType.TEXT_GENERATION


@pytest.mark.usefixtures("no_metadata_env")
def test_for_server_env_wins_over_code_dir_metadata(tmp_path, monkeypatch):
    _write_metadata(tmp_path / "model-metadata.yaml", {"targetType": "textgeneration"})
    monkeypatch.setenv("TARGET_TYPE", "Binary")
    monkeypatch.setenv("CLASS_LABELS", "yes, no")

    metadata = ModelMetadata.for_server(str(tmp_path), runtime_params_file=None)

    assert metadata.target_type == TargetType.BINARY
    assert metadata.inference_model.class_labels == ["yes", "no"]


@pytest.mark.usefixtures("no_metadata_env")
def test_for_server_defaults_without_metadata(tmp_path):
    metadata = ModelMetadata.for_server(str(tmp_path), runtime_params_file=None)

    assert metadata.target_type == TargetType.REGRESSION
