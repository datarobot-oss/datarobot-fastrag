import json
import os
from typing import Any
from typing import Dict

import yaml
from pydantic import ValidationError

from .model_metadata import ModelMetadata
from .runtime_parameters_schema import RuntimeParameterBooleanPayload
from .runtime_parameters_schema import RuntimeParameterCredentialPayload
from .runtime_parameters_schema import RuntimeParameterDefinition
from .runtime_parameters_schema import RuntimeParameterDeploymentPayload
from .runtime_parameters_schema import RuntimeParameterNumericPayload
from .runtime_parameters_schema import RuntimeParameterPayload
from .runtime_parameters_schema import RuntimeParameterStringPayload
from .runtime_parameters_schema import RuntimeParameterTypes

MODEL_CONFIG_FILENAME = "model-metadata.yaml"


class RuntimeParameters:
    """
    A class that is used to read runtime-parameters that are delivered to the executed
    custom model.
    """

    PARAM_PREFIX = "MLOPS_RUNTIME_PARAM"

    @classmethod
    def get(cls, key: str) -> Any:
        runtime_param_key = cls.namespaced_param_name(key)
        if runtime_param_key not in os.environ:
            raise ValueError(f"Runtime parameter '{key}' does not exist!")

        try:
            env_value = json.loads(os.environ[runtime_param_key])
            payload = RuntimeParameterPayload.model_validate(env_value)
            return payload.payload
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Failed to parse runtime parameter '{key}': {e}")

    @classmethod
    def namespaced_param_name(cls, param_name: str) -> str:
        return f"{cls.PARAM_PREFIX}_{param_name}"

    @classmethod
    def has(cls, param_name: str) -> bool:
        runtime_param_key = cls.namespaced_param_name(param_name)
        return runtime_param_key in os.environ


class RuntimeParametersLoader:
    """
    This class is used by DRUM to load runtime parameter values from a provided YAML file.
    """

    def __init__(self, values_filepath: str, code_dir: str):
        self.values_filepath = values_filepath
        self.code_dir = code_dir
        self.parameter_definitions: Dict[str, RuntimeParameterDefinition] = {}
        self.yaml_content: Dict[str, Any] = {}

        self._load_parameter_definitions()
        self._load_values()

    def _load_parameter_definitions(self) -> None:
        config_path = os.path.join(self.code_dir, MODEL_CONFIG_FILENAME)
        if not os.path.exists(config_path):
            return

        metadata = ModelMetadata.from_yaml(config_path)
        for defn in metadata.runtime_parameters:
            self.parameter_definitions[defn.name] = defn

    def _load_values(self) -> None:
        if not os.path.exists(self.values_filepath):
            raise FileNotFoundError(
                f"Runtime parameter values file not found: {self.values_filepath}"
            )

        with open(self.values_filepath, "r") as f:
            loaded = yaml.safe_load(f)

        if loaded is None:
            self.yaml_content = {}
            return
        if not isinstance(loaded, dict):
            raise ValueError(
                "Runtime parameter values file must be a mapping of parameter names to values"
            )
        self.yaml_content = loaded

    def setup_environment_variables(self) -> None:
        for name, defn in self.parameter_definitions.items():
            value = self.yaml_content.get(name, defn.default)
            payload: RuntimeParameterPayload

            if defn.type == RuntimeParameterTypes.STRING:
                payload = RuntimeParameterStringPayload(payload=value)
            elif defn.type == RuntimeParameterTypes.BOOLEAN:
                payload = RuntimeParameterBooleanPayload(payload=value)
            elif defn.type == RuntimeParameterTypes.NUMERIC:
                payload = RuntimeParameterNumericPayload(payload=value)
            elif defn.type == RuntimeParameterTypes.CREDENTIAL:
                payload = RuntimeParameterCredentialPayload(payload=value)
            elif defn.type == RuntimeParameterTypes.DEPLOYMENT:
                payload = RuntimeParameterDeploymentPayload(payload=value)
            else:
                continue

            env_key = RuntimeParameters.namespaced_param_name(name)
            os.environ[env_key] = payload.model_dump_json()
