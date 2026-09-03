import os
from enum import Enum
from typing import List
from typing import Literal
from typing import Optional

import yaml
from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .runtime_parameters_schema import RuntimeParameterDefinition


class TargetType(str, Enum):
    BINARY = "binary"
    REGRESSION = "regression"
    ANOMALY = "anomaly"
    UNSTRUCTURED = "unstructured"
    MULTICLASS = "multiclass"
    TEXT_GENERATION = "textgeneration"
    GEO_POINT = "geopoint"
    VECTOR_DATABASE = "vectordatabase"
    AGENTIC_WORKFLOW = "agenticworkflow"

    def is_classification(self) -> bool:
        return self in [self.BINARY, self.MULTICLASS]

    def is_single_column(self) -> bool:
        return self in [
            self.REGRESSION,
            self.ANOMALY,
            self.TEXT_GENERATION,
            self.GEO_POINT,
            self.VECTOR_DATABASE,
            self.AGENTIC_WORKFLOW,
        ]


class InferenceModelConfig(BaseModel):
    target_name: str = Field(default="target", alias="targetName")
    positive_class_label: Optional[str] = Field(default=None, alias="positiveClassLabel")
    negative_class_label: Optional[str] = Field(default=None, alias="negativeClassLabel")
    class_labels: Optional[List[str]] = Field(default=None, alias="classLabels")
    prediction_threshold: Optional[float] = Field(default=0.5, alias="predictionThreshold")
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class ModelMetadata(BaseModel):
    name: Optional[str] = None
    type: Literal["inference"] = "inference"
    target_type: TargetType = Field(default=TargetType.REGRESSION, alias="targetType")
    inference_model: InferenceModelConfig = Field(
        default_factory=InferenceModelConfig, alias="inferenceModel"
    )
    runtime_parameters: List[RuntimeParameterDefinition] = Field(
        default_factory=list,
        validation_alias=AliasChoices("runtimeParameters", "runtimeParameterDefinitions"),
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @classmethod
    def from_yaml(cls, path: str) -> "ModelMetadata":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        metadata = cls.model_validate(data)
        return metadata

    def merge_env_overrides(self) -> None:
        if target_type := os.environ.get("TARGET_TYPE"):
            self.target_type = TargetType(target_type.lower())
        if target_name := os.environ.get("TARGET_NAME"):
            # DataRobot exports TARGET_NAME wrapped in double quotes; DRUM strips
            # them before matching the column, so we do the same.
            if len(target_name) >= 2 and target_name[0] == '"' and target_name[-1] == '"':
                target_name = target_name[1:-1]
            self.inference_model.target_name = target_name
        if pos_label := os.environ.get("POSITIVE_CLASS_LABEL"):
            self.inference_model.positive_class_label = pos_label
        if neg_label := os.environ.get("NEGATIVE_CLASS_LABEL"):
            self.inference_model.negative_class_label = neg_label
        if class_labels := os.environ.get("CLASS_LABELS"):
            self.inference_model.class_labels = [s.strip() for s in class_labels.split(",")]
