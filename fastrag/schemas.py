from typing import Any
from typing import Literal
from typing import Optional
from typing import TypeAlias
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict


class ChatCompletionMessage(BaseModel):
    role: str
    content: Any = None
    refusal: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class HealthResponse(BaseModel):
    message: str
    model_config = ConfigDict(extra="allow")


class InfoResponse(BaseModel):
    server: str
    status: str
    model_loaded: bool
    code_dir: str
    target_type: Optional[str] = None
    positive_class_label: Optional[str] = None
    negative_class_label: Optional[str] = None
    class_labels: Optional[list[str]] = None
    model_config = ConfigDict(extra="allow")


class CapabilitiesResponse(BaseModel):
    supported_payload_formats: dict[str, str]
    supported_methods: dict[str, bool]
    model_config = ConfigDict(extra="allow")


class ExtraModelOutput(BaseModel):
    columns: list[str]
    index: list[Any]
    data: list[list[Any]]
    model_config = ConfigDict(extra="allow")


Primitive = Union[str, int, float, bool, None]
LegacyPredictions = Union[
    list[float], list[dict[str, float]], list[str], list[dict[str, Any]], dict[str, Any]
]


class BasePredictionResponse(BaseModel):
    extraModelOutput: Optional[ExtraModelOutput] = None
    model_config = ConfigDict(extra="allow")


class RegressionPredictionValue(BaseModel):
    label: str
    value: float
    model_config = ConfigDict(extra="allow")


class RegressionPredictionItem(BaseModel):
    prediction: float
    predictionValues: Optional[list[RegressionPredictionValue]] = None
    model_config = ConfigDict(extra="allow")


class BinaryPredictionValue(BaseModel):
    label: str
    value: float
    model_config = ConfigDict(extra="allow")


class BinaryPredictionItem(BaseModel):
    prediction: str
    predictionValues: list[BinaryPredictionValue]
    predictionThreshold: Optional[float] = None
    model_config = ConfigDict(extra="allow")


class MulticlassPredictionValue(BaseModel):
    label: str
    value: float
    model_config = ConfigDict(extra="allow")


class MulticlassPredictionItem(BaseModel):
    prediction: str
    predictionValues: list[MulticlassPredictionValue]
    model_config = ConfigDict(extra="allow")


class RegressionPredictionResponse(BasePredictionResponse):
    predictions: list[RegressionPredictionItem]


class BinaryPredictionResponse(BasePredictionResponse):
    predictions: list[BinaryPredictionItem]


class MulticlassPredictionResponse(BasePredictionResponse):
    predictions: list[MulticlassPredictionItem]


class TextGenerationPredictionResponse(BasePredictionResponse):
    predictions: list[str]


class VectorDatabasePredictionResponse(BasePredictionResponse):
    predictions: list[list[Any]]


class GenericPredictionResponse(BasePredictionResponse):
    predictions: LegacyPredictions


PredictionResponse: TypeAlias = (
    BinaryPredictionResponse
    | RegressionPredictionResponse
    | MulticlassPredictionResponse
    | TextGenerationPredictionResponse
    | VectorDatabasePredictionResponse
    | GenericPredictionResponse
)


class ChatCompletionRequestMessage(BaseModel):
    role: str
    content: Any = None
    model_config = ConfigDict(extra="allow")


class OpenAIChatCompletionRequest(BaseModel):
    messages: list[ChatCompletionRequestMessage]
    model: str
    stream: Optional[bool] = None
    model_config = ConfigDict(extra="allow")


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None
    model_config = ConfigDict(extra="allow")


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_config = ConfigDict(extra="allow")


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    choices: list[ChatCompletionChoice]
    model: Optional[str] = None
    usage: Optional[ChatCompletionUsage] = None
    system_fingerprint: Optional[str] = None
    model_config = ConfigDict(extra="allow")
