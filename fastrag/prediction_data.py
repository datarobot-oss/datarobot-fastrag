"""Non-blocking reporter for chat prompt/response rows (DataRobot "prediction data")."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .prediction_stats import BatchReporter
from .prediction_stats import StatsConfig
from .prediction_stats import _api_v2_base
from .prediction_stats import dr_timestamp

logger = logging.getLogger("fastrag.prediction_data")

DEFAULT_PROMPT_COLUMN_NAME = "promptText"


@dataclass(frozen=True, slots=True)
class ChatRow:
    epoch_s: float
    prompt: str
    response: str | None
    association_id: str


def chat_prompt_text(completion_create_params: dict[str, Any]) -> str | None:
    messages = completion_create_params.get("messages") or []
    content = messages[-1].get("content") if messages else None
    if not isinstance(content, list):
        return content if isinstance(content, str) else None
    parts = []
    for part in content:
        if part.get("type") == "text":
            parts.append(str(part.get("text", "")))
        elif part.get("type") == "image_url":
            parts.append(f"Image URL: {part.get('image_url', {}).get('url')}")
        else:
            parts.append(f"Unhandled content type: {part.get('type')}")
    return "\n".join(parts)


class PredictionDataReporter(BatchReporter[ChatRow]):
    name = "prediction data"

    def __init__(self, config: StatsConfig, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(config, client)
        self._base = f"{_api_v2_base(config.endpoint)}/api/v2/deployments/{config.deployment_id}"
        self._prompt_column = DEFAULT_PROMPT_COLUMN_NAME
        self._model_id = config.model_id

    @property
    def url(self) -> str:
        return f"{self._base}/predictionInputs/fromJSON/"

    async def start(self) -> None:
        await super().start()
        assert self._client is not None
        try:
            response = await self._client.get(f"{self._base}/")
            response.raise_for_status()
            model = response.json().get("model") or {}
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "Could not read the deployment; using prompt column %r: %s",
                self._prompt_column,
                exc,
            )
            return
        self._prompt_column = model.get("prompt") or DEFAULT_PROMPT_COLUMN_NAME
        self._model_id = self._model_id or model.get("id")

    def report(self, *, prompt: str | None, response: str | None, association_id: str) -> None:
        if prompt is not None:
            self._put(ChatRow(time.time(), prompt, response, association_id))

    def _payload(self, batch: list[ChatRow]) -> dict[str, Any]:
        items = []
        for row in batch:
            item: dict[str, Any] = {
                "timestamp": dr_timestamp(row.epoch_s),
                "features": [{"name": self._prompt_column, "values": [row.prompt]}],
                "predictions": [row.response],
                "associationIds": [row.association_id],
            }
            if self._model_id:
                item["modelId"] = self._model_id
            items.append(item)
        return {"data": items, "skipDriftTracking": False, "skipAccuracyTracking": False}


async def start_data_reporter() -> PredictionDataReporter | None:
    config = StatsConfig.from_env()
    if config is None:
        return None
    reporter = PredictionDataReporter(config)
    await reporter.start()
    return reporter
