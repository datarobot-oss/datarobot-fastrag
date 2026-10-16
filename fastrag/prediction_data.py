"""Non-blocking reporter for chat prompt/response rows (DataRobot "prediction data").

DRUM reports one row per chat through ``datarobot-mlops`` ``report_predictions_data``; these
rows back the deployment's Data exploration (Data quality, Data export) and join to custom
metrics by association ID. This posts the same payload to the same endpoint directly.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .prediction_stats import RETRY_STATUS_CODES
from .prediction_stats import SUCCESS_STATUS_CODES
from .prediction_stats import StatsConfig
from .prediction_stats import _api_v2_base
from .prediction_stats import dr_timestamp

logger = logging.getLogger("fastrag.prediction_data")

DATA_URL = "{base}/api/v2/deployments/{deployment_id}/predictionInputs/fromJSON/"
DEPLOYMENT_URL = "{base}/api/v2/deployments/{deployment_id}/"
SETTINGS_URL = "{base}/api/v2/deployments/{deployment_id}/settings/"
# DRUM's fallback when the deployment does not name its prompt column.
DEFAULT_PROMPT_COLUMN_NAME = "promptText"


@dataclass(frozen=True, slots=True)
class ChatRow:
    epoch_s: float
    prompt: str
    response: str | None
    association_id: str


def chat_prompt_text(completion_create_params: dict[str, Any]) -> str | None:
    """Return the latest message's text, flattening content parts the way DRUM does."""
    messages = completion_create_params.get("messages") or []
    if not messages:
        return None
    content = messages[-1].get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts = []
    for part in content:
        part_type = part.get("type") if isinstance(part, dict) else None
        if part_type == "text":
            parts.append(str(part.get("text", "")))
        elif part_type == "image_url":
            parts.append(f"Image URL: {part.get('image_url', {}).get('url')}")
        elif part_type == "input_audio":
            parts.append(f"Audio Input, Format: {part.get('input_audio', {}).get('format')}")
        else:
            parts.append(f"Unhandled content type: {part_type}")
    return "\n".join(parts)


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class PredictionDataReporter:
    """Queue chat rows on the request path; drain and POST them from a background task.

    Rows are only queued while the deployment has prediction data collection or feature drift
    tracking enabled, matching DRUM. Those settings and the prompt column name are re-read from
    the deployment every ``settings_refresh_s`` seconds.
    """

    def __init__(
        self,
        config: StatsConfig,
        client: httpx.AsyncClient | None = None,
        settings_refresh_s: float = 60.0,
    ) -> None:
        self._config = config
        self._client = client
        self._owns_client = client is None
        self._settings_refresh_s = settings_refresh_s
        self._queue: asyncio.Queue[ChatRow | None] = asyncio.Queue(maxsize=config.queue_max_size)
        self._drain_task: asyncio.Task[None] | None = None
        self._refresh_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._enabled = False
        self._prompt_column = DEFAULT_PROMPT_COLUMN_NAME
        self._model_id = config.model_id
        self.dropped = 0
        self._posted = 0
        self._closed = False

        base = _api_v2_base(config.endpoint)
        self._data_url = DATA_URL.format(base=base, deployment_id=config.deployment_id)
        self._deployment_url = DEPLOYMENT_URL.format(base=base, deployment_id=config.deployment_id)
        self._settings_url = SETTINGS_URL.format(base=base, deployment_id=config.deployment_id)

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._config.timeout_s,
                headers={
                    "Authorization": f"Bearer {self._config.api_token}",
                    "Content-Type": "application/json",
                },
            )
        await self.refresh_settings()
        self._drain_task = asyncio.create_task(self._run(), name="prediction-data-drain")
        self._refresh_task = asyncio.create_task(
            self._refresh_loop(), name="prediction-data-settings"
        )

    def report(self, *, prompt: str | None, response: str | None, association_id: str) -> None:
        if self._closed or not self._enabled or prompt is None:
            return
        row = ChatRow(
            epoch_s=time.time(), prompt=prompt, response=response, association_id=association_id
        )
        # Sync streaming responses finish in a Starlette worker thread, and asyncio.Queue is not
        # thread-safe, so rows from other threads are handed to the event loop.
        if self._loop is not None and _running_loop() is not self._loop:
            self._loop.call_soon_threadsafe(self._enqueue, row)
        else:
            self._enqueue(row)

    def _enqueue(self, row: ChatRow) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(row)
        except asyncio.QueueFull:
            self.dropped += 1
            if self.dropped % 100 == 1:
                logger.warning(
                    "Prediction data queue is full; discarded %d row(s) so far.", self.dropped
                )

    async def refresh_settings(self) -> None:
        """Re-read whether rows are collected, the prompt column and the model ID."""
        assert self._client is not None
        try:
            settings_response = await self._client.get(self._settings_url)
            settings_response.raise_for_status()
            settings = settings_response.json()
            deployment_response = await self._client.get(self._deployment_url)
            deployment_response.raise_for_status()
            model = deployment_response.json().get("model") or {}
        except (httpx.HTTPError, ValueError) as exc:
            # Keep the last known settings; before the first success that means "off".
            logger.warning("Could not read deployment settings for prediction data: %s", exc)
            return

        collection = (settings.get("predictionsDataCollection") or {}).get("enabled", False)
        drift = (settings.get("featureDrift") or {}).get("enabled", False)
        enabled = bool(collection or drift)
        if enabled != self._enabled:
            logger.info(
                "Prediction data reporting %s (data collection=%s, feature drift=%s).",
                "on" if enabled else "off",
                collection,
                drift,
            )
        self._enabled = enabled
        self._prompt_column = model.get("prompt") or DEFAULT_PROMPT_COLUMN_NAME
        self._model_id = self._config.model_id or model.get("id")

    async def aclose(self, timeout_s: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        if self._refresh_task is not None:
            self._refresh_task.cancel()
        if self._drain_task is not None:
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                try:
                    if self._queue.get_nowait() is not None:
                        self.dropped += 1
                    self._queue.put_nowait(None)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
            try:
                await asyncio.wait_for(self._drain_task, timeout=timeout_s)
            except (TimeoutError, asyncio.CancelledError):
                logger.warning(
                    "Prediction data queue did not finish draining within %ss; "
                    "remaining rows are discarded.",
                    timeout_s,
                )
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self._settings_refresh_s)
            try:
                await self.refresh_settings()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Refreshing prediction data settings failed; continuing.")

    async def _run(self) -> None:
        while True:
            try:
                batch, stopping = await self._collect_batch()
                if batch:
                    await self._post(batch)
                if stopping:
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Prediction data drain failed; continuing.")

    async def _collect_batch(self) -> tuple[list[ChatRow], bool]:
        first = await self._queue.get()
        if first is None:
            return [], True

        batch = [first]
        deadline = time.monotonic() + self._config.flush_interval_s
        while len(batch) < self._config.max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                row = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except TimeoutError:
                break
            if row is None:
                return batch, True
            batch.append(row)
        return batch, False

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

    async def _post(self, batch: list[ChatRow]) -> None:
        assert self._client is not None
        payload = self._payload(batch)

        for attempt in range(self._config.max_attempts):
            last_attempt = attempt + 1 == self._config.max_attempts
            try:
                response = await self._client.post(self._data_url, json=payload)
            except httpx.HTTPError as exc:
                if last_attempt:
                    logger.warning("Discarding %d prediction data row(s): %s", len(batch), exc)
                    return
                await asyncio.sleep(self._config.retry_backoff_s * 2**attempt)
                continue

            if response.status_code in SUCCESS_STATUS_CODES:
                if self._posted == 0:
                    logger.info(
                        "Prediction data accepted by DataRobot (first %d row(s)).", len(batch)
                    )
                self._posted += len(batch)
                return
            if response.status_code in RETRY_STATUS_CODES and not last_attempt:
                await asyncio.sleep(self._config.retry_backoff_s * 2**attempt)
                continue

            logger.warning(
                "Discarding %d prediction data row(s): HTTP %s %s",
                len(batch),
                response.status_code,
                response.text[:300],
            )
            return


async def start_data_reporter() -> PredictionDataReporter | None:
    config = StatsConfig.from_env()
    if config is None:
        return None
    reporter = PredictionDataReporter(config)
    await reporter.start()
    return reporter
