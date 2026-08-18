"""Report deployment prediction stats to DataRobot without blocking the request path.

DRUM reported these stats through ``datarobot-mlops`` configured with the API spooler
plus ``set_async_reporting()``, which offloads to a forked ``multiprocessing`` worker
that POSTs one record at a time. This module does the same job for an asyncio server:
an ``asyncio.Queue``, one drain task, and batched POSTs over a pooled HTTP client.

Nothing here imports FastAPI or the rest of ``fastrag``, so it can be lifted into a
shared package when another front server needs the same reporting.

Environment contract is the one DataRobot already injects for DRUM:

``MONITOR``
    Enables reporting when truthy.
``EXTERNAL_WEB_SERVER_URL``, ``API_TOKEN``
    DataRobot endpoint and credentials.
``DEPLOYMENT_ID``, ``MODEL_ID``
    Target of the report. The ``MLOPS_``-prefixed spellings are accepted too.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime

import httpx

logger = logging.getLogger("fastrag.prediction_stats")

STATS_ENDPOINT_TEMPLATE = "{base}/api/v2/deployments/{deployment_id}/predictionRequests/fromJSON/"

RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
SUCCESS_STATUS_CODES = frozenset({200, 202})


def dr_timestamp(epoch_s: float) -> str:
    """Format an epoch the way the MLOps library does.

    Mirrors ``datarobot_mlops.metric.GeneralStats.to_dr_timestamp``: local time with a
    UTC offset, microseconds truncated to milliseconds.

    Called from the drain task rather than the request path: ``strftime`` costs a few
    microseconds, which is most of what reporting a request would otherwise spend.
    """
    micro = datetime.fromtimestamp(epoch_s).astimezone().strftime("%Y-%m-%d %H:%M:%S.%f%z")
    return micro[0:23] + micro[26:]


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class StatsRecord:
    """One prediction request, as the DataRobot API expects it.

    ``epoch_s`` is when the request finished; it is formatted at post time so the cost
    of formatting stays off the request path.
    """

    epoch_s: float
    num_predictions: int
    execution_time_ms: float
    user_error: bool = False
    system_error: bool = False

    def to_payload(self, model_id: str | None) -> dict[str, object]:
        payload: dict[str, object] = {
            "timestamp": dr_timestamp(self.epoch_s),
            "numPredictions": self.num_predictions,
            "executionTime": self.execution_time_ms,
            "userError": self.user_error,
            "systemError": self.system_error,
        }
        if model_id:
            payload["modelId"] = model_id
        return payload


@dataclass(frozen=True, slots=True)
class StatsConfig:
    endpoint: str
    api_token: str
    deployment_id: str
    model_id: str | None = None
    # Defaults match the MLOps agent's own batching (agentMaxAggregatedRecords and
    # agentPendingRecordsTimeoutMs in default.agent.conf.yaml).
    max_batch: int = 500
    flush_interval_s: float = 5.0
    queue_max_size: int = 10_000
    timeout_s: float = 10.0
    max_attempts: int = 3
    retry_backoff_s: float = 0.5

    @property
    def url(self) -> str:
        return STATS_ENDPOINT_TEMPLATE.format(
            base=self.endpoint.rstrip("/"), deployment_id=self.deployment_id
        )

    @classmethod
    def from_env(cls) -> "StatsConfig | None":
        """Build a config from the deployment environment, or None if reporting is off."""
        if not _is_truthy(os.environ.get("MONITOR")):
            logger.info("Prediction stats reporting is off (MONITOR is not set).")
            return None

        endpoint = _first_env("EXTERNAL_WEB_SERVER_URL", "DATAROBOT_ENDPOINT", "MLOPS_SERVICE_URL")
        api_token = _first_env("API_TOKEN", "DATAROBOT_API_TOKEN", "MLOPS_API_TOKEN")
        deployment_id = _first_env("DEPLOYMENT_ID", "MLOPS_DEPLOYMENT_ID")
        model_id = _first_env("MODEL_ID", "MLOPS_MODEL_ID")

        missing = [
            name
            for name, value in (
                ("EXTERNAL_WEB_SERVER_URL", endpoint),
                ("API_TOKEN", api_token),
                ("DEPLOYMENT_ID", deployment_id),
            )
            if not value
        ]
        if missing:
            logger.warning(
                "MONITOR is set but prediction stats reporting is off; missing %s.",
                ", ".join(missing),
            )
            return None

        # mypy: the `missing` check above proves these are set.
        assert endpoint is not None
        assert api_token is not None
        assert deployment_id is not None
        return cls(
            endpoint=endpoint,
            api_token=api_token,
            deployment_id=deployment_id,
            model_id=model_id,
        )


class PredictionStatsReporter:
    """Queues prediction stats and posts them to DataRobot in batches."""

    def __init__(self, config: StatsConfig, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._client = client
        self._owns_client = client is None
        self._queue: asyncio.Queue[StatsRecord | None] = asyncio.Queue(
            maxsize=config.queue_max_size
        )
        self._task: asyncio.Task[None] | None = None
        self._dropped = 0
        self._posted = 0
        self._closed = False

    @property
    def dropped(self) -> int:
        """Records discarded because the queue was full."""
        return self._dropped

    @property
    def posted(self) -> int:
        """Records DataRobot has accepted."""
        return self._posted

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._config.timeout_s,
                headers={
                    "Authorization": f"Bearer {self._config.api_token}",
                    "Content-Type": "application/json",
                },
            )
        self._task = asyncio.create_task(self._run(), name="prediction-stats-drain")
        logger.info("Reporting prediction stats to %s", self._config.url)

    def report(
        self,
        *,
        num_predictions: int,
        execution_time_ms: float,
        user_error: bool = False,
        system_error: bool = False,
    ) -> None:
        """Queue one record. Never awaits, never raises: safe on the request path."""
        if self._closed:
            return

        record = StatsRecord(
            epoch_s=time.time(),
            num_predictions=num_predictions,
            execution_time_ms=execution_time_ms,
            user_error=user_error,
            system_error=system_error,
        )
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self._dropped += 1
            if self._dropped % 100 == 1:
                logger.warning(
                    "Prediction stats queue is full; discarded %d record(s) so far.",
                    self._dropped,
                )

    async def aclose(self, timeout_s: float = 5.0) -> None:
        """Post whatever is queued, then stop. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True

        if self._task is not None:
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=timeout_s)
            except (TimeoutError, asyncio.CancelledError):
                logger.warning(
                    "Prediction stats queue did not finish draining within %ss; "
                    "remaining records are discarded.",
                    timeout_s,
                )
        if self._owns_client and self._client is not None:
            await self._client.aclose()

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
                # Reporting must never take down the drain task or the server with it.
                logger.exception("Prediction stats drain failed; continuing.")

    async def _collect_batch(self) -> tuple[list[StatsRecord], bool]:
        """Collect up to max_batch records, waiting at most flush_interval_s for them."""
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
                record = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except TimeoutError:
                break
            if record is None:
                return batch, True
            batch.append(record)
        return batch, False

    async def _post(self, batch: list[StatsRecord]) -> None:
        assert self._client is not None
        payload = {"data": [record.to_payload(self._config.model_id) for record in batch]}

        for attempt in range(self._config.max_attempts):
            last_attempt = attempt + 1 == self._config.max_attempts
            try:
                response = await self._client.post(self._config.url, json=payload)
            except httpx.HTTPError as exc:
                if last_attempt:
                    logger.warning("Discarding %d prediction stats record(s): %s", len(batch), exc)
                    return
                await asyncio.sleep(self._config.retry_backoff_s * 2**attempt)
                continue

            if response.status_code in SUCCESS_STATUS_CODES:
                if self._posted == 0:
                    # One line at INFO proves the whole path works in a deployment, where
                    # the alternative is a silently empty Total Predictions.
                    logger.info(
                        "Prediction stats accepted by DataRobot (first %d record(s)).",
                        len(batch),
                    )
                self._posted += len(batch)
                logger.debug("Reported %d prediction stats record(s).", len(batch))
                return
            if response.status_code in RETRY_STATUS_CODES and not last_attempt:
                await asyncio.sleep(self._config.retry_backoff_s * 2**attempt)
                continue

            logger.warning(
                "Discarding %d prediction stats record(s): HTTP %s %s",
                len(batch),
                response.status_code,
                response.text[:300],
            )
            return


async def start_reporter() -> PredictionStatsReporter | None:
    """Start a reporter from the environment, or return None when reporting is off."""
    config = StatsConfig.from_env()
    if config is None:
        return None
    reporter = PredictionStatsReporter(config)
    await reporter.start()
    return reporter
