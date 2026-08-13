"""DataRobot MLOps deployment-stats reporting.

Without this, a deployment served by fastrag never advances its "Total
Predictions" counter. That counter is driven by exactly one thing: a call to
the ``datarobot-mlops`` library's ``MLOps.report_deployment_stats(...)``. DRUM
makes that call on every request (see ``base_language_predictor.py`` in
datarobot-user-models: ``num_predictions=1`` per chat completion,
``len(predictions)`` for structured scoring, ``0`` on error). fastrag exports
OpenTelemetry traces on a separate rail, so traces work while the prediction
counter stays at zero -- this module closes that gap.

All reporting is best-effort: any failure to initialise or report is logged and
swallowed so monitoring never breaks model serving.
"""

import logging
import os
import time
from typing import Any
from typing import Optional

logger = logging.getLogger("fastrag.monitoring")

try:  # opentelemetry-instrumentation is already a fastrag dependency
    from opentelemetry.instrumentation.utils import suppress_instrumentation
except Exception:  # pragma: no cover - fallback if the util moves/renames
    # nullcontext is a class, not a function, so strict mypy flags the rebind.
    from contextlib import nullcontext as suppress_instrumentation  # type: ignore[assignment]


def elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since a ``time.monotonic()`` timestamp."""
    return (time.monotonic() - start) * 1000.0


def _env_flag(name: str) -> Optional[bool]:
    val = os.environ.get(name)
    if val is None:
        return None
    return val.strip().lower() in {"1", "true", "yes", "on"}


class MLOpsReporter:
    """Reports prediction counts + latency to DataRobot MLOps.

    Mirrors DRUM's ``BaseLanguagePredictor`` monitoring semantics: a successful
    chat completion counts as one prediction, structured scoring counts one per
    output row, and failures report zero predictions.

    Monitoring is enabled when ``MLOPS_DEPLOYMENT_ID`` is present in the
    environment (DataRobot injects it, along with the spooler configuration, at
    deploy time) and is not explicitly disabled via ``MLOPS_MONITORING_DISABLED``
    or ``MONITOR=false``.
    """

    def __init__(self) -> None:
        self._mlops: Any = None

    @property
    def enabled(self) -> bool:
        return self._mlops is not None

    def initialize(self) -> None:
        """Build and initialise the MLOps client. Never raises."""
        if _env_flag("MLOPS_MONITORING_DISABLED"):
            logger.info("MLOps monitoring explicitly disabled (MLOPS_MONITORING_DISABLED)")
            return
        if _env_flag("MONITOR") is False:
            logger.info("MLOps monitoring explicitly disabled (MONITOR=false)")
            return

        deployment_id = os.environ.get("MLOPS_DEPLOYMENT_ID") or os.environ.get("DEPLOYMENT_ID")
        if not deployment_id:
            logger.info("MLOps monitoring disabled: MLOPS_DEPLOYMENT_ID not set")
            return

        try:
            # Optional dependency: present in the deployment image, not in dev.
            from datarobot_mlops.mlops import MLOps  # noqa: PLC0415
        except Exception as exc:  # library not installed in this image
            logger.warning("datarobot-mlops not importable; monitoring disabled: %s", exc)
            return

        model_id = os.environ.get("MLOPS_MODEL_ID") or os.environ.get("MODEL_ID")
        try:
            with suppress_instrumentation():
                mlops = MLOps()
                mlops.set_deployment_id(deployment_id)
                if model_id:
                    mlops.set_model_id(model_id)
                self._configure_spooler(mlops)
                mlops.init()
        except Exception as exc:
            logger.warning(
                "Failed to initialise MLOps monitoring; disabling: %s", exc, exc_info=True
            )
            return

        self._mlops = mlops
        logger.info(
            "MLOps monitoring enabled (deployment_id=%s, model_id=%s)", deployment_id, model_id
        )

    @staticmethod
    def _configure_spooler(mlops: Any) -> None:
        """Configure how the client ships records to DataRobot.

        Preferred path mirrors DRUM: the API spooler, which reports directly to
        the DataRobot MLOps service with no agent needed. DataRobot injects
        ``EXTERNAL_WEB_SERVER_URL`` + ``API_TOKEN`` (also surfaced as
        ``DATAROBOT_ENDPOINT`` / ``DATAROBOT_API_TOKEN``). If instead the spooler
        is already described by ``MLOPS_SPOOLER_TYPE`` (+ its own env vars), leave
        that in place. Raise otherwise so ``initialize`` disables monitoring.
        """
        webserver = (
            os.environ.get("EXTERNAL_WEB_SERVER_URL")
            or os.environ.get("MLOPS_SERVICE_URL")
            or os.environ.get("DATAROBOT_ENDPOINT")
        )
        api_token = (
            os.environ.get("API_TOKEN")
            or os.environ.get("MLOPS_API_TOKEN")
            or os.environ.get("DATAROBOT_API_TOKEN")
        )
        if webserver and api_token:
            mlops.set_api_spooler(mlops_service_url=webserver, mlops_api_token=api_token)
            mlops.set_async_reporting()
            return
        if os.environ.get("MLOPS_SPOOLER_TYPE"):
            # Spooler fully configured via MLOPS_* env vars; just report async.
            mlops.set_async_reporting()
            return
        raise RuntimeError(
            "No MLOps spooler configuration found "
            "(need EXTERNAL_WEB_SERVER_URL + API_TOKEN, or MLOPS_SPOOLER_TYPE)"
        )

    def report_chat(self, execution_time_ms: float) -> None:
        """Report one prediction for a successful chat completion."""
        self._report(num_predictions=1, execution_time_ms=execution_time_ms)

    def report_predictions(self, num_predictions: int, execution_time_ms: float) -> None:
        """Report ``num_predictions`` for a successful structured scoring request."""
        self._report(num_predictions=num_predictions, execution_time_ms=execution_time_ms)

    def report_error(self, execution_time_ms: float) -> None:
        """Report a failed request (zero predictions), matching DRUM."""
        self._report(num_predictions=0, execution_time_ms=execution_time_ms)

    def _report(self, num_predictions: int, execution_time_ms: float) -> None:
        if self._mlops is None:
            return
        try:
            with suppress_instrumentation():
                self._mlops.report_deployment_stats(
                    num_predictions=int(num_predictions),
                    execution_time_ms=float(execution_time_ms),
                )
        except Exception as exc:
            logger.warning("Failed to report deployment stats: %s", exc)

    def shutdown(self) -> None:
        """Flush and tear down the MLOps client. Never raises."""
        if self._mlops is None:
            return
        try:
            self._mlops.shutdown()
        except Exception as exc:
            logger.warning("Error shutting down MLOps: %s", exc)
        finally:
            self._mlops = None
