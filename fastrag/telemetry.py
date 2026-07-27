import logging
import os
from typing import Any
from typing import NamedTuple
from typing import Optional

from opentelemetry import metrics
from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs import LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import Counter
from opentelemetry.sdk.metrics import Histogram
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics import ObservableCounter
from opentelemetry.sdk.metrics.export import AggregationTemporality
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from .runtime_parameters import RuntimeParameters

OpenAIInstrumentor().instrument()
logger = logging.getLogger("fastrag.telemetry")


class OTELProviders(NamedTuple):
    trace_provider: TracerProvider
    metric_provider: MeterProvider
    logger_provider: LoggerProvider


class _ExcludeOtelLogsFilter(logging.Filter):
    """A logging filter to exclude logs from the opentelemetry library."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith("opentelemetry")


def _setup_otel_logging(resource: Any, multiprocessing: bool = False) -> LoggerProvider:
    logger_provider = LoggerProvider(resource=resource)
    set_logger_provider(logger_provider)
    exporter = OTLPLogExporter()
    if multiprocessing:
        logger_provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    else:
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    handler = LoggingHandler(level=logging.DEBUG, logger_provider=logger_provider)
    # Remove own logs to avoid infinite recursion if endpoint is not available
    handler.addFilter(_ExcludeOtelLogsFilter())
    logging.getLogger().addHandler(handler)
    return logger_provider


def _setup_otel_metrics(resource: Any) -> MeterProvider:
    # OTEL SDK default temporality is CUMULATIVE, but this is rarely what users
    # actually want to work with, so here we switch default. Also in case of delta
    # PeriodicExportingMetricReader does not spam collector with same data.
    preferred_temporality: dict[type, AggregationTemporality] = {
        Counter: AggregationTemporality.DELTA,
        Histogram: AggregationTemporality.DELTA,
        ObservableCounter: AggregationTemporality.DELTA,
    }
    metric_exporter = OTLPMetricExporter(preferred_temporality=preferred_temporality)
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    metric_provider = MeterProvider(metric_readers=[metric_reader], resource=resource)
    metrics.set_meter_provider(metric_provider)
    return metric_provider


def _setup_otel_tracing(resource: Any, multiprocessing: bool = False) -> TracerProvider:
    otlp_exporter = OTLPSpanExporter()
    trace_provider = TracerProvider(resource=resource)
    if multiprocessing:
        trace_provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))
    else:
        trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    trace.set_tracer_provider(trace_provider)
    return trace_provider


def setup_otel(app: Any, settings: Any) -> Optional[OTELProviders]:
    """Setups OTEL tracer, metrics, and logging.

    OTEL is configured by OTEL_EXPORTER_OTLP_ENDPOINT and
    OTEL_EXPORTER_OTLP_HEADERS env vars set by DR.
    """
    # Can be used to disable OTEL reporting from env var parameters
    # https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/
    if RuntimeParameters.has("OTEL_SDK_DISABLED"):
        try:
            if RuntimeParameters.get("OTEL_SDK_DISABLED"):
                logger.info("OTEL explicitly disabled")
                return None
        except ValueError as e:
            logger.warning("Invalid OTEL_SDK_DISABLED runtime parameter: %s", e)

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        logger.info("OTEL is not configured (OTEL_EXPORTER_OTLP_ENDPOINT not set)")
        return None

    resource = Resource.create()
    trace_provider = _setup_otel_tracing(resource=resource)
    logger_provider = _setup_otel_logging(resource=resource)
    metric_provider = _setup_otel_metrics(resource=resource)

    prefix = os.environ.get("URL_PREFIX", "").rstrip("/")
    excluded_urls = f"^{prefix}/$,^{prefix}/ping/$,^{prefix}/health/$"
    FastAPIInstrumentor.instrument_app(
        app, tracer_provider=trace_provider, excluded_urls=excluded_urls
    )

    return OTELProviders(
        trace_provider=trace_provider,
        metric_provider=metric_provider,
        logger_provider=logger_provider,
    )
