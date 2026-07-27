from fastapi import FastAPI

from fastrag.telemetry import setup_otel


def test_setup_otel_returns_providers(monkeypatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    providers = setup_otel(FastAPI(), settings=None)

    assert providers is not None
    assert providers.trace_provider is not None
    assert providers.metric_provider is not None
    assert providers.logger_provider is not None

    providers.trace_provider.shutdown()
    providers.metric_provider.shutdown()
    providers.logger_provider.shutdown()
