import json
import logging
import os
import time
from collections.abc import AsyncIterable
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Coroutine
from collections.abc import Iterable
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import FastAPI
from fastapi import File
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from opentelemetry import context as otel_context
from opentelemetry import propagate
from opentelemetry import trace

from .loader import HookLoader
from .loader import HookName
from .model_adapter import ModelAdapter
from .model_metadata import ModelMetadata
from .model_metadata import TargetType
from .monitoring import MLOpsReporter
from .monitoring import elapsed_ms
from .schemas import CapabilitiesResponse
from .schemas import HealthResponse
from .schemas import InfoResponse
from .schemas import OpenAIChatCompletionRequest
from .schemas import OpenAIChatCompletionResponse
from .schemas import PredictionResponse
from .settings import Settings
from .settings import load_settings
from .telemetry import OTELProviders
from .telemetry import setup_otel
from .validation import ApiError
from .validation import InternalServerError
from .validation import NotFoundError
from .validation import NotImplementedApiError
from .validation import UnprocessableEntityError
from .validation import format_prediction_response
from .validation import parse_content_type
from .validation import read_csv_or_raise
from .validation import read_structured_payload
from .validation import resolve_incoming_unstructured_data
from .validation import resolve_outgoing_unstructured_data
from .validation import target_type_is_unstructured

logger = logging.getLogger("fastrag.server")


def _ensure_hook_available(
    model_adapter: ModelAdapter,
    hook_name: HookName,
) -> None:
    if not model_adapter.hooks.has(hook_name):
        raise NotImplementedApiError(detail=f"{hook_name.value} hook not implemented.")


URL_PREFIX = os.environ.get("URL_PREFIX", "").rstrip("/")


class TracedRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        route_handler = super().get_route_handler()
        if self.path in {f"{URL_PREFIX}/", f"{URL_PREFIX}/ping/", f"{URL_PREFIX}/health/"}:
            return route_handler

        tracer = trace.get_tracer("fastrag.server")

        async def traced_route_handler(request: Request) -> Response:
            span_name = f"handler {request.method} {self.path}"
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("http.method", request.method)
                span.set_attribute("http.route", self.path)
                return await route_handler(request)

        return traced_route_handler


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = load_settings()
    app.state.settings = settings

    otel_providers: OTELProviders | None = setup_otel(app, settings)
    if settings.runtime_params_file:
        model_metadata = ModelMetadata.from_yaml(settings.runtime_params_file)
    else:
        model_metadata = ModelMetadata()

    model_metadata.merge_env_overrides()
    app.state.model_metadata = model_metadata

    hook_loader = HookLoader(settings.code_dir, max_workers=settings.thread_pool_workers)
    model_adapter = await hook_loader.load()
    app.state.model_adapter = model_adapter

    mlops_reporter = MLOpsReporter()
    mlops_reporter.initialize()
    app.state.mlops_reporter = mlops_reporter

    try:
        yield
    finally:
        mlops_reporter.shutdown()
        model_adapter.shutdown()
        if otel_providers is not None:
            otel_providers.trace_provider.shutdown()
            otel_providers.metric_provider.shutdown()
            otel_providers.logger_provider.shutdown()  # type: ignore[no-untyped-call]


router = APIRouter(prefix=URL_PREFIX, route_class=TracedRoute)


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_model_metadata(request: Request) -> ModelMetadata:
    model_metadata: ModelMetadata = request.app.state.model_metadata
    return model_metadata


def get_model_adapter(request: Request) -> ModelAdapter:
    model_adapter: ModelAdapter = request.app.state.model_adapter
    return model_adapter


def get_mlops_reporter(request: Request) -> MLOpsReporter:
    reporter: MLOpsReporter = request.app.state.mlops_reporter
    return reporter


@router.get("/", response_model=HealthResponse)
@router.get("/ping/", response_model=HealthResponse)
@router.get("/health/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(message="ok")


@router.get("/info/", response_model=InfoResponse)
async def info(
    settings: Settings = Depends(get_settings),
    model_metadata: ModelMetadata = Depends(get_model_metadata),
    model_adapter: ModelAdapter = Depends(get_model_adapter),
) -> InfoResponse:
    return InfoResponse(
        server="fastrag",
        status="ok",
        model_loaded=model_adapter.model is not None,
        code_dir=settings.code_dir,
        target_type=model_metadata.target_type,
        positive_class_label=model_metadata.inference_model.positive_class_label,
        negative_class_label=model_metadata.inference_model.negative_class_label,
        class_labels=model_metadata.inference_model.class_labels,
    )


@router.post("/predict/", response_model=PredictionResponse)
@router.post("/predictions/", response_model=PredictionResponse)
@router.post("/invocations", response_model=PredictionResponse)
async def predict(
    request: Request,
    X: UploadFile = File(None),
    model_adapter: ModelAdapter = Depends(get_model_adapter),
    model_metadata: ModelMetadata = Depends(get_model_metadata),
    reporter: MLOpsReporter = Depends(get_mlops_reporter),
) -> PredictionResponse:
    content, _ = await read_structured_payload(request, X)
    df = read_csv_or_raise(content)

    _ensure_hook_available(model_adapter, HookName.SCORE)

    kwargs = {
        "positive_class_label": model_metadata.inference_model.positive_class_label,
        "negative_class_label": model_metadata.inference_model.negative_class_label,
        "class_labels": model_metadata.inference_model.class_labels,
        "target_type": model_metadata.target_type,
        "target_name": model_metadata.inference_model.target_name,
    }

    start = time.monotonic()
    try:
        predictions = await _await_or_api_error(
            model_adapter.score(df, **kwargs),
            detail="Prediction failed.",
            log_message="Prediction failed.",
        )
        response = format_prediction_response(predictions, model_metadata)
    except ValueError as exc:
        reporter.report_error(elapsed_ms(start))
        raise UnprocessableEntityError(
            detail=str(exc),
            log_message="Prediction output validation failed.",
        ) from exc
    except Exception:
        reporter.report_error(elapsed_ms(start))
        raise

    reporter.report_predictions(_prediction_count(predictions), elapsed_ms(start))
    return response


@router.post(
    "/chat/completions",
    response_model=OpenAIChatCompletionResponse,
    response_model_exclude_none=True,
)
@router.post(
    "/v1/chat/completions",
    response_model=OpenAIChatCompletionResponse,
    response_model_exclude_none=True,
)
async def chat_completions(
    body: OpenAIChatCompletionRequest,
    model_adapter: ModelAdapter = Depends(get_model_adapter),
    model_metadata: ModelMetadata = Depends(get_model_metadata),
    reporter: MLOpsReporter = Depends(get_mlops_reporter),
) -> Any:
    _ensure_hook_available(model_adapter, HookName.CHAT)

    target_type = model_metadata.target_type
    kwargs = {"target_type": target_type}

    # Count one prediction per successful chat completion (DRUM parity). For
    # streaming responses the count is reported only after the stream is fully
    # drained, so cancelled/failed streams are not counted -- see the generators
    # in _format_chat_response.
    start = time.monotonic()
    try:
        response = await _await_or_api_error(
            model_adapter.chat(body.model_dump(exclude_none=True), **kwargs),
            detail="Chat completion failed.",
            log_message="Chat completion failed.",
        )
    except Exception:
        reporter.report_error(elapsed_ms(start))
        raise

    return _format_chat_response(
        response,
        on_complete=lambda: reporter.report_chat(elapsed_ms(start)),
        on_error=lambda: reporter.report_error(elapsed_ms(start)),
    )


@router.post("/predictUnstructured/")
@router.post("/predictionsUnstructured/")
async def predict_unstructured(
    request: Request,
    model_adapter: ModelAdapter = Depends(get_model_adapter),
    model_metadata: ModelMetadata = Depends(get_model_metadata),
) -> Response:
    # Intentionally no MLOps reporting here: unstructured models use DRUM's
    # *embedded* monitoring (MONITOR_EMBEDDED), where the score_unstructured hook
    # self-reports via the mlops library. Counting here too would double-count.
    target_type = model_metadata.target_type
    if not target_type_is_unstructured(target_type):
        raise UnprocessableEntityError(
            detail=(
                f"This model has target type {target_type}, "
                "use either /predict/ or /predictions/ endpoint."
            ),
        )

    _ensure_hook_available(model_adapter, HookName.SCORE_UNSTRUCTURED)

    raw_body = await request.body()
    mimetype, charset = parse_content_type(request.headers.get("content-type"))
    data, mimetype, charset = resolve_incoming_unstructured_data(raw_body, mimetype, charset)

    kwargs: dict[str, Any] = {
        "mimetype": mimetype,
        "query": dict(request.query_params),
        "headers": dict(request.headers),
    }
    if charset is not None:
        kwargs["charset"] = charset

    result = await _await_or_api_error(
        model_adapter.score_unstructured(data, **kwargs),
        detail="Prediction failed.",
        log_message="Unstructured prediction failed.",
    )

    response_data, response_mimetype, response_charset = resolve_outgoing_unstructured_data(result)

    response = Response(response_data)
    if response_mimetype is not None:
        content_type = response_mimetype
        if response_charset is not None:
            content_type += f"; charset={response_charset}"
        response.headers["Content-Type"] = content_type

    return response


@router.get("/capabilities/", response_model=CapabilitiesResponse)
async def capabilities(
    model_adapter: ModelAdapter = Depends(get_model_adapter),
) -> CapabilitiesResponse:
    supports_chat = model_adapter.hooks.has(HookName.CHAT)
    return CapabilitiesResponse(
        supported_payload_formats={"text/csv": "1.0"},
        supported_methods={"chat": supports_chat},
    )


@router.get("/models")
@router.get("/v1/models")
async def get_supported_llm_models(
    model_adapter: ModelAdapter = Depends(get_model_adapter),
    model_metadata: ModelMetadata = Depends(get_model_metadata),
) -> Any:
    target_type = model_metadata.target_type
    if target_type != TargetType.TEXT_GENERATION:
        raise NotFoundError(
            detail="get_supported_llm_models is supported only for TextGen models",
        )
    _ensure_hook_available(model_adapter, HookName.GET_SUPPORTED_LLM_MODELS)
    return await _await_or_api_error(
        model_adapter.get_supported_llm_models(),
        detail="Models endpoint failed.",
        log_message="Models endpoint failed.",
    )


def _prediction_count(predictions: Any) -> int:
    """Number of predictions in a structured scoring result (row count).

    Mirrors DRUM's ``num_predictions=len(predictions)``; falls back to 1 for
    return types without a length.
    """
    try:
        return len(predictions)
    except TypeError:
        return 1


def _format_chat_response(
    response: Any,
    on_complete: Callable[[], None],
    on_error: Callable[[], None],
) -> Any:
    # Metering lives in the wrappers below, keeping the chunk formatters pure:
    # on_complete fires once the response is fully produced (after the stream is
    # drained for streaming), on_error if producing it raises.
    if _is_async_streaming_response(response):
        return StreamingResponse(
            _meter_async(_astream_openai_chunks(response), on_complete, on_error),
            media_type="text/event-stream",
        )
    if _is_streaming_response(response):
        return StreamingResponse(
            _meter_sync(_stream_openai_chunks(response), on_complete, on_error),
            media_type="text/event-stream",
        )
    # Serialize first, then count: a conversion failure must report an error, not
    # a success, so failed responses never inflate the Total Predictions counter.
    try:
        payload = _to_jsonable(response)
    except Exception:
        on_error()
        raise
    on_complete()
    return payload


def _meter_sync(
    lines: Iterator[str], on_complete: Callable[[], None], on_error: Callable[[], None]
) -> Iterator[str]:
    """Pass a sync line stream through, firing on_complete on drain / on_error on failure."""
    try:
        yield from lines
    except Exception:
        on_error()
        raise
    on_complete()


async def _meter_async(
    lines: AsyncIterator[str], on_complete: Callable[[], None], on_error: Callable[[], None]
) -> AsyncIterator[str]:
    """Async counterpart of _meter_sync."""
    try:
        async for line in lines:
            yield line
    except Exception:
        on_error()
        raise
    on_complete()


def _is_non_streaming(response: Any) -> bool:
    if isinstance(response, (dict, str, bytes)):
        return True
    return getattr(response, "object", None) == "chat.completion"


def _is_async_streaming_response(response: Any) -> bool:
    if _is_non_streaming(response):
        return False
    return hasattr(response, "__aiter__")


def _is_streaming_response(response: Any) -> bool:
    if _is_non_streaming(response):
        return False
    return hasattr(response, "__iter__")


def _to_jsonable(response: Any) -> Any:
    for method_name in ("model_dump", "dict", "to_dict"):
        if method := getattr(response, method_name, None):
            return method()
    return response


def _stream_openai_chunks(stream: Iterable[Any]) -> Iterator[str]:
    for chunk in stream:
        yield from _format_chunk_as_sse_lines(chunk)

    yield "data: [DONE]\n\n"


async def _astream_openai_chunks(stream: AsyncIterable[Any]) -> AsyncIterator[str]:
    async for chunk in stream:
        for line in _format_chunk_as_sse_lines(chunk):
            yield line

    yield "data: [DONE]\n\n"


def _format_chunk_as_sse_lines(chunk: Any) -> list[str]:
    if hasattr(chunk, "to_json"):
        chunk_json = chunk.to_json(indent=None)
    else:
        chunk_json = json.dumps(_to_jsonable(chunk))
    lines = [f"data: {line}\n" for line in chunk_json.splitlines()]
    lines.append("\n")
    return lines


async def attach_trace_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if request.url.path in {f"{URL_PREFIX}/", f"{URL_PREFIX}/ping/", f"{URL_PREFIX}/health/"}:
        return await call_next(request)

    incoming_context = propagate.extract(request.headers)
    token = otel_context.attach(incoming_context)
    try:
        return await call_next(request)
    finally:
        otel_context.detach(token)


async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
    if exc.log_message:
        logger.error(exc.log_message, exc_info=exc.__cause__ or exc)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


async def _await_or_api_error(awaitable: Awaitable[Any], detail: str, log_message: str) -> Any:
    try:
        return await awaitable
    except ApiError:
        raise
    except Exception as exc:
        raise InternalServerError(detail=detail, log_message=log_message) from exc


def main() -> FastAPI:
    app = FastAPI(
        title="fastrag",
        description="",
        version="0.0.1",
        terms_of_service="https://www.datarobot.com/terms-of-service/",
        contact={
            "name": "DataRobot, Inc.",
            "url": "https://www.datarobot.com/contact-us/",
            "email": "support@datarobot.com",
        },
        lifespan=lifespan,
    )

    app.exception_handler(ApiError)(handle_api_error)
    app.exception_handler(Exception)(handle_unexpected_error)
    app.middleware("http")(attach_trace_context)
    app.include_router(router)
    return app


app = main()
