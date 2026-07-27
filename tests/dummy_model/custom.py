import calendar
import time
from typing import Any
from typing import Iterator

import pandas as pd
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessage
from openai.types.chat import CompletionCreateParams
from openai.types.chat.chat_completion import Choice
from openai.types.model import Model
from opentelemetry import trace

tracer = trace.get_tracer("fastrag.server")


async def init(code_dir: str) -> None:
    with tracer.start_as_current_span("init"):
        return None


async def load_model(*args, **kwargs) -> Any:
    with tracer.start_as_current_span("load_model"):
        return b"no binary model"


async def score(data, model, **kwargs) -> pd.DataFrame:
    with tracer.start_as_current_span("score"):
        rows = data["promptText"].astype(str).tolist()
        return pd.DataFrame({"predictions": ["score: {}".format(r) for r in rows]})


async def get_supported_llm_models(model):
    return [
        Model(
            id="datarobot_llm_id",
            created=1744854432,
            object="model",
            owned_by="tester@datarobot.com",
        )
    ]


async def chat(
    completion_create_params: CompletionCreateParams, model, **kwargs
) -> ChatCompletion | Iterator[dict]:
    messages = completion_create_params.get("messages") or []
    prompt = messages[-1].get("content", "") if messages else ""
    requested_model = completion_create_params.get("model") or "fastrag-dummy-model"
    result = f"my.chat with prompt: {prompt}" if prompt else "Hello form fastrag!"

    if completion_create_params.get("stream"):
        created = calendar.timegm(time.gmtime())

        def _stream() -> Iterator[dict]:
            yield {
                "id": "association_id",
                "object": "chat.completion.chunk",
                "created": created,
                "model": requested_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": result},
                        "finish_reason": None,
                    }
                ],
            }
            yield {
                "id": "association_id",
                "object": "chat.completion.chunk",
                "created": created,
                "model": requested_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

        return _stream()

    return ChatCompletion(
        id="association_id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=result),
            )
        ],
        created=calendar.timegm(time.gmtime()),
        model=requested_model,
        object="chat.completion",
    )
