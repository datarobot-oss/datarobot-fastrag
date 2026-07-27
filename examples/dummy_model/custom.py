import asyncio
import calendar
import random
import time
from typing import Iterator

import pandas as pd
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessage
from openai.types.chat import CompletionCreateParams
from openai.types.chat.chat_completion import Choice
from openai.types.model import Model
from opentelemetry import trace

tracer = trace.get_tracer("fastrag.server")


async def init(code_dir):
    with tracer.start_as_current_span("init"):
        return None


async def load_model(*args, **kwargs):
    with tracer.start_as_current_span("load_model"):
        return b"no binary model"


async def score(data, model, **kwargs):
    with tracer.start_as_current_span("score"):
        t = max(random.gauss(mu=30, sigma=10), 1)
        time.sleep(t)
        rows = data["promptText"].astype(str).tolist()
        return pd.DataFrame({"predictions": ["score: {}".format(r) for r in rows]})


async def get_supported_llm_models(model):
    with tracer.start_as_current_span("get_supported_llm_models"):
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
    with tracer.start_as_current_span("chat"):
        messages = completion_create_params.get("messages") or []
        prompt = messages[-1].get("content", "") if messages else ""
        requested_model = completion_create_params.get("model") or "fastrag-dummy-model"
        t = max(random.gauss(mu=30, sigma=10), 5)

        result = f"Sleep for {t} with prompt: {prompt}"

        await asyncio.sleep(t)

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
