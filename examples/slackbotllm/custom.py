import logging
from typing import Iterator

import pandas as pd
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import CompletionCreateParams
from openai.types.model import Model
from opentelemetry import trace
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

from fastrag import RuntimeParameters

OpenAIInstrumentor().instrument()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
tracer = trace.get_tracer(__name__)


async def init(code_dir):
    logger.info(f"Custom model init called with code_dir: {code_dir}")


async def load_model(*args, **kwargs):
    client = AsyncAzureOpenAI(
        azure_endpoint=RuntimeParameters.get("AZURE_OPENAI_ENDPOINT"),
        api_key=RuntimeParameters.get("AZURE_OPENAI_API_KEY")["apiToken"],
        azure_deployment=RuntimeParameters.get("AZURE_DEPLOYMENT"),
        api_version=RuntimeParameters.get("AZURE_API_VERSION"),
    )
    return client


async def chat(
    completion_create_params: CompletionCreateParams, model, **kwargs
) -> ChatCompletion | Iterator[ChatCompletionChunk]:
    client = model
    response = await client.chat.completions.create(**completion_create_params)
    return response


async def score(data: pd.DataFrame, model, **kwargs):
    client = model
    prompts = data["promptText"].tolist()
    responses = []

    for prompt in prompts:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"{prompt}"},
            ],
            temperature=0,
        )
        responses.append(response.choices[0].message.content)
    return pd.DataFrame({"resultText": responses})


async def get_supported_llm_models(model):
    return [
        Model(
            id="datarobot_llm_id",
            created=1744854432,
            object="model",
            owned_by="tester@datarobot.com",
        )
    ]
