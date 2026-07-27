import calendar
import time
import uuid
from typing import Any

from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessage
from openai.types.chat import CompletionCreateParams
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.model import Model


async def load_model(code_dir: str) -> Any:
    return "dummy"


async def get_supported_llm_models(model: Any):
    return [
        Model(
            id="datarobot_llm_id",
            created=1744854432,
            object="model",
            owned_by="tester@datarobot.com",
        )
    ]


async def chat(completion_create_params: CompletionCreateParams, model: Any, **kwargs: Any):
    inter_token_latency_seconds = 0.01
    model_id = completion_create_params["model"]
    message_content = "Echo: " + completion_create_params["messages"][0]["content"]
    stream = completion_create_params.get("stream", False)

    if stream:

        def gen_chunks():
            chunk_id = str(uuid.uuid4())
            for token in message_content.split():
                yield ChatCompletionChunk(
                    id=chunk_id,
                    object="chat.completion.chunk",
                    created=calendar.timegm(time.gmtime()),
                    model=model_id,
                    choices=[
                        ChunkChoice(
                            finish_reason=None,
                            index=0,
                            delta=ChoiceDelta(content=token),
                        )
                    ],
                )
                time.sleep(inter_token_latency_seconds)
            yield ChatCompletionChunk(
                id=chunk_id,
                object="chat.completion.chunk",
                created=calendar.timegm(time.gmtime()),
                model=model_id,
                choices=[
                    ChunkChoice(
                        finish_reason="stop",
                        index=0,
                        delta=ChoiceDelta(),
                    )
                ],
            )

        return gen_chunks()

    return ChatCompletion(
        id="association_id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=message_content),
            )
        ],
        created=calendar.timegm(time.gmtime()),
        model=model_id,
        object="chat.completion",
    )
