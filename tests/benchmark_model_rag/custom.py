import asyncio
import time

from openai.types.chat import ChatCompletion


async def chat(completion_create_params, model=None, **kwargs):
    """
    Async RAG benchmark: two sequential async I/O operations.

    50ms  — vector DB similarity search
    200ms — LLM completion call

    Under DRUM (sync, thread-per-request) each in-flight request holds a thread
    for the full 250ms. Under FRAG the event loop yields at each await, so
    throughput scales with concurrency instead of thread count.

    Returns a ChatCompletion rather than a dict: fastrag accepts either, DRUM
    accepts only the former, and the sync twin of this model must return the same
    shape for the two to be comparable.
    """
    await asyncio.sleep(0.05)  # vector DB
    await asyncio.sleep(0.20)  # LLM
    messages = completion_create_params.get("messages", [])
    prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-rag-benchmark",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "frag-rag-benchmark",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"RAG response: {prompt}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
    )
