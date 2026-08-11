import asyncio
import time


async def chat(completion_create_params, model=None, **kwargs):
    """
    Async RAG benchmark: two sequential async I/O operations.

    50ms  — vector DB similarity search
    200ms — LLM completion call

    Under DRUM (sync, thread-per-request) each in-flight request holds a thread
    for the full 250ms. Under FRAG the event loop yields at each await, so
    throughput scales with concurrency instead of thread count.
    """
    await asyncio.sleep(0.05)  # vector DB
    await asyncio.sleep(0.20)  # LLM
    messages = completion_create_params.get("messages", [])
    prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return {
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
