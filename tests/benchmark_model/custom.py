import asyncio
import time


async def chat(completion_create_params, model=None, **kwargs):
    """
    Async benchmark hook. Simulates 100ms I/O-bound LLM latency.

    Under DRUM (sync, thread-per-request) each concurrent request occupies a thread
    for the full 100ms sleep. Under FastDRUM the event loop serves other requests
    while this request awaits, so concurrency throughput should scale linearly.
    """
    await asyncio.sleep(0.1)
    messages = completion_create_params.get("messages", [])
    prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return {
        "id": "chatcmpl-benchmark",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "fastrag-benchmark",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"echo: {prompt}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
