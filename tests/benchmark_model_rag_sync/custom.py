import time


def chat(completion_create_params, model=None, **kwargs):
    """
    Sync RAG benchmark: same 250ms latency as the async version, but blocking.

    50ms  — vector DB similarity search (blocks thread)
    200ms — LLM completion call (blocks thread)
    """
    time.sleep(0.05)  # vector DB
    time.sleep(0.20)  # LLM
    messages = completion_create_params.get("messages", [])
    prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return {
        "id": "chatcmpl-rag-benchmark",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "drum-rag-benchmark",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"RAG response: {prompt}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
