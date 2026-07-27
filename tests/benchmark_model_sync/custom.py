import time


def chat(completion_create_params, model=None, **kwargs):
    time.sleep(0.1)  # simulate 100ms LLM latency (blocks the thread)
    messages = completion_create_params.get("messages", [])
    prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return {
        "id": "chatcmpl-benchmark",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "drum-benchmark",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"echo: {prompt}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
