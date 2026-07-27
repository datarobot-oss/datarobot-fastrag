import pandas as pd


def load_model(code_dir):
    return {"name": "moderated-test-model"}


def score(data, model, **kwargs):
    # promptText is the standard column name for TextGeneration input
    return pd.DataFrame({"completion": ["test response"] * len(data)})


def chat(completion_create_params, model, **kwargs):
    return {
        "id": "test-id",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "test response"}}],
    }
