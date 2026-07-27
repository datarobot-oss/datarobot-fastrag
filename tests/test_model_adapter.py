import threading
from typing import Any

import pandas as pd
import pytest
from opentelemetry import context

from fastrag.loader import HookRegistry
from fastrag.model_adapter import AsyncModelAdapter
from fastrag.model_adapter import SyncModelAdapter


@pytest.mark.asyncio
async def test_async_model_adapter_hooks_execution() -> None:
    async def init(code_dir: str) -> None:
        return None

    async def load_model(code_dir: str) -> dict[str, str]:
        return {"name": "test_model"}

    async def score(data: pd.DataFrame, model: dict[str, str], **kwargs: Any) -> dict[str, Any]:
        return {"predictions": [1], "model_name": model["name"], "extra": kwargs.get("extra")}

    async def chat(params: dict[str, Any], model: dict[str, str], **kwargs: Any) -> dict[str, str]:
        return {"chat": "ok", "model_name": model["name"]}

    async def score_unstructured(
        data: bytes, model: dict[str, str], **kwargs: Any
    ) -> dict[str, Any]:
        return {"unstructured": data, "model_name": model["name"]}

    async def get_supported_llm_models(model: dict[str, str]) -> list[str]:
        return ["model1", "model2"]

    hooks = HookRegistry(
        init=init,
        load_model=load_model,
        score=score,
        chat=chat,
        score_unstructured=score_unstructured,
        get_supported_llm_models=get_supported_llm_models,
    )
    adapter = AsyncModelAdapter(hooks=hooks, code_dir=".")
    await adapter.initialize()

    assert adapter.model == {"name": "test_model"}
    assert await adapter.score(pd.DataFrame({"a": [1]}), extra="foo") == {
        "predictions": [1],
        "model_name": "test_model",
        "extra": "foo",
    }
    assert await adapter.chat({"msg": "hi"}) == {"chat": "ok", "model_name": "test_model"}
    assert await adapter.score_unstructured(b"raw") == {
        "unstructured": b"raw",
        "model_name": "test_model",
    }
    assert await adapter.get_supported_llm_models() == ["model1", "model2"]


@pytest.mark.asyncio
async def test_sync_model_adapter_context_propagation() -> None:
    key = context.create_key("test_key")

    def score(data: pd.DataFrame, model: Any, **kwargs: Any) -> str:
        return context.get_value(key)

    hooks = HookRegistry(score=score)
    adapter = SyncModelAdapter(hooks=hooks, code_dir=".", max_workers=2)
    await adapter.initialize()

    token = context.attach(context.set_value(key, "test_value"))
    try:
        assert await adapter.score(pd.DataFrame({"a": [1]})) == "test_value"
    finally:
        context.detach(token)
        adapter.shutdown()


@pytest.mark.asyncio
async def test_sync_model_adapter_uses_thread_local_model() -> None:
    def load_model(code_dir: str) -> int:
        return threading.get_ident()

    def score(data: pd.DataFrame, model: int, **kwargs: Any) -> dict[str, int]:
        return {"model_thread_id": model, "current_thread_id": threading.get_ident()}

    hooks = HookRegistry(load_model=load_model, score=score)
    adapter = SyncModelAdapter(hooks=hooks, code_dir=".", max_workers=2)
    await adapter.initialize()

    result = await adapter.score(pd.DataFrame({"a": [1]}))
    assert result["model_thread_id"] == result["current_thread_id"]
    adapter.shutdown()
