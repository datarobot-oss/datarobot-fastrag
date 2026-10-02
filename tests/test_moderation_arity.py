"""
Integration tests: chat hook signatures vs. the real moderation pipeline.

FastRAG's server passes ``target_type`` and ``headers`` to every chat hook. Without a
moderation pipeline, ``_chat_hook_kwargs`` drops whatever a fixed-signature hook does not
declare. With a pipeline, kwargs go to ``async_chat`` unfiltered and moderations decides what
to forward, so hooks must tolerate (or be shielded from) the extra kwargs.

Requires datarobot-moderations:

    uv run --group integration pytest tests/test_moderation_arity.py -m integration -v
"""

import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator

import pytest
from openai.types.chat import ChatCompletion

from fastrag.loader import HookRegistry
from fastrag.model_adapter import AsyncModelAdapter
from fastrag.model_adapter import ModelAdapter
from fastrag.model_adapter import SyncModelAdapter

pytestmark = pytest.mark.integration

_MODERATED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "moderated_model")
_CHAT_PARAMS = {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
_SERVER_KWARGS = {"target_type": "textgeneration", "headers": {"x-test": "1"}}
# moderations treats any non-ChatCompletion iterable (e.g. a dict) as a stream.
_RESPONSE = ChatCompletion.model_validate(
    {
        "id": "test-id",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "test response"},
            }
        ],
    }
)


@pytest.fixture(autouse=True)
def _require_moderations(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("datarobot_moderation_interface", reason="datarobot-moderations missing")
    monkeypatch.delenv("TARGET_TYPE", raising=False)


def _hook_shapes(received: Dict[str, Any], is_async: bool) -> Dict[str, Callable[..., Any]]:
    """Chat hooks with different signatures; each records the kwargs it was called with."""
    if is_async:

        async def two_args(params, model):  # type: ignore[no-untyped-def]
            received["kwargs"] = {}
            return _RESPONSE.model_copy(deep=True)

        async def declares_association_id(params, model, association_id=None):  # type: ignore[no-untyped-def]
            received["kwargs"] = {"association_id": association_id}
            return _RESPONSE.model_copy(deep=True)

        async def declares_target_type(params, model, target_type=None):  # type: ignore[no-untyped-def]
            received["kwargs"] = {"target_type": target_type}
            return _RESPONSE.model_copy(deep=True)

        async def var_kwargs(params, model, **kwargs):  # type: ignore[no-untyped-def]
            received["kwargs"] = kwargs
            return _RESPONSE.model_copy(deep=True)

    else:

        def two_args(params, model):  # type: ignore[no-untyped-def,misc]
            received["kwargs"] = {}
            return _RESPONSE.model_copy(deep=True)

        def declares_association_id(params, model, association_id=None):  # type: ignore[no-untyped-def,misc]
            received["kwargs"] = {"association_id": association_id}
            return _RESPONSE.model_copy(deep=True)

        def declares_target_type(params, model, target_type=None):  # type: ignore[no-untyped-def,misc]
            received["kwargs"] = {"target_type": target_type}
            return _RESPONSE.model_copy(deep=True)

        def var_kwargs(params, model, **kwargs):  # type: ignore[no-untyped-def,misc]
            received["kwargs"] = kwargs
            return _RESPONSE.model_copy(deep=True)

    return {
        "two_args": two_args,
        "declares_association_id": declares_association_id,
        "declares_target_type": declares_target_type,
        "var_kwargs": var_kwargs,
    }


def _adapter(kind: str, chat: Callable[..., Any]) -> ModelAdapter:
    hooks = HookRegistry(chat=chat)
    if kind == "async":
        return AsyncModelAdapter(hooks=hooks, code_dir=_MODERATED_MODEL_DIR)
    return SyncModelAdapter(hooks=hooks, code_dir=_MODERATED_MODEL_DIR, max_workers=1)


@pytest.fixture(params=["sync", "async"])
def adapter_kind(request: pytest.FixtureRequest) -> Iterator[str]:
    yield request.param


@pytest.mark.parametrize(
    "shape", ["two_args", "declares_association_id", "declares_target_type", "var_kwargs"]
)
async def test_chat_hook_signature_through_real_pipeline(adapter_kind: str, shape: str) -> None:
    """A fixed-signature chat hook must not be called with kwargs it doesn't declare."""
    received: Dict[str, Any] = {}
    hook = _hook_shapes(received, is_async=adapter_kind == "async")[shape]
    adapter = _adapter(adapter_kind, hook)
    await adapter.initialize()
    try:
        assert adapter._mod_pipeline is not None, "expected a real moderation pipeline"
        result = await adapter.chat(dict(_CHAT_PARAMS), **_SERVER_KWARGS)
    finally:
        adapter.shutdown()

    assert received, "chat hook was never called"
    if shape == "declares_target_type":
        assert received["kwargs"] == {"target_type": "textgeneration"}
    elif shape == "declares_association_id":
        assert "target_type" not in received["kwargs"]
    elif shape == "two_args":
        assert received["kwargs"] == {}
    assert result.choices[0].message.content == "test response"
