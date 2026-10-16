"""
Tests for moderation pipeline integration in SyncModelAdapter.

Unit tests use a mock pipeline and run without any external dependencies.
Integration tests require datarobot-moderations to be installed:

    uv run --group integration pytest tests/test_moderation.py -m integration -v
"""

import asyncio
import os
import threading
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from fastrag.loader import HookRegistry
from fastrag.model_adapter import SyncModelAdapter

_MODERATED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "moderated_model")
_DUMMY_DF = pd.DataFrame({"promptText": ["hello", "world"]})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(code_dir: str = ".", max_workers: int = 1, **hooks: Any) -> SyncModelAdapter:
    registry = HookRegistry(**hooks)
    return SyncModelAdapter(hooks=registry, code_dir=code_dir, max_workers=max_workers)


def _dummy_score(data: pd.DataFrame, model: Any, **kwargs: Any) -> pd.DataFrame:
    return pd.DataFrame({"completion": ["response"] * len(data)})


def _dummy_chat(params: dict, model: Any, **kwargs: Any) -> dict:
    return {"object": "chat.completion", "choices": [{"message": {"content": "hi"}}]}


# ---------------------------------------------------------------------------
# Unit tests — mock pipeline, no datarobot_dome required
# ---------------------------------------------------------------------------


async def test_no_moderation_config_skips_pipeline() -> None:
    """Adapter with no moderation_config.yaml must not load a pipeline."""
    adapter = _make_adapter(code_dir=".", score=_dummy_score)
    await adapter.initialize()
    try:
        assert adapter._mod_pipeline is None
    finally:
        adapter.shutdown()


async def test_moderation_config_missing_datarobot_dome_logs_warning() -> None:
    """moderation_config.yaml present but datarobot_dome not installed.

    Expects pipeline to be None and a warning to be logged.
    """
    mods = {"datarobot_dome": None, "datarobot_dome.drum_integration": None}
    with patch.dict("sys.modules", mods):
        adapter = _make_adapter(code_dir=_MODERATED_MODEL_DIR, score=_dummy_score)
        await adapter.initialize()
        try:
            assert adapter._mod_pipeline is None
        finally:
            adapter.shutdown()


async def test_score_calls_mod_pipeline_when_present() -> None:
    """score() must delegate to pipeline.async_score(data, model, score_hook)."""
    mock_pipeline = MagicMock()
    mock_pipeline.async_score = AsyncMock(
        return_value=pd.DataFrame({"completion": ["moderated"] * len(_DUMMY_DF)})
    )

    adapter = _make_adapter(score=_dummy_score)
    await adapter.initialize()
    adapter._mod_pipeline = mock_pipeline
    try:
        result = await adapter.score(_DUMMY_DF)
        assert mock_pipeline.async_score.call_count == 1
        call_args = mock_pipeline.async_score.call_args
        # First positional arg is data, second is model, third is the score hook
        assert isinstance(call_args.args[0], pd.DataFrame)
        assert call_args.args[2] is _dummy_score
        assert list(result["completion"]) == ["moderated", "moderated"]
    finally:
        adapter.shutdown()


async def test_chat_calls_mod_pipeline_when_present() -> None:
    """chat() must delegate to pipeline.async_chat(params, model, chat_hook)."""
    mock_pipeline = MagicMock()
    mock_pipeline.async_chat = AsyncMock(return_value={"object": "chat.completion", "choices": []})

    adapter = _make_adapter(score=_dummy_score, chat=_dummy_chat)
    await adapter.initialize()
    adapter._mod_pipeline = mock_pipeline
    try:
        params = {"messages": [{"role": "user", "content": "hi"}]}
        await adapter.chat(params, association_id="assoc-1")
        assert mock_pipeline.async_chat.call_count == 1
        call_args = mock_pipeline.async_chat.call_args
        assert call_args.args[0] == params
        assert call_args.args[2] is _dummy_chat
        assert call_args.kwargs["association_id"] == "assoc-1"
    finally:
        adapter.shutdown()


class _BlockingUploadPipeline:
    """Mimics datarobot_dome: uploads metrics with ``run_until_complete`` on one shared loop."""

    def __init__(self) -> None:
        self.upload_threads: list[str] = []
        self._loop = asyncio.new_event_loop()

    def upload_custom_metrics(self, payload: Any) -> None:
        self._loop.run_until_complete(asyncio.sleep(0.01))
        self.upload_threads.append(threading.current_thread().name)

    def report_custom_metrics(self, result_df: Any) -> None:
        self.upload_custom_metrics({"buckets": [result_df]})


async def test_metric_uploads_from_event_loop_run_off_loop() -> None:
    """A blocking upload made on the event loop thread must not raise and must still run."""
    inner = _BlockingUploadPipeline()
    adapter = _make_adapter(chat=_dummy_chat)
    adapter._offload_sync_metric_uploads(MagicMock(_pipeline=inner))

    # Without the offload this raises "Cannot run the event loop while another loop is running".
    inner.report_custom_metrics("row")
    adapter.shutdown()

    assert len(inner.upload_threads) == 1
    assert inner.upload_threads[0].startswith("fastrag-metrics")


async def test_concurrent_metric_uploads_do_not_overlap(caplog: pytest.LogCaptureFixture) -> None:
    """Streaming-thread and event-loop uploads share one loop, so they must not overlap."""
    inner = _BlockingUploadPipeline()
    adapter = _make_adapter(chat=_dummy_chat)
    adapter._offload_sync_metric_uploads(MagicMock(_pipeline=inner))

    # Streaming uploads come from their own threads; the event loop adds a non-streaming one.
    threads = [
        threading.Thread(target=inner.report_custom_metrics, args=("row",), name=f"stream-{i}")
        for i in range(5)
    ]
    for thread in threads:
        thread.start()
    inner.report_custom_metrics("row")
    for thread in threads:
        thread.join()
    adapter.shutdown()

    # Overlapping run_until_complete calls on one loop raise "This event loop is already running".
    assert "Moderation custom metrics upload failed." not in caplog.text
    assert len(inner.upload_threads) == 6
    assert all(name.startswith("fastrag-metrics") for name in inner.upload_threads)


async def test_score_bypasses_pipeline_when_none() -> None:
    """score() must call the hook directly when no pipeline is loaded."""
    called_with: list[Any] = []

    def score_hook(data: pd.DataFrame, model: Any, **kwargs: Any) -> pd.DataFrame:
        called_with.append(data)
        return pd.DataFrame({"completion": ["direct"] * len(data)})

    adapter = _make_adapter(score=score_hook)
    await adapter.initialize()
    assert adapter._mod_pipeline is None
    try:
        result = await adapter.score(_DUMMY_DF)
        assert len(called_with) == 1
        assert list(result["completion"]) == ["direct", "direct"]
    finally:
        adapter.shutdown()


async def test_resolve_target_type_from_env(tmp_path: Any) -> None:
    """TARGET_TYPE env var must take priority over model-metadata.yaml."""
    (tmp_path / "moderation_config.yaml").write_text(
        "guards: []\ntimeout_action: score\ntimeout_sec: 60\n"
    )
    adapter = _make_adapter(code_dir=str(tmp_path), score=_dummy_score)
    old = os.environ.pop("TARGET_TYPE", None)
    try:
        os.environ["TARGET_TYPE"] = "TextGeneration"
        assert adapter._resolve_target_type() == "textgeneration"
    finally:
        if old is not None:
            os.environ["TARGET_TYPE"] = old
        else:
            os.environ.pop("TARGET_TYPE", None)
        adapter.shutdown()


async def test_resolve_target_type_from_metadata(tmp_path: Any) -> None:
    """target_type must be read from model-metadata.yaml when no env var is set."""
    (tmp_path / "model-metadata.yaml").write_text("targetType: TextGeneration\n")
    adapter = _make_adapter(code_dir=str(tmp_path), score=_dummy_score)
    os.environ.pop("TARGET_TYPE", None)
    try:
        assert adapter._resolve_target_type() == "textgeneration"
    finally:
        adapter.shutdown()


async def test_resolve_target_type_defaults_to_regression(tmp_path: Any) -> None:
    """Falls back to 'regression' when neither env var nor model-metadata.yaml is present."""
    os.environ.pop("TARGET_TYPE", None)
    adapter = _make_adapter(code_dir=str(tmp_path), score=_dummy_score)
    try:
        assert adapter._resolve_target_type() == "regression"
    finally:
        adapter.shutdown()


# ---------------------------------------------------------------------------
# Integration tests — require datarobot-moderations
# (run with: uv run --group integration pytest -m integration)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_moderation_pipeline_loads_from_config_dir() -> None:
    """moderation_pipeline_factory must return a pipeline for a TextGeneration model."""
    pytest.importorskip("datarobot_dome", reason="datarobot-moderations not installed")
    old = os.environ.pop("TARGET_TYPE", None)
    try:
        adapter = _make_adapter(
            code_dir=_MODERATED_MODEL_DIR,
            score=_dummy_score,
            chat=_dummy_chat,
        )
        await adapter.initialize()
        try:
            assert adapter._mod_pipeline is not None, (
                "Expected a moderation pipeline for TextGeneration model"
                " with moderation_config.yaml"
            )
        finally:
            adapter.shutdown()
    finally:
        if old is not None:
            os.environ["TARGET_TYPE"] = old


@pytest.mark.integration
async def test_score_flows_through_real_pipeline() -> None:
    """End-to-end: real moderation pipeline must wrap score() without errors."""
    pytest.importorskip("datarobot_dome", reason="datarobot-moderations not installed")
    old = os.environ.pop("TARGET_TYPE", None)
    try:
        adapter = _make_adapter(
            code_dir=_MODERATED_MODEL_DIR,
            score=_dummy_score,
            max_workers=2,
        )
        await adapter.initialize()
        try:
            assert adapter._mod_pipeline is not None
            results = await asyncio.gather(*[adapter.score(_DUMMY_DF) for _ in range(5)])
            assert len(results) == 5
            for r in results:
                assert len(r) == len(_DUMMY_DF)
        finally:
            adapter.shutdown()
    finally:
        if old is not None:
            os.environ["TARGET_TYPE"] = old


@pytest.mark.integration
async def test_score_without_moderation_config_uses_hook_directly() -> None:
    """No moderation_config.yaml → pipeline is None even with datarobot_dome installed."""
    pytest.importorskip("datarobot_dome", reason="datarobot-moderations not installed")
    adapter = _make_adapter(code_dir=".", score=_dummy_score)
    await adapter.initialize()
    try:
        assert adapter._mod_pipeline is None
        result = await adapter.score(_DUMMY_DF)
        assert len(result) == len(_DUMMY_DF)
    finally:
        adapter.shutdown()
