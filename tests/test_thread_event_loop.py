"""
Smoke tests for asyncio event loop availability in SyncModelAdapter thread pool workers.

These tests target the failure mode where moderation hooks (specifically AsyncGuardExecutor
from datarobot_dome) call asyncio.get_event_loop() inside a ThreadPoolExecutor worker thread.
In Python 3.12, get_event_loop() raises RuntimeError in non-main threads that have no event
loop set. The fix is to call asyncio.set_event_loop(asyncio.new_event_loop()) inside
SyncModelAdapter._make_thread_initializer() so every worker thread owns a loop.

Tests 1-3 will FAIL on unpatched code. They document the required behaviour and act as
regression guards once the fix is applied.
"""

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from fastrag.loader import HookRegistry
from fastrag.model_adapter import SyncModelAdapter

_DUMMY_DF = pd.DataFrame({"x": [1.0, 2.0]})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(score_hook: Any, max_workers: int = 1) -> SyncModelAdapter:
    hooks = HookRegistry(score=score_hook)
    return SyncModelAdapter(hooks=hooks, code_dir=".", max_workers=max_workers)


class _MockGuardExecutor:
    """Minimal stand-in for AsyncGuardExecutor.

    AsyncGuardExecutor.__init__ (moderations/datarobot_dome/guard_executor.py:100)
    calls asyncio.get_event_loop() unconditionally. Instantiating this class inside
    a thread pool worker reproduces the exact failure.
    """

    def __init__(self) -> None:
        self.loop = asyncio.get_event_loop()


# ---------------------------------------------------------------------------
# Test 1 — event loop is available inside worker threads
# ---------------------------------------------------------------------------


async def test_event_loop_available_in_worker_thread() -> None:
    """Worker threads must have an event loop after initialize()."""
    loop_in_thread: list[asyncio.AbstractEventLoop] = []

    def score(data: pd.DataFrame, model: Any, **kwargs: Any) -> pd.DataFrame:
        loop_in_thread.append(asyncio.get_event_loop())
        return pd.DataFrame({"Prediction": [0.0] * len(data)})

    adapter = _make_adapter(score, max_workers=2)
    await adapter.initialize()
    try:
        await adapter.score(_DUMMY_DF)
        assert loop_in_thread, "score hook did not execute"
        assert isinstance(loop_in_thread[0], asyncio.AbstractEventLoop)
    finally:
        adapter.shutdown()


# ---------------------------------------------------------------------------
# Test 2 — mimics AsyncGuardExecutor instantiation in a worker thread
# ---------------------------------------------------------------------------


async def test_mock_guard_executor_in_thread_pool() -> None:
    """Instantiating a guard executor inside the worker must not raise RuntimeError."""

    def score(data: pd.DataFrame, model: Any, **kwargs: Any) -> pd.DataFrame:
        _MockGuardExecutor()  # raises RuntimeError without the fix
        return pd.DataFrame({"Prediction": [0.0] * len(data)})

    adapter = _make_adapter(score)
    await adapter.initialize()
    try:
        result = await adapter.score(_DUMMY_DF)
        assert len(result) == len(_DUMMY_DF)
    finally:
        adapter.shutdown()


# ---------------------------------------------------------------------------
# Test 3 — concurrent requests all succeed (parametrized on worker count)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_workers", [1, 4])
async def test_concurrent_score_with_guard_instantiation(max_workers: int) -> None:
    """10 concurrent requests must all return successfully regardless of worker count."""
    n_requests = 10

    def score(data: pd.DataFrame, model: Any, **kwargs: Any) -> pd.DataFrame:
        _MockGuardExecutor()
        return pd.DataFrame({"Prediction": [0.0] * len(data)})

    hooks = HookRegistry(score=score)
    adapter = SyncModelAdapter(hooks=hooks, code_dir=".", max_workers=max_workers)
    await adapter.initialize()
    try:
        results = await asyncio.gather(*[adapter.score(_DUMMY_DF) for _ in range(n_requests)])
        assert len(results) == n_requests
        for result in results:
            assert len(result) == len(_DUMMY_DF)
    finally:
        adapter.shutdown()


# ---------------------------------------------------------------------------
# Test 4 — each worker thread gets its own independent event loop
# ---------------------------------------------------------------------------


async def test_each_worker_thread_has_independent_event_loop() -> None:
    """Each worker thread should own its own loop, not share the main thread's loop."""
    n_workers = 4
    loops_seen: list[int] = []
    lock = threading.Lock()

    def score(data: pd.DataFrame, model: Any, **kwargs: Any) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        with lock:
            loops_seen.append(id(loop))
        return pd.DataFrame({"Prediction": [0.0] * len(data)})

    hooks = HookRegistry(score=score)
    adapter = SyncModelAdapter(hooks=hooks, code_dir=".", max_workers=n_workers)
    await adapter.initialize()
    try:
        await asyncio.gather(*[adapter.score(_DUMMY_DF) for _ in range(n_workers * 3)])
        main_loop_id = id(asyncio.get_event_loop())
        assert all(lid != main_loop_id for lid in loops_seen), (
            "A worker thread is sharing the main event loop"
        )
    finally:
        adapter.shutdown()


# ---------------------------------------------------------------------------
# Test 5 — real AsyncGuardExecutor (integration, skipped if not installed)
# ---------------------------------------------------------------------------


async def test_real_async_guard_executor_in_thread_pool() -> None:
    """AsyncGuardExecutor from datarobot_dome must instantiate without error in a worker."""
    guard_executor_mod = pytest.importorskip(
        "datarobot_dome.guard_executor", reason="datarobot_dome not installed"
    )
    AsyncGuardExecutor = guard_executor_mod.AsyncGuardExecutor

    def score(data: pd.DataFrame, model: Any, **kwargs: Any) -> pd.DataFrame:
        mock_pipeline = MagicMock()
        AsyncGuardExecutor(mock_pipeline)  # requires a pipeline arg; mock satisfies it
        return pd.DataFrame({"Prediction": [0.0] * len(data)})

    adapter = _make_adapter(score)
    await adapter.initialize()
    try:
        result = await adapter.score(_DUMMY_DF)
        assert len(result) == len(_DUMMY_DF)
    finally:
        adapter.shutdown()
