import asyncio
import os

import pandas as pd
import pytest

from fastrag.loader import HookLoader


@pytest.mark.asyncio
async def test_sync_model_thread_local():
    code_dir = os.path.join(os.path.dirname(__file__), "dummy_sync_model_thread_local")
    loader = HookLoader(code_dir=code_dir, max_workers=2)

    adapter = await loader.load()
    assert adapter.model is None

    async def run_score():
        result = await adapter.score(pd.DataFrame({"a": [1]}))
        return result

    tasks = [run_score() for _ in range(20)]
    results = await asyncio.gather(*tasks)

    thread_ids = set()
    for res in results:
        model_tid = res["model_thread_id"]
        current_tid = res["current_thread_id"]

        assert model_tid == current_tid, (
            f"Model thread ID {model_tid} != Current thread ID {current_tid}"
        )
        thread_ids.add(model_tid)

    print(f"Used thread IDs: {thread_ids}")
    assert len(thread_ids) > 0
