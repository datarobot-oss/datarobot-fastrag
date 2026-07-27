import asyncio

import pandas as pd


def load_model(code_dir):
    return {"name": "local-test-model"}


def score(data, model, **kwargs):
    # Exercises the event loop fix: simulates what datarobot_dome's AsyncGuardExecutor
    # does — calls get_event_loop() inside a ThreadPoolExecutor worker thread.
    # Without the fix this raises RuntimeError: There is no current event loop in thread.
    loop = asyncio.get_event_loop()
    assert loop is not None, "No event loop in thread — fix not applied"
    return pd.DataFrame({"Prediction": [0.42] * len(data)})
