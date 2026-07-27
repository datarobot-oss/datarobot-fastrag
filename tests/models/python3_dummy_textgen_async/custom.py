from typing import Any

import pandas as pd


async def load_model(code_dir: str) -> Any:
    return "dummy"


async def score(data, model, **kwargs):
    inputs = list(data["input"])
    flipped = ["".join(reversed(inp)) for inp in inputs]
    result = pd.DataFrame({"output": flipped})
    return result
