from typing import Any

import pandas as pd


async def load_model(code_dir: str) -> Any:
    return "dummy"


async def score(data, model, **kwargs):
    prompts = list(data["promptText"])
    return pd.DataFrame(
        {
            "relevant": [[f"chunk about {p}", "supporting chunk"] for p in prompts],
            "CITATION_SOURCE_0": ["docs/autopilot.pdf" for _ in prompts],
            "CITATION_PAGE_0": [3 for _ in prompts],
        }
    )
