from typing import Any

import pandas as pd


def load_model(code_dir: str) -> Any:
    return "dummy"


def score(data, model, **kwargs):
    """
    Vector database models must return one list of retrieved documents per input row
    in the target column. Any other column is treated as extra model output, which is
    how real vector databases return their citation metadata.
    """
    prompts = list(data["promptText"])
    return pd.DataFrame(
        {
            "relevant": [[f"chunk about {p}", "supporting chunk"] for p in prompts],
            "CITATION_SOURCE_0": ["docs/autopilot.pdf" for _ in prompts],
            "CITATION_PAGE_0": [3 for _ in prompts],
        }
    )
