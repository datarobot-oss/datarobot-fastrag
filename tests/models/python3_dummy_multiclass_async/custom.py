from typing import Any
from typing import Dict

import pandas as pd


async def load_model(code_dir: str) -> Any:
    return "dummy"


async def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    class_labels = kwargs["class_labels"]
    M = len(class_labels)
    rows = [[0.75] + (M - 1) * [0.25 / (M - 1)]] * data.shape[0]
    predictions = pd.DataFrame(data=rows, columns=class_labels)
    return predictions
