from typing import Any
from typing import Dict

import pandas as pd


async def load_model(code_dir: str) -> Any:
    return "dummy"


async def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    predictions = pd.DataFrame([0.0] * data.shape[0], columns=["Predictions"])
    for i in range(data.shape[0] // 10):
        predictions.loc[(i + 1) * 10 - 1, "Predictions"] = 0.75
    return predictions
