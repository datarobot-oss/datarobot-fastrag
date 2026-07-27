from typing import Any
from typing import Dict

import pandas as pd


async def load_model(code_dir: str) -> Any:
    return "dummy"


async def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    preds = pd.DataFrame([42 for _ in range(data.shape[0])], columns=["Predictions"])
    return preds
