from typing import Any
from typing import Dict

import pandas as pd


async def load_model(code_dir: str) -> Any:
    return "dummy"


async def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    positive_label = kwargs["positive_class_label"]
    negative_label = kwargs["negative_class_label"]
    preds = pd.DataFrame([[0.75, 0.25]] * data.shape[0], columns=[positive_label, negative_label])
    return preds
