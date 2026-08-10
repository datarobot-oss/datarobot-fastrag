from collections.abc import Callable
from io import BytesIO
from typing import Any
from typing import cast

import pandas as pd
from fastapi import Request
from fastapi import UploadFile
from pandas.errors import EmptyDataError
from pandas.errors import ParserError

from .model_metadata import InferenceModelConfig
from .model_metadata import ModelMetadata
from .model_metadata import TargetType
from .schemas import BinaryPredictionItem
from .schemas import BinaryPredictionResponse
from .schemas import BinaryPredictionValue
from .schemas import ExtraModelOutput
from .schemas import GenericPredictionResponse
from .schemas import MulticlassPredictionItem
from .schemas import MulticlassPredictionResponse
from .schemas import MulticlassPredictionValue
from .schemas import PredictionResponse
from .schemas import RegressionPredictionItem
from .schemas import RegressionPredictionResponse
from .schemas import TextGenerationPredictionResponse


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str, log_message: str | None = None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.log_message = log_message


class BadRequestError(ApiError):
    def __init__(self, detail: str, log_message: str | None = None):
        super().__init__(status_code=400, detail=detail, log_message=log_message)


class UnprocessableEntityError(ApiError):
    def __init__(self, detail: str, log_message: str | None = None):
        super().__init__(status_code=422, detail=detail, log_message=log_message)


class NotFoundError(ApiError):
    def __init__(self, detail: str, log_message: str | None = None):
        super().__init__(status_code=404, detail=detail, log_message=log_message)


class NotImplementedApiError(ApiError):
    def __init__(self, detail: str, log_message: str | None = None):
        super().__init__(status_code=501, detail=detail, log_message=log_message)


class InternalServerError(ApiError):
    def __init__(self, detail: str, log_message: str | None = None):
        super().__init__(status_code=500, detail=detail, log_message=log_message)


def split_predictions_and_extra_output(
    result_df: pd.DataFrame,
    prediction_columns: list[str] | None = None,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Split a DataFrame into predictions and extra model output.

    For classification: pass `prediction_columns` (class labels).
    For regression/other: pass `target_column` (defaults to first column).

    Returns (predictions_df, extra_model_output or None).
    If prediction_columns are provided but not all exist, returns original df unchanged.
    """
    if prediction_columns:
        # Only split if all prediction columns exist
        if not all(col in result_df.columns for col in prediction_columns):
            return result_df, None
        extra_cols = [c for c in result_df.columns if c not in prediction_columns]
        if extra_cols:
            return result_df[prediction_columns], result_df[extra_cols]
        return result_df, None

    if result_df.shape[1] == 1:
        return result_df, None

    target = target_column or str(result_df.columns[0])
    if target not in result_df.columns:
        return result_df, None
    extra_model_output = result_df.drop(columns=[target])
    predictions_df = result_df[[target]]
    return predictions_df, extra_model_output


async def read_structured_payload(request: Request, X: UploadFile | None) -> tuple[bytes, str]:
    if X is not None:
        content = await X.read()
        if not content:
            raise BadRequestError(
                detail="Invalid CSV file: Empty file provided under 'X'.",
            )
        return content, X.filename or ""

    content = await request.body()
    if content:
        return content, "(raw body)"

    raise UnprocessableEntityError(
        detail="Samples should be provided as multipart form-data under 'X' or raw body.",
    )


def read_csv_or_raise(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(BytesIO(content))
    except (ParserError, EmptyDataError, ValueError) as exc:
        raise BadRequestError(detail="Invalid CSV file.") from exc
    except Exception as exc:
        raise InternalServerError(
            detail="Error reading request payload.",
            log_message="Error reading prediction payload.",
        ) from exc


def target_type_is_unstructured(target_type: str | None) -> bool:
    if target_type is None:
        return False
    return target_type.lower() in {"unstructured", "text_generation", "textgeneration"}


def parse_content_type(content_type: str | None) -> tuple[str | None, str | None]:
    if not content_type:
        return None, None
    parts = [p.strip() for p in content_type.split(";")]
    mimetype = parts[0] if parts else None
    charset = None
    for part in parts[1:]:
        if part.lower().startswith("charset="):
            charset = part.split("=", 1)[1].strip()
            break
    return mimetype, charset


def _is_text_mimetype(mimetype: str | None) -> bool:
    if mimetype is None:
        return True
    return mimetype.startswith("text/") or mimetype == "application/json"


def resolve_incoming_unstructured_data(
    in_data: bytes, mimetype: str | None, charset: str | None
) -> tuple[Any, str, str | None]:
    if not isinstance(in_data, (bytes, bytearray)):
        raise InternalServerError(detail=f"bytes data is expected, received {type(in_data)}")

    ret_mimetype = mimetype if mimetype else "text/plain"
    ret_charset: str | None
    ret_data: Any

    if _is_text_mimetype(ret_mimetype):
        ret_charset = charset if charset is not None else "utf8"
        try:
            ret_data = in_data.decode(ret_charset)
        except (LookupError, UnicodeDecodeError) as exc:
            raise BadRequestError(
                detail="Invalid text payload encoding.",
            ) from exc
    else:
        ret_charset = charset
        ret_data = bytes(in_data)

    return ret_data, ret_mimetype, ret_charset


def resolve_outgoing_unstructured_data(
    result: Any,
) -> tuple[bytes | None, str | None, str | None]:
    if isinstance(result, tuple):
        if len(result) != 2:
            raise InternalServerError(
                detail="In unstructured mode tuple return value must have length 2.",
            )
        ret_data, ret_kwargs = result
    else:
        ret_data, ret_kwargs = result, None

    if ret_kwargs is None:
        ret_kwargs = {}
    if not isinstance(ret_kwargs, dict):
        raise InternalServerError(
            detail="In unstructured mode tuple return value must be (str/bytes, dict).",
        )

    if isinstance(ret_data, (str, type(None))):
        ret_mimetype = ret_kwargs.get("mimetype", "text/plain")
        ret_charset = ret_kwargs.get("charset", "utf8")
        if ret_data is not None:
            ret_data = ret_data.encode(ret_charset)
    elif isinstance(ret_data, (bytes, bytearray)):
        ret_mimetype = ret_kwargs.get("mimetype", "application/octet-stream")
        ret_charset = ret_kwargs.get("charset", None)
        ret_data = bytes(ret_data)
    else:
        raise InternalServerError(
            detail="In unstructured mode return value must be str/bytes or (str/bytes, dict).",
        )

    return ret_data, ret_mimetype, ret_charset


def format_prediction_response(
    predictions: pd.DataFrame,
    model_metadata: ModelMetadata,
) -> PredictionResponse:
    inference_model = model_metadata.inference_model
    target_type = model_metadata.target_type

    # Split extra columns from predictions based on target type
    if target_type == TargetType.BINARY:
        prediction_cols = [
            label
            for label in (
                inference_model.positive_class_label,
                inference_model.negative_class_label,
            )
            if label
        ]
        predictions, extra_model_output = split_predictions_and_extra_output(
            predictions, prediction_columns=prediction_cols if prediction_cols else None
        )
    elif target_type == TargetType.MULTICLASS:
        predictions, extra_model_output = split_predictions_and_extra_output(
            predictions, prediction_columns=inference_model.class_labels
        )
    elif target_type in {TargetType.REGRESSION, TargetType.ANOMALY, TargetType.TEXT_GENERATION}:
        target_col = "prediction" if "prediction" in predictions.columns else None
        predictions, extra_model_output = split_predictions_and_extra_output(
            predictions, target_column=target_col
        )
    else:
        extra_model_output = None

    extra = _format_extra_output(extra_model_output) if extra_model_output is not None else None

    match target_type:
        case TargetType.REGRESSION | TargetType.ANOMALY:
            return _build_regression_response(predictions, extra)
        case TargetType.TEXT_GENERATION:
            return _build_text_generation_response(predictions, extra)
        case TargetType.BINARY:
            return _build_binary_response(predictions, inference_model, extra)
        case TargetType.MULTICLASS:
            return _build_multiclass_response(predictions, inference_model, extra)
        case _:
            return _build_generic_response(predictions, extra)


def _format_extra_output(extra_model_output: pd.DataFrame) -> ExtraModelOutput:
    data = extra_model_output.to_dict(orient="split")
    return ExtraModelOutput(**data)


def _extract_scalar_predictions(df: pd.DataFrame, cast: Callable[[Any], Any]) -> list[Any]:
    if "prediction" in df.columns:
        return [cast(row["prediction"]) for _, row in df.iterrows()]
    if df.shape[1] == 1:
        return [cast(val) for val in df.iloc[:, 0]]
    return [cast(row.iloc[0]) for _, row in df.iterrows()]


def _build_regression_response(
    df: pd.DataFrame, extra: ExtraModelOutput | None
) -> RegressionPredictionResponse:
    items = [RegressionPredictionItem(prediction=v) for v in _extract_scalar_predictions(df, float)]
    return RegressionPredictionResponse(predictions=items, extraModelOutput=extra)


def _build_text_generation_response(
    df: pd.DataFrame, extra: ExtraModelOutput | None
) -> TextGenerationPredictionResponse:
    items = _extract_scalar_predictions(df, str)
    return TextGenerationPredictionResponse(predictions=items, extraModelOutput=extra)


def _build_classification_values(
    row: pd.Series, columns: list[str], value_cls: type[Any]
) -> tuple[list[Any], str]:
    values = [value_cls(label=str(col), value=float(row[col])) for col in columns]
    predicted_label = max(values, key=lambda x: x.value).label
    return values, predicted_label


def _build_binary_response(
    df: pd.DataFrame, inference_model: InferenceModelConfig, extra: ExtraModelOutput | None
) -> BinaryPredictionResponse:
    pos_label = inference_model.positive_class_label
    neg_label = inference_model.negative_class_label
    expected_labels = {label for label in (pos_label, neg_label) if label}

    items: list[BinaryPredictionItem] = []
    columns = list(df.columns)

    for idx, (_, row) in enumerate(df.iterrows()):
        values, predicted_label = _build_classification_values(row, columns, BinaryPredictionValue)
        labels = {v.label for v in values}

        if pos_label and pos_label not in labels:
            raise ValueError(f"Binary prediction at index {idx} is missing label '{pos_label}'.")
        if neg_label and neg_label not in labels:
            raise ValueError(f"Binary prediction at index {idx} is missing label '{neg_label}'.")
        if len(expected_labels) == 2 and labels != expected_labels:
            raise ValueError(
                f"Binary prediction at index {idx} labels {sorted(labels)} do not match "
                f"metadata labels {sorted(expected_labels)}."
            )
        if expected_labels and predicted_label not in expected_labels:
            raise ValueError(
                f"Binary prediction at index {idx} has prediction '{predicted_label}' "
                f"not present in metadata labels {sorted(expected_labels)}."
            )

        items.append(
            BinaryPredictionItem(
                prediction=predicted_label,
                predictionValues=cast(list[BinaryPredictionValue], values),
            )
        )

    return BinaryPredictionResponse(predictions=items, extraModelOutput=extra)


def _build_multiclass_response(
    df: pd.DataFrame, inference_model: InferenceModelConfig, extra: ExtraModelOutput | None
) -> MulticlassPredictionResponse:
    class_labels = inference_model.class_labels
    expected_labels = set(class_labels) if class_labels else None

    items: list[MulticlassPredictionItem] = []
    columns = list(df.columns)

    for idx, (_, row) in enumerate(df.iterrows()):
        values, predicted_label = _build_classification_values(
            row, columns, MulticlassPredictionValue
        )

        if expected_labels:
            labels = {v.label for v in values}
            if labels != expected_labels:
                raise ValueError(
                    f"Multiclass prediction at index {idx} labels {sorted(labels)} do not match "
                    f"metadata labels {sorted(expected_labels)}."
                )
            if predicted_label not in expected_labels:
                raise ValueError(
                    f"Multiclass prediction at index {idx} has prediction '{predicted_label}' "
                    "not present in metadata labels."
                )

        items.append(
            MulticlassPredictionItem(
                prediction=predicted_label,
                predictionValues=cast(list[MulticlassPredictionValue], values),
            )
        )

    return MulticlassPredictionResponse(predictions=items, extraModelOutput=extra)


def _build_generic_response(
    df: pd.DataFrame, extra: ExtraModelOutput | None
) -> GenericPredictionResponse:
    if df.shape[1] == 1 and list(df.columns) == ["predictions"]:
        predictions = df.to_dict(orient="records")
    elif df.shape[1] == 1:
        predictions = df.iloc[:, 0].tolist()
    else:
        predictions = df.to_dict(orient="records")
    return GenericPredictionResponse(predictions=predictions, extraModelOutput=extra)
