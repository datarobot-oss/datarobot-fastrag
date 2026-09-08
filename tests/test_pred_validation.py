import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from fastrag.model_metadata import ModelMetadata
from fastrag.model_metadata import TargetType
from fastrag.schemas import BinaryPredictionResponse
from fastrag.schemas import MulticlassPredictionResponse
from fastrag.schemas import RegressionPredictionResponse
from fastrag.schemas import TextGenerationPredictionResponse
from fastrag.schemas import VectorDatabasePredictionResponse
from fastrag.validation import format_prediction_response


@pytest.fixture
def regression_metadata():
    return ModelMetadata(target_type=TargetType.REGRESSION)


@pytest.fixture
def binary_metadata():
    return ModelMetadata(
        target_type=TargetType.BINARY,
        inference_model={
            "positiveClassLabel": "Yes",
            "negativeClassLabel": "No",
            "predictionThreshold": 0.5,
        },
    )


@pytest.fixture
def multiclass_metadata():
    return ModelMetadata(
        target_type=TargetType.MULTICLASS,
        inference_model={"classLabels": ["apple", "banana", "orange"]},
    )


@pytest.fixture
def text_generation_metadata():
    return ModelMetadata(target_type=TargetType.TEXT_GENERATION)


@pytest.fixture
def vector_database_metadata():
    return ModelMetadata(
        target_type=TargetType.VECTOR_DATABASE,
        inference_model={"targetName": "relevant"},
    )


def test_regression_example(regression_metadata):
    df = pd.DataFrame({"prediction": [1.5, 2.3, 4.8, 10.1]})
    resp = format_prediction_response(df, regression_metadata)
    assert isinstance(resp, RegressionPredictionResponse)
    assert [p.prediction for p in resp.predictions] == [1.5, 2.3, 4.8, 10.1]


def test_binary_classification_example(binary_metadata):
    df = pd.DataFrame({"Yes": [0.9, 0.2, 0.55], "No": [0.1, 0.8, 0.45]})
    resp = format_prediction_response(df, binary_metadata)
    assert isinstance(resp, BinaryPredictionResponse)
    assert [p.prediction for p in resp.predictions] == ["Yes", "No", "Yes"]


def test_multiclass_classification_example(multiclass_metadata):
    df = pd.DataFrame(
        [
            {"apple": 0.1, "banana": 0.8, "orange": 0.1},
            {"apple": 0.7, "banana": 0.1, "orange": 0.2},
        ]
    )
    resp = format_prediction_response(df, multiclass_metadata)
    assert isinstance(resp, MulticlassPredictionResponse)
    assert [p.prediction for p in resp.predictions] == ["banana", "apple"]


def test_with_extra_output_example(regression_metadata):
    df = pd.DataFrame(
        {
            "prediction": [0.5, 0.6],
            "explanation_1": ["feature_a", "feature_b"],
            "explanation_2": [0.9, 0.8],
        }
    )

    resp = format_prediction_response(df, regression_metadata)

    assert [p.prediction for p in resp.predictions] == [0.5, 0.6]
    assert resp.extraModelOutput is not None
    assert resp.extraModelOutput.columns == ["explanation_1", "explanation_2"]
    assert resp.extraModelOutput.data == [["feature_a", 0.9], ["feature_b", 0.8]]


def test_dataframe_to_regression_example(regression_metadata):
    df = pd.DataFrame({"other_col": [1.5, 2.3, 4.8, 10.1]})
    resp = format_prediction_response(df, regression_metadata)
    assert [p.prediction for p in resp.predictions] == [1.5, 2.3, 4.8, 10.1]


def test_dataframe_to_classification_example(binary_metadata):
    df = pd.DataFrame({"Yes": [0.9, 0.2, 0.55], "No": [0.1, 0.8, 0.45]})
    resp = format_prediction_response(df, binary_metadata)
    assert [p.prediction for p in resp.predictions] == ["Yes", "No", "Yes"]


def test_target_type_selects_regression_response(regression_metadata):
    df = pd.DataFrame({"prediction": [1.5, 2.3]})
    resp = format_prediction_response(df, regression_metadata)
    assert isinstance(resp, RegressionPredictionResponse)


def test_target_type_selects_binary_response_and_validates_values(binary_metadata):
    df = pd.DataFrame({"Yes": [0.9], "No": [0.1]})
    resp = format_prediction_response(df, binary_metadata)
    assert isinstance(resp, BinaryPredictionResponse)

    df_bad = pd.DataFrame({"Yes": ["bad"], "No": [0.1]})
    with pytest.raises((ValidationError, ValueError)):
        format_prediction_response(df_bad, binary_metadata)


def test_binary_metadata_label_mismatch_raises(binary_metadata):
    df = pd.DataFrame({"yes": [0.9], "no": [0.1]})
    with pytest.raises(ValueError, match="missing label"):
        format_prediction_response(df, binary_metadata)


def test_multiclass_metadata_label_mismatch_raises(multiclass_metadata):
    df = pd.DataFrame({"apple": [0.7], "banana": [0.2], "pear": [0.1]})
    with pytest.raises(ValueError, match="do not match metadata labels"):
        format_prediction_response(df, multiclass_metadata)


def test_binary_with_extra_columns_auto_split(binary_metadata):
    df = pd.DataFrame(
        {
            "Yes": [0.9, 0.3],
            "No": [0.1, 0.7],
            "explanation": ["reason1", "reason2"],
            "confidence": [0.95, 0.85],
        }
    )
    resp = format_prediction_response(df, binary_metadata)
    assert isinstance(resp, BinaryPredictionResponse)
    assert [p.prediction for p in resp.predictions] == ["Yes", "No"]
    assert resp.extraModelOutput is not None
    assert resp.extraModelOutput.columns == ["explanation", "confidence"]


def test_multiclass_with_extra_columns_auto_split(multiclass_metadata):
    df = pd.DataFrame(
        {
            "apple": [0.1, 0.7],
            "banana": [0.8, 0.2],
            "orange": [0.1, 0.1],
            "extra_info": ["info1", "info2"],
        }
    )
    resp = format_prediction_response(df, multiclass_metadata)
    assert isinstance(resp, MulticlassPredictionResponse)
    assert [p.prediction for p in resp.predictions] == ["banana", "apple"]
    assert resp.extraModelOutput is not None
    assert resp.extraModelOutput.columns == ["extra_info"]


def test_regression_with_extra_columns_auto_split(regression_metadata):
    df = pd.DataFrame(
        {
            "prediction": [1.5, 2.3],
            "uncertainty": [0.1, 0.2],
            "feature_importance": ["high", "low"],
        }
    )
    resp = format_prediction_response(df, regression_metadata)
    assert isinstance(resp, RegressionPredictionResponse)
    assert [p.prediction for p in resp.predictions] == [1.5, 2.3]
    assert resp.extraModelOutput is not None
    assert set(resp.extraModelOutput.columns) == {"uncertainty", "feature_importance"}


def test_text_generation_selects_typed_response(text_generation_metadata):
    df = pd.DataFrame({"prediction": ["hello", "world"]})
    resp = format_prediction_response(df, text_generation_metadata)
    assert isinstance(resp, TextGenerationPredictionResponse)
    assert resp.predictions == ["hello", "world"]


def test_text_generation_with_extra_columns_auto_split(text_generation_metadata):
    df = pd.DataFrame(
        {
            "prediction": ["hello", "world"],
            "token_count": [1, 1],
            "metadata": ["a", "b"],
        }
    )

    resp = format_prediction_response(df, text_generation_metadata)

    assert isinstance(resp, TextGenerationPredictionResponse)
    assert resp.predictions == ["hello", "world"]
    assert resp.extraModelOutput is not None
    assert set(resp.extraModelOutput.columns) == {"token_count", "metadata"}


def test_vector_database_selects_typed_response(vector_database_metadata):
    df = pd.DataFrame({"relevant": [["chunk a", "chunk b"], ["chunk c"]]})
    resp = format_prediction_response(df, vector_database_metadata)
    assert isinstance(resp, VectorDatabasePredictionResponse)
    assert resp.predictions == [["chunk a", "chunk b"], ["chunk c"]]
    assert resp.extraModelOutput is None


def test_vector_database_splits_citation_columns(vector_database_metadata):
    df = pd.DataFrame(
        {
            "relevant": [["chunk a", "chunk b"]],
            "CITATION_SOURCE_0": ["docs/autopilot.pdf"],
            "CITATION_PAGE_0": [3],
        }
    )

    resp = format_prediction_response(df, vector_database_metadata)

    assert resp.predictions == [["chunk a", "chunk b"]]
    assert resp.extraModelOutput is not None
    assert resp.extraModelOutput.columns == ["CITATION_SOURCE_0", "CITATION_PAGE_0"]
    assert resp.extraModelOutput.data == [["docs/autopilot.pdf", 3]]


def test_vector_database_target_column_from_quoted_env(monkeypatch):
    """DataRobot exports TARGET_NAME quoted, so the column lookup has to strip it.

    Without the strip the target name never matches and the split silently falls back
    to column order, picking the citation column as the predictions.
    """
    monkeypatch.setenv("TARGET_NAME", '"relevant"')
    metadata = ModelMetadata(target_type=TargetType.VECTOR_DATABASE)
    metadata.merge_env_overrides()

    df = pd.DataFrame(
        {
            "CITATION_SOURCE_0": ["docs/autopilot.pdf"],
            "relevant": [["chunk a"]],
        }
    )

    resp = format_prediction_response(df, metadata)

    assert resp.predictions == [["chunk a"]]
    assert resp.extraModelOutput is not None
    assert resp.extraModelOutput.columns == ["CITATION_SOURCE_0"]


def test_vector_database_target_column_is_not_first(vector_database_metadata):
    """The target name, not column order, decides which column holds the documents."""
    df = pd.DataFrame(
        {
            "CITATION_SOURCE_0": ["docs/autopilot.pdf"],
            "relevant": [["chunk a"]],
        }
    )

    resp = format_prediction_response(df, vector_database_metadata)

    assert resp.predictions == [["chunk a"]]
    assert resp.extraModelOutput is not None
    assert resp.extraModelOutput.columns == ["CITATION_SOURCE_0"]


def test_vector_database_falls_back_to_first_column():
    """Without a matching target name the first column holds the documents, as in DRUM."""
    metadata = ModelMetadata(target_type=TargetType.VECTOR_DATABASE)
    df = pd.DataFrame({"documents": [["chunk a"]], "score": [0.9]})

    resp = format_prediction_response(df, metadata)

    assert resp.predictions == [["chunk a"]]
    assert resp.extraModelOutput is not None
    assert resp.extraModelOutput.columns == ["score"]


def test_vector_database_accepts_non_list_sequences(vector_database_metadata):
    df = pd.DataFrame({"relevant": [("chunk a", "chunk b"), np.array(["chunk c"])]})
    resp = format_prediction_response(df, vector_database_metadata)
    assert resp.predictions == [["chunk a", "chunk b"], ["chunk c"]]


def test_vector_database_scalar_prediction_raises(vector_database_metadata):
    df = pd.DataFrame({"relevant": ["chunk a"]})
    with pytest.raises(ValueError, match="must be a list of retrieved documents, got str"):
        format_prediction_response(df, vector_database_metadata)
