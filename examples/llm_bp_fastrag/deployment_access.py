# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
import csv
import json
from io import StringIO
from typing import Any
from uuid import uuid4

import pandas as pd
from langchain.schema import Document
from loguru import logger
from pydantic import BaseModel
from vector_database.inference.entities import QueryEmbeddings

DEFAULT_EXTRA_TYPES = {"source": str, "page": int, "start_index": int, "similarity_score": float}
METADATA_COLUMN = "metadata"
EXTRA_MODEL_OUTPUT_KEY = "extraModelOutput"
QUERY_EMBEDDING_COLUMN = "_LLM_PROMPT_VECTOR"


class DataRobotDeploymentCredentials(BaseModel):
    """Credentials necessary to access a DataRobot deployment."""

    datarobot_key: str | None
    authorization_header: str


def construct_headers(
    model_type: str,
    credentials: DataRobotDeploymentCredentials,
    accept_csv_response: bool = False,
    is_chat_api_request: bool = False,
) -> dict:
    """Construct the headers needed for accessing the DataRobot deployment."""
    headers = {"Authorization": credentials.authorization_header}
    if credentials.datarobot_key:
        headers["DataRobot-Key"] = credentials.datarobot_key
    if model_type != "UNSTRUCTURED":
        if accept_csv_response:
            # Structured may accept csv response
            headers["Accept"] = "text/csv"
        # Unstructured doesn't need this but structured fails without it
        headers["Content-Type"] = "text/plain; charset=UTF-8"
    if is_chat_api_request:
        # Chat API requests must send JSON and can receive JSON or event streams
        headers["Content-Type"] = "application/json"
        headers.pop("Accept", None)
    return headers


def construct_payload(
    input_type: str,
    column_names: list[str],
    column_values: list[str],
    association_id_column: str | None,
) -> bytes:
    """Construct the payload expected by the deployment."""
    if not (
        isinstance(column_names, list)
        and isinstance(column_values, list)
        and (len(column_names) == len(column_values))
    ):
        raise ValueError("Column names and values must have the same length.")
    if input_type == "CSV":
        # CSV writer will take care of escaping the commas and other stuff
        filelike = StringIO()
        writer = csv.writer(filelike)
        if association_id_column:
            writer.writerow([*column_names, association_id_column])
            writer.writerow([*column_values, uuid4()])
        else:
            writer.writerow([*column_names])
            writer.writerow([*column_values])
        payload = filelike.getvalue().encode("utf8")
    else:
        data = {col_name: col_value for col_name, col_value in zip(column_names, column_values)}
        payload = json.dumps(data).encode("utf8")
    return payload


def construct_payload_from_dict(
    data: dict[str, Any], input_type: str, association_id_column: str | None
) -> bytes:
    """Construct the payload expected by the deployment from a data dict."""
    if input_type == "CSV":
        if association_id_column:
            data[association_id_column] = uuid4()
        filelike = StringIO()
        writer = csv.writer(filelike)
        writer.writerow(data.keys())
        writer.writerow(data.values())
        payload = filelike.getvalue().encode("utf8")
    else:
        payload = json.dumps(data).encode("utf8")
    return payload


def parse_response_data(
    response_data: dict | pd.DataFrame, model_type: str, target_column_name: str
) -> Any:
    """Parse the response from the deployment."""
    if model_type != "UNSTRUCTURED":
        if isinstance(response_data, pd.DataFrame):
            return response_data[target_column_name][0]
        else:
            return response_data["data"][0]["prediction"]
    else:
        return response_data[target_column_name]


def is_valid_response_with_extra_output(response_data: dict) -> bool:
    """Check if the response is valid and has extra output."""
    if (
        "data" in response_data
        and EXTRA_MODEL_OUTPUT_KEY in response_data["data"][0]
        and isinstance(response_data["data"][0][EXTRA_MODEL_OUTPUT_KEY], dict)
        and response_data["data"][0].get(EXTRA_MODEL_OUTPUT_KEY)
    ):
        return True
    return False


def parse_documents_from_response_data(
    response_data: dict | Any, model_type: str, target_column_name: str
) -> list[Document]:
    """Parse the response and extract documents from the deployment."""
    metadata_column_contents = None
    if model_type == "UNSTRUCTURED":
        document_list = response_data.pop(target_column_name)
        if METADATA_COLUMN in response_data:
            metadata_column_contents = response_data[METADATA_COLUMN]
    else:
        document_list = parse_response_data(response_data, model_type, target_column_name)
        if (
            is_valid_response_with_extra_output(response_data)
            and METADATA_COLUMN in response_data["data"][0][EXTRA_MODEL_OUTPUT_KEY]
        ):
            metadata_column_contents = response_data["data"][0][EXTRA_MODEL_OUTPUT_KEY][
                METADATA_COLUMN
            ]
    if not isinstance(document_list, list):
        document_list = [document_list]
    if metadata_column_contents:
        metadata_list = _parse_metadata(document_list, metadata_column_contents)
    else:
        metadata_list = [{}] * len(document_list)
    return [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(document_list, metadata_list)
    ]


def _parse_metadata(
    document_list: list[str], metadata_column: dict[str, list[str]] | list[dict[str, str]]
) -> list[dict[str, Any]]:
    """
    Parse document metadata a from prediction response.

    We support 2 formats in the metadata column - a list of dicts and a dict of lists:
    {'metadata_field1': ['value 0', 'value 1'], 'metadata_field2': ['value 2', 'value 3']}
    or
    [{'metadata_field1': 'value 0', 'metadata_field2': 'value 2'},
     {'metadata_field1': 'value 1', 'metadata_field2': 'value 3'}]

    This function is capable of parsing both formats.
    """
    parsed_metadata_list = []
    for i in range(len(document_list)):
        parsed_metadata = {}
        metadata_to_parse = (
            metadata_column[i] if isinstance(metadata_column, list) else metadata_column
        )
        for key, values in metadata_to_parse.items():
            try:
                # all non-expected metadata types are cast to str
                cast_type = DEFAULT_EXTRA_TYPES.get(key, str)
                # if metadata is a single value all docs get the same value
                value = values[i] if isinstance(values, list) else values
                if value is None:
                    continue
                parsed_metadata[key] = cast_type(value)
            except Exception:
                parsed_metadata[key] = None
        parsed_metadata_list.append(parsed_metadata)
    return parsed_metadata_list


def parse_query_embeddings_from_response_data(response_data: dict | Any) -> QueryEmbeddings | None:
    """Parse a response from an external Vector Database and extract query embeddings."""
    if not is_valid_response_with_extra_output(response_data):
        return None
    query_embedding = response_data["data"][0][EXTRA_MODEL_OUTPUT_KEY].get(QUERY_EMBEDDING_COLUMN)
    if isinstance(query_embedding, str):
        try:
            query_embedding = json.loads(query_embedding)
        except json.JSONDecodeError:
            logger.warning(
                "Query embeddings found for external model, but were not valid json. Embeddings will not be reported."
            )
            query_embedding = None
    return query_embedding
