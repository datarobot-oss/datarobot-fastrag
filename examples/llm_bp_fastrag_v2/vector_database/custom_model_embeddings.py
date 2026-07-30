# Copyright 2023 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
import asyncio
import csv
import json
import os
from io import StringIO
import backoff
import numpy as np
from aiohttp import ClientResponseError
from aiohttp.client import DEFAULT_TIMEOUT
from aiohttp.client import ClientSession
from pydantic import BaseModel
from pydantic import TypeAdapter
from deployment_access import DataRobotDeploymentCredentials
from deployment_access import construct_headers
from exceptions import RetryableCustomModelError
from i18n import gettext
from vector_database.enum import EmbeddingStage
from vector_database.exception import CustomModelEmbeddingError
from vector_database.exception import IncorrectResponseFromCustomModelError

def get_factor() -> int:
    # This variable allows control of the initial factor for exponential backoff strategy.
    return int(os.getenv('EXPONENTIAL_BACKOFF_FACTOR', '30'))

def get_max_tries() -> int:
    return 5

@backoff.on_exception(backoff.expo, exception=(asyncio.TimeoutError, RetryableCustomModelError), max_tries=get_max_tries, factor=get_factor, jitter=None)
async def _make_encode_call(session: ClientSession, prediction_api_url: str, payload: bytes) -> str:
    try:
        response = await session.post(prediction_api_url, data=payload, timeout=DEFAULT_TIMEOUT)
        return await response.text()
    except ClientResponseError as e:
        if e.status in [502, 503, 504]:
            raise RetryableCustomModelError() from e
        raise e

class BaseCustomModelEmbeddingErrorHandler:

    async def handle(self) -> None:
        pass

class CustomModelEmbeddingClient:
    """Text embedding implemented as deployed custom model."""

    def __init__(self, credentials: BaseModel, prediction_api_url: str, model_type: str, input_type: str, prompt_column_name: str, target_column_name: str, error_handler: BaseCustomModelEmbeddingErrorHandler | None=None, reuse_session: bool=False):
        self.credentials = DataRobotDeploymentCredentials.model_validate(credentials, from_attributes=True)
        self.prediction_api_url = prediction_api_url
        self.model_type = model_type
        self.input_type = input_type
        self.prompt_column_name = prompt_column_name
        self.target_column_name = target_column_name
        self.error_handler = error_handler
        self.type_adapter_for_response = TypeAdapter(list[list[float]])
        self.type_adapter_for_request = TypeAdapter(list[str])
        self.headers = construct_headers(self.model_type, self.credentials)
        self.tokenizer = None
        self.session = None
        if reuse_session:
            self.session = ClientSession(headers=self.headers, raise_for_status=True, timeout=DEFAULT_TIMEOUT)
        self.max_seq_length = None

    def _construct_payload(self, texts: list[str], embedding_stage: EmbeddingStage) -> bytes:
        self.type_adapter_for_request.validate_python(texts)
        'Construct the payload expected by the deployment.'
        if self.input_type == 'CSV':
            # CSV writer will take care of escaping the commas and other stuff
            filelike = StringIO()
            writer = csv.writer(filelike)
            writer.writerow([self.prompt_column_name])
            for text in texts:
                writer.writerow([text])
            payload = filelike.getvalue().encode('utf8')
        else:
            payload = json.dumps({self.prompt_column_name: texts, 'embedding_stage': embedding_stage}).encode('utf8')
        return payload

    def _parse_response_data(self, response_data: str) -> np.ndarray:
        """Parse the response from the deployment."""
        try:
            response_dict = json.loads(response_data)
            embeddings = response_dict[self.target_column_name]
            self.type_adapter_for_response.validate_python(embeddings)
            return np.array(embeddings)
        except Exception:
            msg = "The GenAI service received an incorrect response which couldn't be interpreted."
            # We don't want to log the original exception as it could contain PII
            raise IncorrectResponseFromCustomModelError(msg) from None

    async def encode(self, sentences: list[str], embedding_stage: EmbeddingStage) -> np.ndarray:
        """Submit texts to a custom model embedding.

        Parameters
        ----------
        sentences
            The list of texts to embed
        embedding_stage
            Type of embedding query: indexing or prompt

        Returns
        -------
        np.ndarray
            Array of floats constituting the embeddings for the texts
        """
        try:
            payload = self._construct_payload(sentences, embedding_stage)
            if self.session:
                _data = await _make_encode_call(self.session, self.prediction_api_url, payload)
            else:
                async with ClientSession(headers=self.headers, raise_for_status=True, timeout=DEFAULT_TIMEOUT) as session:
                    _data = await _make_encode_call(session, self.prediction_api_url, payload)
            return self._parse_response_data(_data)
        except Exception:
            if self.error_handler is not None:
                await self.error_handler.handle()
            # This code runs inside custom models, where there is no diagnostic access to
            # pinpoint the exact reason of the error.
            msg = gettext('Custom model embedding request returned an error. Try again or check the custom model response in playground for additional details.')
            # We don't want to log the original exception as it could contain PII
            raise CustomModelEmbeddingError(msg) from None

    async def close(self) -> None:
        if self.session:
            await self.session.close()