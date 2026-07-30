# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
import numpy as np
from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse
from vector_database.enum import EmbeddingStage

class OpenAIEmbeddingClient:

    def __init__(self, base_url: str, api_key: str, model: str, extra_body_params: list[dict[str, str]] | None=None, reuse_client: bool=False):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.extra_body_params = extra_body_params
        self.client = None
        if reuse_client:
            self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    async def encode(self, sentences: list[str], embedding_stage: EmbeddingStage) -> np.ndarray:
        if self.client:
            embeddings = await self.make_embeddings_call(self.client, embedding_stage, sentences)
        else:
            async with AsyncOpenAI(base_url=self.base_url, api_key=self.api_key) as client:
                embeddings = await self.make_embeddings_call(client, embedding_stage, sentences)
        response = []
        for data in embeddings.data:
            response.append(data.embedding)
        return np.array(response)

    def _prepare_extra_body_params(self, embedding_stage: EmbeddingStage) -> dict | None:
        if self.extra_body_params:
            return {param['key']: param['value'] for param in self.extra_body_params if param['stage'] in {embedding_stage.value, 'both'}}
        else:
            return None

    async def make_embeddings_call(self, client: AsyncOpenAI, embedding_stage: EmbeddingStage, sentences: list[str]) -> CreateEmbeddingResponse:
        extra_body = self._prepare_extra_body_params(embedding_stage)
        return await client.embeddings.create(input=sentences, model=self.model, encoding_format='float', extra_body=extra_body)

    async def close(self) -> None:
        if self.client:
            await self.client.close()