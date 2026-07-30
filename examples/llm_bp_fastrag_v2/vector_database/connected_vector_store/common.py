# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from enum import StrEnum
from typing import Literal
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from pydantic import BaseModel
from vector_database.dr_embeddings import Embeddings

class ConnectedVectorDatabaseType(StrEnum):
    PINECONE = 'pinecone'
    ELASTICSEARCH = 'elasticsearch'
    MILVUS = 'milvus'
    POSTGRES = 'postgres'

class ElasticsearchConnectionParams(BaseModel):
    """Connection parameters for Elasticsearch (model_execution_lib only)."""
    type: Literal[ConnectedVectorDatabaseType.ELASTICSEARCH] = ConnectedVectorDatabaseType.ELASTICSEARCH
    url: str | None = None
    cloud_id: str | None = None

class PineconeConnectionParams(BaseModel):
    """Connection parameters for Pinecone (model_execution_lib only)."""
    type: Literal[ConnectedVectorDatabaseType.PINECONE] = ConnectedVectorDatabaseType.PINECONE

class MilvusConnectionParams(BaseModel):
    """Connection parameters for Milvus (model_execution_lib only)."""
    type: Literal[ConnectedVectorDatabaseType.MILVUS] = ConnectedVectorDatabaseType.MILVUS
    uri: str

VectorDatabaseConnection = ElasticsearchConnectionParams | PineconeConnectionParams | MilvusConnectionParams

class ConnectedVectorStoreEmbeddings(LangChainEmbeddings):

    def __init__(self, embedder: Embeddings) -> None:
        self.embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return list(self.embedder.embed_query(text).astype(float))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = await self.embedder.aembed_documents(texts)
        return embeddings.tolist()

    async def aembed_query(self, text: str) -> list[float]:
        embedding = await self.embedder.aembed_query(text)
        return list(embedding.astype(float))