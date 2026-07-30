# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from __future__ import annotations
import json
import uuid
from typing import Any
from typing import Iterable
from typing import Sequence
import backoff
import numpy as np
from backoff import random_jitter
from langchain_core.documents import Document
from langchain_core.utils import batch_iterate
from langchain_pinecone import PineconeVectorStore
from loguru import logger
from pinecone import ForbiddenException
from pinecone import NotFoundException
from pinecone import PineconeApiException
from pinecone import PineconeAsyncio as PineconeClient
from pinecone import ServiceException
from pinecone import UnauthorizedException
from pinecone.db_data import IndexAsyncio
from pinecone.db_data.types import VectorTupleWithMetadata
from pydantic import BaseModel
from vector_database.connected_vector_store.common import ConnectedVectorStoreEmbeddings
from vector_database.dr_embeddings import Embeddings
from vector_database.inference.entities import MetadataColumnNames
from vector_database.inference.retrieval import maximal_marginal_relevance
PINECONE_MAX_BATCH_SIZE = 1000
PINECONE_MAX_REQUEST_SIZE = 2 * 1000 * 1000  # 2 * 1024 * 1024 but let's leave a buffer
PINECONE_MAX_METADATA_SIZE = 40 * 1024

def construct_vector_tuples(text_key: str, ids: list[str] | None, metadatas: list[dict] | None, text_embeddings: Iterable[tuple[str, list[float]]]) -> list[tuple[str, list[float], dict]]:
    texts, embeddings = zip(*text_embeddings)
    ids = ids or [str(uuid.uuid4()) for _ in texts]
    metadatas = metadatas or [{} for _ in texts]
    vector_tuples = []
    for metadata, text, id, embedding in zip(metadatas, texts, ids, embeddings):
        metadata[text_key] = text
        try:
            metadata_size = len(json.dumps(metadata).encode('utf-8'))
            if metadata_size > PINECONE_MAX_METADATA_SIZE:
                continue
        except TypeError:
            pass
        vector_tuples.append((id, embedding, metadata))
    return vector_tuples

class PineconeCredentials(BaseModel):
    """Credentials necessary to access Pinecone vector database."""
    api_key: str

class DRPineconeVectorStore(PineconeVectorStore):

    async def asimilarity_search_by_vector_with_scores_and_vectors(self, embedding: list[float], k: int=4, filter: dict[str, Any] | None=None) -> list[tuple[Document, float, list[float]]]:
        """Return pinecone documents most similar to embedding, along with scores and vectors."""
        async with self._async_index_context() as idx:
            results = await idx.query(vector=embedding, top_k=k, include_metadata=True, namespace=self._namespace, include_values=True, filter=filter)
        return self._get_docs_scores_vectors_from_matches(results['matches'])

    def _get_docs_scores_vectors_from_matches(self, matches: list[dict]) -> list[tuple[Document, float, list[float]]]:
        docs_scores_vectors = []
        for res in matches:
            metadata = res['metadata']
            id = res.get('id')
            if self._text_key in metadata:
                text = metadata.pop(self._text_key)
                score = res['score']
                vector = res['values']
                docs_scores_vectors.append((Document(id=id, page_content=text, metadata=metadata), score, vector))
            else:
                logger.warning('Found document with no text. Skipping.')
        return docs_scores_vectors

    async def ammr_search_by_vector_with_scores_and_vectors(self, embedding: list[float], k: int=4, fetch_k: int=20, lambda_mult: float=0.5, filter: dict[str, Any] | None=None) -> list[tuple[Document, float, list[float]]]:
        async with self._async_index_context() as idx:
            results = await idx.query(vector=embedding, top_k=fetch_k, include_values=True, include_metadata=True, namespace=self._namespace, filter=filter)
        mmr_selected = maximal_marginal_relevance(np.array([embedding], dtype=np.float32), [item['values'] for item in results['matches']], k=k, lambda_mult=lambda_mult)
        selected = [results['matches'][i] for i in mmr_selected]
        return self._get_docs_scores_vectors_from_matches(selected)

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        docs = []
        async with self._async_index_context() as idx:
            results = await idx.fetch(ids=ids)
        for id, res in results['vectors'].items():
            metadata = res['metadata']
            if self._text_key in metadata:
                text = metadata.pop(self._text_key)
                metadata[MetadataColumnNames.doc_vector.value] = res['values']
                docs.append(Document(id=id, page_content=text, metadata=metadata))
            else:
                logger.warning('Found document with no text. Skipping.')
        return docs

    async def aadd_embeddings(self, text_embeddings: Iterable[tuple[str, list[float]]], metadatas: list[dict] | None=None, ids: list[str] | None=None, batch_size: int=32, **kwargs: Any) -> list[str]:
        """Add the given texts and embeddings to the store.

        Args:
            text_embeddings: Iterable pairs of string and embedding to
                add to the store.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.

        Returns
        -------
            List of ids from adding the texts into the store.
        """
        vector_tuples = construct_vector_tuples(self._text_key, ids, metadatas, text_embeddings)
        try:
            first_vector_tuple = vector_tuples[0]
            size = len(json.dumps(first_vector_tuple).encode('utf-8'))
            batch_size = min(PINECONE_MAX_REQUEST_SIZE // size, PINECONE_MAX_BATCH_SIZE)
        except TypeError:
            pass
        async with PineconeClient(api_key=self._pinecone_api_key.get_secret_value(), source_tag='langchain') as client:
            async with client.IndexAsyncio(host=self.index.config.host) as idx:
                for batch_vector_tuples in batch_iterate(batch_size, vector_tuples):
                    await self._retriable_upsert(idx=idx, vectors=batch_vector_tuples, **kwargs)
        return [tup[0] for tup in vector_tuples]

    @backoff.on_exception(backoff.expo, ServiceException, max_tries=7, factor=10, jitter=random_jitter)
    async def _retriable_upsert(self, idx: IndexAsyncio, vectors: list[VectorTupleWithMetadata], **kwargs: Any) -> None:
        await idx.upsert(vectors=vectors, namespace=self._namespace, **kwargs)

def get_pinecone_vector_store_from_credentials(credentials: PineconeCredentials, index_name: str, embedder: Embeddings) -> DRPineconeVectorStore:
    try:
        return DRPineconeVectorStore(embedding=ConnectedVectorStoreEmbeddings(embedder), pinecone_api_key=credentials.api_key, index_name=index_name)
    except NotFoundException:
        raise ValueError('Connected vector database index was not found.')
    except UnauthorizedException:
        raise ValueError('Connected vector database credentials are not correct.')
    except ForbiddenException:
        raise ValueError('You do not have permission to access the connected vector database.')
    except PineconeApiException as e:
        logger.bind(message=str(e)).warning('Failed to connect to external vector database')
        raise ValueError('Could not connect to the external vector database.')