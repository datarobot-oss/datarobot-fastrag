# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from __future__ import annotations
from typing import Any
from langchain_core.documents import Document
from loguru import logger
from i18n import gettext
from vector_database.connected_vector_store.common import ConnectedVectorDatabaseType
from vector_database.connected_vector_store.common import VectorDatabaseConnection
from vector_database.connected_vector_store.elastic import DRElasticsearchStore
from vector_database.connected_vector_store.elastic import ElasticsearchCredentials
from vector_database.connected_vector_store.elastic import get_elastic_vector_store_from_credentials
from vector_database.connected_vector_store.milvus import DRMilvusVectorStore
from vector_database.connected_vector_store.milvus import MilvusCredentials
from vector_database.connected_vector_store.milvus import get_milvus_vector_store_from_credentials
from vector_database.connected_vector_store.pinecone import DRPineconeVectorStore
from vector_database.connected_vector_store.pinecone import PineconeCredentials
from vector_database.connected_vector_store.pinecone import get_pinecone_vector_store_from_credentials
from vector_database.dr_embeddings import Embeddings
from vector_database.inference.entities import MetadataColumnNames
from vector_database.inference.entities import MetadataFilterOperators
from vector_database.inference.entities import RetrievalMode
from vector_database.inference.entities import SearchResult
from vector_database.inference.entities import VectorStore
from vector_database.inference.retrieval import DEFAULT_ADD_NEIGHBOR_CHUNKS
from vector_database.inference.retrieval import DEFAULT_FILTER
from vector_database.inference.retrieval import DEFAULT_K
from vector_database.inference.retrieval import _default_relevance_score_fn
VectorDatabaseCredentials = PineconeCredentials | ElasticsearchCredentials | MilvusCredentials
ConnectedVectorStoreType = DRPineconeVectorStore | DRElasticsearchStore | DRMilvusVectorStore

class InvalidVectorDatabaseError(Exception):
    """Subclass when we have unusable for some reason VD."""

class ConnectedVectorStore(VectorStore):

    def __init__(self, vector_store: ConnectedVectorStoreType):
        """Initialize the connected vector store.

        Parameters
        ----------
        vector_store
            The connected vector store
        """
        self.vector_store = vector_store
        self.logger = logger

    async def search(self, query: str, k: int=DEFAULT_K, filter: dict[str, Any] | None=DEFAULT_FILTER, add_neighbor_chunks: bool=DEFAULT_ADD_NEIGHBOR_CHUNKS, retrieval_mode: RetrievalMode=RetrievalMode.SIMILARITY, maximal_marginal_relevance_lambda: float=0.5, **kwargs: Any) -> SearchResult:
        """Return docs most similar to query asynchronously.
        The similarity scores are added to the metadata of the documents.

        Parameters
        ----------
        query
            Text to look up documents similar to.
        k
            Number of Documents to return.
        filter
            Metadata filter dictionary or function.
        add_neighbor_chunks
            Whether to add neighbour document chunks to the result.
            If True, the number of documents returned will be roughly 3 * k.
            Because if index i is found, we also return i-1 and i+1.
        retrieval_mode
            Retrieval mode to use, similarity or maximal marginal relevance.
        maximal_marginal_relevance_lambda
            Lambda parameter for MMR. Smaller favors similarity, larger favors diversity.

        Returns
        -------
            List of Documents most similar to the query and the query embeddings.
        """
        self.logger.debug('Running similarity search')
        if not self.vector_store.embeddings:
            raise InvalidVectorDatabaseError('No embedding function found')
        self.logger.debug('Embedding the prompt')
        query_embedding = await self.vector_store.embeddings.aembed_query(query)
        self.logger.debug('Finished embedding the prompt')
        if retrieval_mode == RetrievalMode.SIMILARITY:
            docs_scores_and_vectors = await self.vector_store.asimilarity_search_by_vector_with_scores_and_vectors(query_embedding, k, filter)
        elif retrieval_mode == RetrievalMode.MAXIMAL_MARGINAL_RELEVANCE:
            docs_scores_and_vectors = await self.vector_store.ammr_search_by_vector_with_scores_and_vectors(query_embedding, k=k, fetch_k=5 * k, lambda_mult=maximal_marginal_relevance_lambda, filter=filter)
        else:
            raise InvalidVectorDatabaseError('Unknown retrieval mode')
        documents = []
        for doc, score, vector in docs_scores_and_vectors:
            doc.metadata[MetadataColumnNames.similarity_score.value] = _default_relevance_score_fn(score)
            doc.metadata[MetadataColumnNames.doc_vector.value] = vector
            try:
                doc.metadata[MetadataColumnNames.chunk_id.value] = int(doc.id)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                pass
            documents.append(doc)
        self.logger.debug('Finished running similarity search')
        if add_neighbor_chunks:
            self.logger.debug('Adding neighbour chunks')
            k = k * 3
            documents = await self.add_neighbor_chunks(documents, filter)
        return (documents[:k], query_embedding)

    async def add_neighbor_chunks(self, docs: list[Document], filter: dict[str, Any] | None=DEFAULT_FILTER, **kwargs: Any) -> list[Document]:
        """Return docs in order of descending similarity with neighbor chunks before and after."""
        doc_ids = [doc.id for doc in docs if doc.id is not None]
        doc_ids_to_return = [*dict.fromkeys([str(j) for i in doc_ids for j in range(int(i) - 1, int(i) + 2)])]
        doc_ids_to_retrieve = list(set(doc_ids_to_return) - set(doc_ids))
        neighbor_docs = await self.vector_store.aget_by_ids(doc_ids_to_retrieve)
        for doc in neighbor_docs:
            try:
                doc.metadata[MetadataColumnNames.chunk_id.value] = int(doc.id)  # type: ignore[arg-type]
            except ValueError:
                pass
        if filter:
            neighbor_docs = filter_docs(neighbor_docs, filter)
        all_docs = {doc.id: doc for doc in docs + neighbor_docs}
        return [all_docs[id] for id in doc_ids_to_return if id in all_docs]

def filter_docs(docs: list[Document], filter: dict[str, Any]) -> list[Document]:
    return [doc for doc in docs if passes_filter(doc.metadata, filter)]

def passes_filter(metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
    for field, value in filter.items():
        if field == MetadataFilterOperators.AND.value:
            for f in value:
                if not passes_filter(metadata, f):
                    return False
        elif field == MetadataFilterOperators.OR.value:
            for f in value:
                if passes_filter(metadata, f):
                    break
            else:
                return False
        elif isinstance(value, dict):
            if len(value) != 1:
                raise ValueError(gettext('Invalid filter dictionary structure.'))
            operator, operand = tuple(value.items())[0]
            if not _passes_comparison_filter(metadata, field, operator, operand):
                return False
        elif not metadata.get(field) == value:
            return False
    return True

def _passes_comparison_filter(metadata: dict[str, Any], key: str, operator: str, operand: Any) -> bool:
    match operator:
        case MetadataFilterOperators.EQ:
            result = metadata.get(key) == operand
        case MetadataFilterOperators.NE:
            result = metadata.get(key) != operand
        case MetadataFilterOperators.GT:
            result = metadata.get(key) > operand
        case MetadataFilterOperators.GTE:
            result = metadata.get(key) >= operand
        case MetadataFilterOperators.LT:
            result = metadata.get(key) < operand
        case MetadataFilterOperators.LTE:
            result = metadata.get(key) <= operand
        case MetadataFilterOperators.IN:
            result = metadata.get(key) in operand
        case MetadataFilterOperators.NIN:
            result = metadata.get(key) not in operand
        case _:
            raise ValueError(gettext('The metadata filter contains an unsupported operator. Valid operators: {supported_operators}').format(supported_operators='$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin'))
    return result

def get_vector_store_from_credentials(credentials: VectorDatabaseCredentials, external_vector_database_connection: VectorDatabaseConnection, index_name: str, embedder: Embeddings) -> ConnectedVectorStoreType:
    match external_vector_database_connection.type:
        case ConnectedVectorDatabaseType.ELASTICSEARCH:  # type: ignore[arg-type]
            return get_elastic_vector_store_from_credentials(credentials, external_vector_database_connection, index_name, embedder)
        case ConnectedVectorDatabaseType.PINECONE:  # type: ignore[arg-type]
            return get_pinecone_vector_store_from_credentials(credentials, index_name, embedder)
        case ConnectedVectorDatabaseType.MILVUS:  # type: ignore[arg-type]
            return get_milvus_vector_store_from_credentials(credentials, external_vector_database_connection, index_name, embedder)
        case _:
            raise ValueError('Unknown external vector database type')