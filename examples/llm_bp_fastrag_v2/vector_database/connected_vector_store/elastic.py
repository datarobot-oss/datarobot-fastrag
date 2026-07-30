# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from __future__ import annotations
import itertools
from base64 import b64encode
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Sequence
from aiohttp.client import DEFAULT_TIMEOUT
from elasticsearch import ApiError
from elasticsearch import AsyncElasticsearch
from elasticsearch import AuthenticationException
from elasticsearch import AuthorizationException
from elasticsearch import NotFoundError
from langchain_core.documents import Document
from langchain_elasticsearch import AsyncElasticsearchStore
from loguru import logger
from pydantic import BaseModel
from pydantic import model_validator
from i18n import gettext
from vector_database.connected_vector_store.common import ConnectedVectorStoreEmbeddings
from vector_database.connected_vector_store.common import ElasticsearchConnectionParams
from vector_database.dr_embeddings import Embeddings
from vector_database.inference.entities import MetadataColumnNames
from vector_database.inference.entities import MetadataFilterOperators
from vector_database.inference.retrieval import maximal_marginal_relevance

def _hits_to_docs_scores(hits: list[dict[str, Any]], content_field: str, fields: list[str] | None=None, doc_builder: Callable[[dict], Document] | None=None) -> list[tuple[Document, float]]:
    """Taken from langchain_elasticsearch._utilities"""
    if fields is None:
        fields = []
    documents = []

    def default_doc_builder(hit: dict) -> Document:
        return Document(page_content=hit['_source'].get(content_field, ''), metadata=hit['_source'].get('metadata', {}))
    doc_builder = doc_builder or default_doc_builder
    for hit in hits:
        for field in fields:
            if 'metadata' not in hit['_source']:
                hit['_source']['metadata'] = {}
            if field in hit['_source'] and field not in ['metadata', content_field]:
                hit['_source']['metadata'][field] = hit['_source'][field]
        doc = doc_builder(hit)
        documents.append((doc, hit['_score']))
    return documents

def _build_comparison_filter(key: str, operator: str, operand: Any) -> dict[str, Any]:
    match operator:
        case MetadataFilterOperators.EQ:
            result = {'term': {key: operand}}
        case MetadataFilterOperators.NE:
            result = {'bool': {'must_not': {'term': {key: operand}}}}
        case MetadataFilterOperators.GT:
            result = {'range': {key: {'gt': operand}}}
        case MetadataFilterOperators.GTE:
            result = {'range': {key: {'gte': operand}}}
        case MetadataFilterOperators.LT:
            result = {'range': {key: {'lt': operand}}}
        case MetadataFilterOperators.LTE:
            result = {'range': {key: {'lte': operand}}}
        case MetadataFilterOperators.IN:
            result = {'terms': {key: operand}}
        case MetadataFilterOperators.NIN:
            result = {'bool': {'must_not': {'terms': {key: operand}}}}
        case _:
            raise ValueError(gettext('The metadata filter contains an unsupported operator. Valid operators: {supported_operators}').format(supported_operators='$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin'))
    return result

def translate_filter_to_elasticsearch(filter: dict[str, Any]) -> list[dict[str, Any]]:
    output_filter = []
    for field, value in filter.items():
        if field == MetadataFilterOperators.AND.value:
            output_filter.append({'bool': {'must': list(itertools.chain.from_iterable([translate_filter_to_elasticsearch(f) for f in value]))}})
        elif field == MetadataFilterOperators.OR.value:
            output_filter.append({'bool': {'should': list(itertools.chain.from_iterable([translate_filter_to_elasticsearch(f) for f in value]))}})
        elif isinstance(value, dict):
            if len(value) != 1:
                raise ValueError(gettext('Invalid filter dictionary structure.'))
            operator, operand = tuple(value.items())[0]
            output_filter.append(_build_comparison_filter('metadata.' + field + '.keyword', operator, operand))
        else:
            output_filter.append({'term': {'metadata.' + field + '.keyword': value}})
    return output_filter

class ElasticsearchCredentials(BaseModel):
    """Credentials necessary to access Elasticsearch vector database."""
    api_key: str | None
    username: str | None
    password: str | None

    @model_validator(mode='after')
    def validate_either_api_key_or_basic_credentials(self) -> ElasticsearchCredentials:
        if self.api_key is None and (self.username is None or self.password is None):
            raise ValueError(gettext('Either `api_key` or `username` and `password` must be present in the credentials.'))
        return self

class DRElasticsearchStore(AsyncElasticsearchStore):

    async def asimilarity_search_by_vector_with_scores_and_vectors(self, embedding: list[float], k: int=4, filter: dict[str, Any] | None=None) -> list[tuple[Document, float, list[float]]]:
        """Return Elasticsearch documents most similar to query, along with scores and vectors.
        Adapted from asimilarity_search_by_vector_with_relevance_scores.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Array of Elasticsearch filter clauses to apply to the query.

        Returns
        -------
            List of Documents most similar to the embedding and score for each
        """
        fields = [self._store.vector_field]
        elastic_filter = translate_filter_to_elasticsearch(filter) if filter else None
        hits = await self._store.search(query=None, query_vector=embedding, k=k, fields=fields, filter=elastic_filter)
        return self._get_docs_scores_vectors_from_hits(hits, fields)

    def _get_docs_scores_vectors_from_hits(self, hits: list[dict[str, Any]], fields: list[str]) -> list[tuple[Document, float, list[float]]]:

        def doc_builder(hit: dict) -> Document:
            return Document(page_content=hit['_source'].get(self.query_field, ''), metadata=hit['_source'].get('metadata', {}), id=hit['_id'])
        docs_and_scores = _hits_to_docs_scores(hits=hits, content_field=self.query_field, fields=fields, doc_builder=doc_builder)
        docs_scores_and_vectors = []
        for doc, score in docs_and_scores:
            docs_scores_and_vectors.append((doc, score, doc.metadata.pop(self._store.vector_field, None)))
        return docs_scores_and_vectors

    async def ammr_search_by_vector_with_scores_and_vectors(self, embedding: list[float], k: int=4, fetch_k: int=20, lambda_mult: float=0.5, filter: dict[str, Any] | None=None) -> list[tuple[Document, float, list[float]]]:
        fields = [self._store.vector_field]
        elastic_filter = translate_filter_to_elasticsearch(filter) if filter else None
        # Fetch the initial documents
        got_hits = await self._store.search(query=None, query_vector=embedding, k=fetch_k, fields=fields, filter=elastic_filter)
        # Get the embeddings for the fetched documents
        got_embeddings = [hit['_source'][self.vector_query_field] for hit in got_hits]
        # Select documents using maximal marginal relevance
        selected_indices = maximal_marginal_relevance(embedding, got_embeddings, lambda_mult=lambda_mult, k=k)
        selected_hits = [got_hits[i] for i in selected_indices]
        return self._get_docs_scores_vectors_from_hits(selected_hits, fields)

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        documents = []
        if ids:
            response = await self.client.mget(index=self._store.index, ids=ids)
            hits = response['docs']
            for hit in hits:
                if hit.get('found'):
                    id = hit.get('_id')
                    text = hit['_source'].get(self.query_field, '')
                    metadata = hit['_source'].get('metadata', {})
                    vector = hit['_source'].get(self._store.vector_field)
                    if vector:
                        metadata[MetadataColumnNames.doc_vector.value] = vector
                    documents.append(Document(page_content=text, metadata=metadata, id=id))
        return documents

    async def aadd_embeddings(self, text_embeddings: Iterable[tuple[str, list[float]]], metadatas: list[dict] | None=None, ids: list[str] | None=None, create_index_if_not_exists: bool=True, **kwargs: Any) -> list[str]:  # type: ignore [override]
        return await super().aadd_embeddings(text_embeddings=text_embeddings, metadatas=metadatas, ids=ids, create_index_if_not_exists=create_index_if_not_exists, bulk_kwargs={'initial_backoff': 10, 'max_retries': 7})

def get_elastic_vector_store_from_credentials(credentials: ElasticsearchCredentials, external_vector_database_connection: ElasticsearchConnectionParams, index_name: str, embedder: Embeddings) -> DRElasticsearchStore:
    basic_auth = None
    if credentials.username and credentials.password:  # type: ignore[union-attr]
        # type: ignore[union-attr]
        basic_auth = b64encode(f'{credentials.username}:{credentials.password}'.encode()).decode('ascii')
    try:
        connection = AsyncElasticsearch(external_vector_database_connection.url, cloud_id=external_vector_database_connection.cloud_id, api_key=credentials.api_key, basic_auth=basic_auth, request_timeout=DEFAULT_TIMEOUT.total)
        return DRElasticsearchStore(index_name=index_name, embedding=ConnectedVectorStoreEmbeddings(embedder), client=connection)
    except NotFoundError:
        raise ValueError('Connected vector database index was not found.')
    except AuthenticationException:
        raise ValueError('Connected vector database credentials are not correct.')
    except AuthorizationException:
        raise ValueError('You do not have permission to access the connected vector database.')
    except ApiError as e:
        logger.bind(message=str(e)).warning('Failed to connect to external vector database')
        raise ValueError('Could not connect to the external vector database.')