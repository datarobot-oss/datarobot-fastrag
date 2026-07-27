# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from __future__ import annotations

import itertools
import json
import uuid
from base64 import b64encode
from enum import StrEnum
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Sequence
from typing import cast

import backoff
import numpy as np
from aiohttp.client import DEFAULT_TIMEOUT
from backoff import random_jitter
from elasticsearch import AsyncElasticsearch
from i18n import gettext
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from langchain_core.utils import batch_iterate
from langchain_elasticsearch import AsyncElasticsearchStore
from langchain_milvus import Milvus as MilvusVectorStore
from langchain_pinecone import PineconeVectorStore
from loguru import logger
from pinecone import PineconeAsyncio as PineconeClient
from pinecone import ServiceException
from pinecone.db_data import IndexAsyncio
from pinecone.db_data.types import VectorTupleWithMetadata
from pydantic import BaseModel
from pydantic import model_validator
from pymilvus import Hit
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
from vector_database.inference.retrieval import maximal_marginal_relevance

PINECONE_MAX_BATCH_SIZE = 1000
PINECONE_MAX_REQUEST_SIZE = 2 * 1000 * 1000  # 2 * 1024 * 1024 but let's leave a buffer
PINECONE_MAX_METADATA_SIZE = 40 * 1024
MILVUS_MAX_VARCHAR_SIZE = 65535


class PineconeCredentials(BaseModel):
    """Credentials necessary to access Pinecone vector database."""

    api_key: str


class ElasticsearchCredentials(BaseModel):
    """Credentials necessary to access Elasticsearch vector database."""

    api_key: str | None
    username: str | None
    password: str | None

    @model_validator(mode="after")
    def validate_either_api_key_or_basic_credentials(self) -> ElasticsearchCredentials:
        if self.api_key is None and (self.username is None or self.password is None):
            raise ValueError(
                gettext(
                    "Either `api_key` or `username` and `password` must be present in the credentials."
                )
            )
        return self


class MilvusCredentials(BaseModel):
    """Credentials necessary to access Milvus vector database."""

    api_key: str | None
    username: str | None
    password: str | None

    @model_validator(mode="after")
    def validate_either_api_key_or_basic_credentials(self) -> MilvusCredentials:
        if self.api_key is None and (self.username is None or self.password is None):
            raise ValueError(
                gettext(
                    "Either `api_key` or `username` and `password` must be present in the credentials."
                )
            )
        return self


VectorDatabaseCredentials = PineconeCredentials | ElasticsearchCredentials | MilvusCredentials


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


class DRElasticsearchStore(AsyncElasticsearchStore):
    async def asimilarity_search_by_vector_with_scores_and_vectors(
        self, embedding: list[float], k: int = 4, filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float, list[float]]]:
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
        hits = await self._store.search(
            query=None, query_vector=embedding, k=k, fields=fields, filter=elastic_filter
        )
        return self._get_docs_scores_vectors_from_hits(hits, fields)

    def _get_docs_scores_vectors_from_hits(
        self, hits: list[dict[str, Any]], fields: list[str]
    ) -> list[tuple[Document, float, list[float]]]:

        def doc_builder(hit: dict) -> Document:
            return Document(
                page_content=hit["_source"].get(self.query_field, ""),
                metadata=hit["_source"].get("metadata", {}),
                id=hit["_id"],
            )

        docs_and_scores = _hits_to_docs_scores(
            hits=hits, content_field=self.query_field, fields=fields, doc_builder=doc_builder
        )
        docs_scores_and_vectors = []
        for doc, score in docs_and_scores:
            docs_scores_and_vectors.append(
                (doc, score, doc.metadata.pop(self._store.vector_field, None))
            )
        return docs_scores_and_vectors

    async def ammr_search_by_vector_with_scores_and_vectors(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float, list[float]]]:
        fields = [self._store.vector_field]
        elastic_filter = translate_filter_to_elasticsearch(filter) if filter else None
        # Fetch the initial documents
        got_hits = await self._store.search(
            query=None, query_vector=embedding, k=fetch_k, fields=fields, filter=elastic_filter
        )
        # Get the embeddings for the fetched documents
        got_embeddings = [hit["_source"][self.vector_query_field] for hit in got_hits]
        # Select documents using maximal marginal relevance
        selected_indices = maximal_marginal_relevance(
            embedding, got_embeddings, lambda_mult=lambda_mult, k=k
        )
        selected_hits = [got_hits[i] for i in selected_indices]
        return self._get_docs_scores_vectors_from_hits(selected_hits, fields)

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        documents = []
        if ids:
            response = await self.client.mget(index=self._store.index, ids=ids)
            hits = response["docs"]
            for hit in hits:
                if hit.get("found"):
                    id = hit.get("_id")
                    text = hit["_source"].get(self.query_field, "")
                    metadata = hit["_source"].get("metadata", {})
                    vector = hit["_source"].get(self._store.vector_field)
                    if vector:
                        metadata[MetadataColumnNames.doc_vector.value] = vector
                    documents.append(Document(page_content=text, metadata=metadata, id=id))
        return documents

    async def aadd_embeddings(
        self,
        text_embeddings: Iterable[tuple[str, list[float]]],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        create_index_if_not_exists: bool = True,
        **kwargs: Any,
    ) -> list[str]:  # type: ignore [override]
        return await super().aadd_embeddings(
            text_embeddings=text_embeddings,
            metadatas=metadatas,
            ids=ids,
            create_index_if_not_exists=create_index_if_not_exists,
            bulk_kwargs={"initial_backoff": 10, "max_retries": 7},
        )


def _hits_to_docs_scores(
    hits: list[dict[str, Any]],
    content_field: str,
    fields: list[str] | None = None,
    doc_builder: Callable[[dict], Document] | None = None,
) -> list[tuple[Document, float]]:
    """Taken from langchain_elasticsearch._utilities"""
    if fields is None:
        fields = []
    documents = []

    def default_doc_builder(hit: dict) -> Document:
        return Document(
            page_content=hit["_source"].get(content_field, ""),
            metadata=hit["_source"].get("metadata", {}),
        )

    doc_builder = doc_builder or default_doc_builder
    for hit in hits:
        for field in fields:
            if "metadata" not in hit["_source"]:
                hit["_source"]["metadata"] = {}
            if field in hit["_source"] and field not in ["metadata", content_field]:
                hit["_source"]["metadata"][field] = hit["_source"][field]
        doc = doc_builder(hit)
        documents.append((doc, hit["_score"]))
    return documents


def translate_filter_to_elasticsearch(filter: dict[str, Any]) -> list[dict[str, Any]]:
    output_filter = []
    for field, value in filter.items():
        if field == MetadataFilterOperators.AND.value:
            output_filter.append(
                {
                    "bool": {
                        "must": list(
                            itertools.chain.from_iterable(
                                [translate_filter_to_elasticsearch(f) for f in value]
                            )
                        )
                    }
                }
            )
        elif field == MetadataFilterOperators.OR.value:
            output_filter.append(
                {
                    "bool": {
                        "should": list(
                            itertools.chain.from_iterable(
                                [translate_filter_to_elasticsearch(f) for f in value]
                            )
                        )
                    }
                }
            )
        elif isinstance(value, dict):
            if len(value) != 1:
                raise ValueError(gettext("Invalid filter dictionary structure."))
            operator, operand = tuple(value.items())[0]
            output_filter.append(
                _build_comparison_filter("metadata." + field + ".keyword", operator, operand)
            )
        else:
            output_filter.append({"term": {"metadata." + field + ".keyword": value}})
    return output_filter


def _build_comparison_filter(key: str, operator: str, operand: Any) -> dict[str, Any]:
    match operator:
        case MetadataFilterOperators.EQ:
            result = {"term": {key: operand}}
        case MetadataFilterOperators.NE:
            result = {"bool": {"must_not": {"term": {key: operand}}}}
        case MetadataFilterOperators.GT:
            result = {"range": {key: {"gt": operand}}}
        case MetadataFilterOperators.GTE:
            result = {"range": {key: {"gte": operand}}}
        case MetadataFilterOperators.LT:
            result = {"range": {key: {"lt": operand}}}
        case MetadataFilterOperators.LTE:
            result = {"range": {key: {"lte": operand}}}
        case MetadataFilterOperators.IN:
            result = {"terms": {key: operand}}
        case MetadataFilterOperators.NIN:
            result = {"bool": {"must_not": {"terms": {key: operand}}}}
        case _:
            raise ValueError(
                gettext(
                    "The metadata filter contains an unsupported operator. Valid operators: {supported_operators}"
                ).format(supported_operators="$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin")
            )
    return result


def construct_vector_tuples(
    text_key: str,
    ids: list[str] | None,
    metadatas: list[dict] | None,
    text_embeddings: Iterable[tuple[str, list[float]]],
) -> list[tuple[str, list[float], dict]]:
    texts, embeddings = zip(*text_embeddings)
    ids = ids or [str(uuid.uuid4()) for _ in texts]
    metadatas = metadatas or [{} for _ in texts]
    vector_tuples = []
    for metadata, text, id, embedding in zip(metadatas, texts, ids, embeddings):
        metadata[text_key] = text
        try:
            metadata_size = len(json.dumps(metadata).encode("utf-8"))
            if metadata_size > PINECONE_MAX_METADATA_SIZE:
                continue
        except TypeError:
            pass
        vector_tuples.append((id, embedding, metadata))
    return vector_tuples


class DRPineconeVectorStore(PineconeVectorStore):
    async def asimilarity_search_by_vector_with_scores_and_vectors(
        self, embedding: list[float], k: int = 4, filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float, list[float]]]:
        """Return pinecone documents most similar to embedding, along with scores and vectors."""
        async with self.async_index as idx:
            results = await idx.query(
                vector=embedding,
                top_k=k,
                include_metadata=True,
                namespace=self._namespace,
                include_values=True,
                filter=filter,
            )
        return self._get_docs_scores_vectors_from_matches(results["matches"])

    def _get_docs_scores_vectors_from_matches(
        self, matches: list[dict]
    ) -> list[tuple[Document, float, list[float]]]:
        docs_scores_vectors = []
        for res in matches:
            metadata = res["metadata"]
            id = res.get("id")
            if self._text_key in metadata:
                text = metadata.pop(self._text_key)
                score = res["score"]
                vector = res["values"]
                docs_scores_vectors.append(
                    (Document(id=id, page_content=text, metadata=metadata), score, vector)
                )
            else:
                logger.warning("Found document with no text. Skipping.")
        return docs_scores_vectors

    async def ammr_search_by_vector_with_scores_and_vectors(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float, list[float]]]:
        async with self.async_index as idx:
            results = await idx.query(
                vector=embedding,
                top_k=fetch_k,
                include_values=True,
                include_metadata=True,
                namespace=self._namespace,
                filter=filter,
            )
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            [item["values"] for item in results["matches"]],
            k=k,
            lambda_mult=lambda_mult,
        )
        selected = [results["matches"][i] for i in mmr_selected]
        return self._get_docs_scores_vectors_from_matches(selected)

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        docs = []
        async with self.async_index as idx:
            results = await idx.fetch(ids=ids)
        for id, res in results["vectors"].items():
            metadata = res["metadata"]
            if self._text_key in metadata:
                text = metadata.pop(self._text_key)
                metadata[MetadataColumnNames.doc_vector.value] = res["values"]
                docs.append(Document(id=id, page_content=text, metadata=metadata))
            else:
                logger.warning("Found document with no text. Skipping.")
        return docs

    async def aadd_embeddings(
        self,
        text_embeddings: Iterable[tuple[str, list[float]]],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> list[str]:
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
            size = len(json.dumps(first_vector_tuple).encode("utf-8"))
            batch_size = min(PINECONE_MAX_REQUEST_SIZE // size, PINECONE_MAX_BATCH_SIZE)
        except TypeError:
            pass
        async with PineconeClient(api_key=self._pinecone_api_key, source_tag="langchain") as client:
            async with client.IndexAsyncio(host=self.index.config.host) as idx:
                for batch_vector_tuples in batch_iterate(batch_size, vector_tuples):
                    await self._retriable_upsert(idx=idx, vectors=batch_vector_tuples, **kwargs)
        return [tup[0] for tup in vector_tuples]

    @backoff.on_exception(
        backoff.expo, ServiceException, max_tries=7, factor=10, jitter=random_jitter
    )
    async def _retriable_upsert(
        self, idx: IndexAsyncio, vectors: list[VectorTupleWithMetadata], **kwargs: Any
    ) -> None:
        await idx.upsert(vectors=vectors, namespace=self._namespace, **kwargs)


def filter_out_big_chunks(
    texts: list[str], embeddings: list[list[float]], metadatas: list[dict] | None = None
) -> tuple[list[str], list[list[float]], list[dict] | None]:

    def value_too_big(value: Any) -> bool:
        if isinstance(value, str):
            return len(value.encode("utf-8")) > MILVUS_MAX_VARCHAR_SIZE
        return False

    filtered_texts, filtered_embeddings, filtered_metadatas = ([], [], [])
    if metadatas is None:
        for text, embedding in zip(texts, embeddings):
            if value_too_big(text):
                continue
            filtered_texts.append(text)
            filtered_embeddings.append(embedding)
        return (filtered_texts, filtered_embeddings, None)
    for text, embedding, metadata in zip(texts, embeddings, metadatas):
        if value_too_big(text):
            continue
        filtered_texts.append(text)
        filtered_embeddings.append(embedding)
        if filtered_metadata := {k: v for k, v in metadata.items() if not value_too_big(v)}:
            filtered_metadatas.append(filtered_metadata)
    return (filtered_texts, filtered_embeddings, filtered_metadatas)


def translate_filter_to_milvus(filter: dict[str, Any] | None) -> str:

    def format_value(value: Any) -> str:
        """Format value for Milvus filter, handling booleans as lowercase."""
        if isinstance(value, bool):
            return "true" if value else "false"
        return repr(value)

    def build_comparison_filter(key: str, operator: str, operand: Any) -> str:
        operator_map = {
            MetadataFilterOperators.EQ: "==",
            MetadataFilterOperators.NE: "!=",
            MetadataFilterOperators.GT: ">",
            MetadataFilterOperators.GTE: ">=",
            MetadataFilterOperators.LT: "<",
            MetadataFilterOperators.LTE: "<=",
            MetadataFilterOperators.IN: "in",
            MetadataFilterOperators.NIN: "not in",
        }
        try:
            operator_str = operator_map[MetadataFilterOperators(operator)]
            return " ".join((key, operator_str, format_value(operand)))
        except (KeyError, ValueError):
            raise ValueError(
                gettext(
                    "The metadata filter contains an unsupported operator. Valid operators: {supported_operators}"
                ).format(supported_operators="$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin")
            )

    if filter is None:
        return ""
    milvus_filter = []
    for key, value in filter.items():
        match key:
            case MetadataFilterOperators.AND:
                milvus_filter.append(" and ".join([translate_filter_to_milvus(f) for f in value]))
            case MetadataFilterOperators.OR:
                expression = " or ".join([translate_filter_to_milvus(f) for f in value])
                milvus_filter.append(f"({expression})")
            case _:
                if isinstance(value, dict):
                    if len(value) != 1:
                        raise ValueError(gettext("Invalid filter dictionary structure."))
                    operator, operand = tuple(value.items())[0]
                    milvus_filter.append(build_comparison_filter(key, operator, operand))
                else:
                    milvus_filter.append(f"{key} == {format_value(value)}")
    return " and ".join(milvus_filter)


class DRMilvusVectorStore(MilvusVectorStore):
    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:  # type: ignore[override]
        texts = list(texts)
        if not texts:
            return []
        if self.embeddings:
            embeddings = await self.embeddings.aembed_documents(texts)
        text_embeddings = list(zip(texts, embeddings))
        return await self.aadd_embeddings(
            text_embeddings=text_embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    async def aadd_embeddings(
        self,
        text_embeddings: Iterable[tuple[str, list[float]]],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        create_index_if_not_exists: bool = True,
        **kwargs: Any,
    ) -> list[str]:  # type: ignore[override]
        texts, embeddings = map(list, zip(*text_embeddings))
        texts, embeddings, metadatas = filter_out_big_chunks(texts, embeddings, metadatas)  # type: ignore[arg-type]
        return await super().aadd_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            batch_size=32,
            **kwargs,
        )

    async def asimilarity_search_by_vector_with_scores_and_vectors(
        self, embedding: list[float], k: int = 4, filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float, list[float]]]:
        milvus_filter = translate_filter_to_milvus(filter)
        hits = await self.aclient.search(
            collection_name=self.collection_name,
            data=[embedding],
            filter=milvus_filter,
            limit=k,
            output_fields=["*"],
        )
        return self._get_docs_scores_vectors_from_hits(hits[0])

    async def ammr_search_by_vector_with_scores_and_vectors(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float, list[float]]]:
        milvus_filter = translate_filter_to_milvus(filter)
        hits = await self.aclient.search(
            collection_name=self.collection_name,
            data=[embedding],
            filter=milvus_filter,
            limit=fetch_k,
            output_fields=["*"],
        )
        embeddings = [hit[self._vector_field] for hit in hits[0]]
        selected_indices = maximal_marginal_relevance(
            embedding, embeddings, lambda_mult=lambda_mult, k=k
        )
        selected_hits = [hits[0][i] for i in selected_indices]
        return self._get_docs_scores_vectors_from_hits(selected_hits)

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        documents = []
        for doc in await self.aclient.get(self.collection_name, ids):
            doc_id = doc.pop(self._primary_field)
            text = doc.pop(self._text_field, "")
            doc.pop(self._vector_field)
            doc.pop("source")
            documents.append(Document(id=doc_id, page_content=text, metadata=doc))
        return documents

    def _get_docs_scores_vectors_from_hits(
        self, hits: list[Hit]
    ) -> list[tuple[Document, float, list[float]]]:
        docs_scores_vectors = []
        for hit in hits:
            fields = hit.fields
            fields.pop(self._primary_field)
            vector = fields.pop(self._vector_field)
            if not (text := fields.pop(self._text_field, "")):
                logger.warning("Found document with no text. Skipping.")
                continue
            docs_scores_vectors.append(
                (Document(id=hit.pk, page_content=text, metadata=fields), hit.score, vector)
            )
        return docs_scores_vectors

    @property
    def embeddings(self) -> LangChainEmbeddings | None:  # type: ignore[override]
        "\n        Get embedding function.\n\n        This property is used to narrow down return types to LangChainEmbeddings | None.\n        Otherwise, it creates a mess with mypy.\n"
        return cast(LangChainEmbeddings | None, self.embedding_func)


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

    async def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        filter: dict[str, Any] | None = DEFAULT_FILTER,
        add_neighbor_chunks: bool = DEFAULT_ADD_NEIGHBOR_CHUNKS,
        retrieval_mode: RetrievalMode = RetrievalMode.SIMILARITY,
        maximal_marginal_relevance_lambda: float = 0.5,
        **kwargs: Any,
    ) -> SearchResult:
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
        self.logger.debug("Running similarity search")
        if not self.vector_store.embeddings:
            raise InvalidVectorDatabaseError("No embedding function found")
        self.logger.debug("Embedding the prompt")
        query_embedding = await self.vector_store.embeddings.aembed_query(query)
        self.logger.debug("Finished embedding the prompt")
        if retrieval_mode == RetrievalMode.SIMILARITY:
            docs_scores_and_vectors = (
                await self.vector_store.asimilarity_search_by_vector_with_scores_and_vectors(
                    query_embedding, k, filter
                )
            )
        elif retrieval_mode == RetrievalMode.MAXIMAL_MARGINAL_RELEVANCE:
            docs_scores_and_vectors = (
                await self.vector_store.ammr_search_by_vector_with_scores_and_vectors(
                    query_embedding,
                    k=k,
                    fetch_k=5 * k,
                    lambda_mult=maximal_marginal_relevance_lambda,
                    filter=filter,
                )
            )
        else:
            raise InvalidVectorDatabaseError("Unknown retrieval mode")
        documents = []
        for doc, score, vector in docs_scores_and_vectors:
            doc.metadata[MetadataColumnNames.similarity_score.value] = _default_relevance_score_fn(
                score
            )
            doc.metadata[MetadataColumnNames.doc_vector.value] = vector
            try:
                doc.metadata[MetadataColumnNames.chunk_id.value] = int(doc.id)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                pass
            documents.append(doc)
        self.logger.debug("Finished running similarity search")
        if add_neighbor_chunks:
            self.logger.debug("Adding neighbour chunks")
            k = k * 3
            documents = await self.add_neighbor_chunks(documents, filter)
        return (documents[:k], query_embedding)

    async def add_neighbor_chunks(
        self, docs: list[Document], filter: dict[str, Any] | None = DEFAULT_FILTER, **kwargs: Any
    ) -> list[Document]:
        """Return docs in order of descending similarity with neighbor chunks before and after."""
        doc_ids = [doc.id for doc in docs if doc.id is not None]
        doc_ids_to_return = [
            *dict.fromkeys([str(j) for i in doc_ids for j in range(int(i) - 1, int(i) + 2)])
        ]
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
                raise ValueError(gettext("Invalid filter dictionary structure."))
            operator, operand = tuple(value.items())[0]
            if not _passes_comparison_filter(metadata, field, operator, operand):
                return False
        elif not metadata.get(field) == value:
            return False
    return True


def _passes_comparison_filter(
    metadata: dict[str, Any], key: str, operator: str, operand: Any
) -> bool:
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
            raise ValueError(
                gettext(
                    "The metadata filter contains an unsupported operator. Valid operators: {supported_operators}"
                ).format(supported_operators="$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin")
            )
    return result


class ConnectedVectorDatabaseType(StrEnum):
    """Types of external vector database connections."""

    PINECONE = "pinecone"
    ELASTICSEARCH = "elasticsearch"
    MILVUS = "milvus"


class VectorDatabaseConnection(BaseModel):
    type: ConnectedVectorDatabaseType
    url: str | None = None
    cloud_id: str | None = None
    cloud: str | None = None
    region: str | None = None


def get_vector_store_from_credentials(
    credentials: VectorDatabaseCredentials,
    external_vector_database_connection: VectorDatabaseConnection,
    index_name: str,
    embedder: Embeddings,
) -> ConnectedVectorStoreType:
    vector_store: ConnectedVectorStoreType
    match external_vector_database_connection.type:
        case ConnectedVectorDatabaseType.ELASTICSEARCH:
            basic_auth = None
            if credentials.username and credentials.password:  # type: ignore[union-attr]
                # type: ignore[union-attr]
                basic_auth = b64encode(
                    f"{credentials.username}:{credentials.password}".encode()
                ).decode("ascii")
            connection = AsyncElasticsearch(
                external_vector_database_connection.url,
                cloud_id=external_vector_database_connection.cloud_id,
                api_key=credentials.api_key,
                basic_auth=basic_auth,
                request_timeout=DEFAULT_TIMEOUT.total,
            )
            vector_store = DRElasticsearchStore(
                index_name=index_name,
                embedding=ConnectedVectorStoreEmbeddings(embedder),
                es_connection=connection,
            )
        case ConnectedVectorDatabaseType.PINECONE:
            vector_store = DRPineconeVectorStore(
                embedding=ConnectedVectorStoreEmbeddings(embedder),
                pinecone_api_key=credentials.api_key,
                index_name=index_name,
            )
        case ConnectedVectorDatabaseType.MILVUS:
            token = credentials.api_key or f"{credentials.username}:{credentials.password}"  # type: ignore[union-attr]
            vector_store = DRMilvusVectorStore(
                enable_dynamic_field=True,
                collection_name=f"_{index_name}",
                embedding_function=ConnectedVectorStoreEmbeddings(embedder),
                connection_args={"uri": external_vector_database_connection.url, "token": token},
            )
        case _:
            raise ValueError("Unknown external vector database type")
    return vector_store
