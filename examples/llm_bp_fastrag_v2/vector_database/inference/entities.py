# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from abc import ABC
from abc import abstractmethod
from enum import StrEnum
from typing import Any
from typing import Tuple
from langchain_core.documents import Document
DEFAULT_K = 10
DEFAULT_FILTER = None
DEFAULT_ADD_NEIGHBOR_CHUNKS = False
RESCORE_MULTIPLIER = 2
SQLITE_METADATA_TABLE = 'metadata'
# Preserved column names for the document metadata
# backward compatibility with the previous implementation

class MetadataColumnNames(StrEnum):
    source = 'source'
    dataset_version = 'dataset_version'
    page = 'page'
    similarity_score = 'similarity_score'
    chunk_id = 'chunk_id'
    start_index = 'start_index'
    pagebreak_indices = 'pagebreak_indices'  # intermediate field used for chunking, not returned
    content = 'content'  # hardcoded custom model citation field
    doc_vector = '_doc_vector'  # used only by custom model as part of _LLM_CONTEXT for monitoring
    chunk_size = 'chunk_size'  # This is an internal field and won't be visible to users

    @classmethod
    def custom_model_citation_fields(cls) -> list[str]:
        return [cls.content.value, cls.source.value, cls.dataset_version.value, cls.page.value, cls.similarity_score.value, cls.chunk_id.value, cls.start_index.value]

    @classmethod
    def column_always_present_in_metadata_sql(cls) -> set[str]:
        return {cls.chunk_id.value, cls.chunk_size.value}

    @classmethod
    def always_present_metadata_columns(cls) -> set[str]:
        return {cls.source.value}
QueryEmbeddings = list[float | int] | None
SearchResult = Tuple[list[Document], QueryEmbeddings]

class RetrievalMode(StrEnum):
    SIMILARITY = 'similarity'
    MAXIMAL_MARGINAL_RELEVANCE = 'maximal_marginal_relevance'
DEFAULT_RETRIEVAL_MODE = RetrievalMode.SIMILARITY
DEFAULT_MAXIMAL_MARGINAL_RELEVANCE_LAMBDA = 0.5

class VectorStore(ABC):
    """Vector store base class for document retrieval."""

    @abstractmethod
    async def search(self, query: str, k: int=DEFAULT_K, filter: dict[str, Any] | None=DEFAULT_FILTER, add_neighbor_chunks: bool=DEFAULT_ADD_NEIGHBOR_CHUNKS, retrieval_mode: RetrievalMode=RetrievalMode.SIMILARITY, maximal_marginal_relevance_lambda: float=0.5, **kwargs: Any) -> SearchResult:
        """Return docs most similar to query asynchronously."""
        pass

    @abstractmethod
    async def add_neighbor_chunks(self, docs: list[Document], filter: dict[str, Any] | None=DEFAULT_FILTER, **kwargs: Any) -> list[Document]:
        """Return docs with additional neighbor chunks asynchronously."""
        pass

class MetadataFilterOperators(StrEnum):
    AND = '$and'
    OR = '$or'
    EQ = '$eq'
    NE = '$ne'
    GT = '$gt'
    GTE = '$gte'
    LT = '$lt'
    LTE = '$lte'
    IN = '$in'
    NIN = '$nin'
    CONTAINS = '$contains'
    NOT_CONTAINS = '$not_contains'

    @classmethod
    def all(cls) -> list[str]:
        return [op.value for op in cls]