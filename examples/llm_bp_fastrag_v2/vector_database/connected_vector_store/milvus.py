# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from __future__ import annotations
from typing import Any
from typing import Iterable
from typing import Sequence
from typing import cast
from grpc import RpcError
from grpc import StatusCode
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from langchain_milvus import Milvus as MilvusVectorStore
from loguru import logger
from pydantic import BaseModel
from pydantic import model_validator
from pymilvus import Hit
from pymilvus import MilvusException
from i18n import gettext
from vector_database.connected_vector_store.common import ConnectedVectorStoreEmbeddings
from vector_database.connected_vector_store.common import MilvusConnectionParams
from vector_database.dr_embeddings import Embeddings
from vector_database.inference.entities import MetadataFilterOperators
from vector_database.inference.retrieval import maximal_marginal_relevance
MILVUS_MAX_VARCHAR_SIZE = 65535

def translate_filter_to_milvus(filter: dict[str, Any] | None) -> str:

    def format_value(value: Any) -> str:
        """Format value for Milvus filter, handling booleans as lowercase."""
        if isinstance(value, bool):
            return 'true' if value else 'false'
        return repr(value)

    def build_comparison_filter(key: str, operator: str, operand: Any) -> str:
        operator_map = {MetadataFilterOperators.EQ: '==', MetadataFilterOperators.NE: '!=', MetadataFilterOperators.GT: '>', MetadataFilterOperators.GTE: '>=', MetadataFilterOperators.LT: '<', MetadataFilterOperators.LTE: '<=', MetadataFilterOperators.IN: 'in', MetadataFilterOperators.NIN: 'not in'}
        try:
            operator_str = operator_map[MetadataFilterOperators(operator)]
            return ' '.join((key, operator_str, format_value(operand)))
        except (KeyError, ValueError):
            raise ValueError(gettext('The metadata filter contains an unsupported operator. Valid operators: {supported_operators}').format(supported_operators='$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin'))
    if filter is None:
        return ''
    milvus_filter = []
    for key, value in filter.items():
        match key:
            case MetadataFilterOperators.AND:
                milvus_filter.append(' and '.join([translate_filter_to_milvus(f) for f in value]))
            case MetadataFilterOperators.OR:
                expression = ' or '.join([translate_filter_to_milvus(f) for f in value])
                milvus_filter.append(f'({expression})')
            case _:
                if isinstance(value, dict):
                    if len(value) != 1:
                        raise ValueError(gettext('Invalid filter dictionary structure.'))
                    operator, operand = tuple(value.items())[0]
                    milvus_filter.append(build_comparison_filter(key, operator, operand))
                else:
                    milvus_filter.append(f'{key} == {format_value(value)}')
    return ' and '.join(milvus_filter)

def filter_out_big_chunks(texts: list[str], embeddings: list[list[float]], metadatas: list[dict] | None=None) -> tuple[list[str], list[list[float]], list[dict] | None]:

    def value_too_big(value: Any) -> bool:
        if isinstance(value, str):
            return len(value.encode('utf-8')) > MILVUS_MAX_VARCHAR_SIZE
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
        if (filtered_metadata := {k: v for k, v in metadata.items() if not value_too_big(v)}):
            filtered_metadatas.append(filtered_metadata)
    return (filtered_texts, filtered_embeddings, filtered_metadatas)

class MilvusCredentials(BaseModel):
    """Credentials necessary to access Milvus vector database."""
    api_key: str | None
    username: str | None
    password: str | None

    @model_validator(mode='after')
    def validate_either_api_key_or_basic_credentials(self) -> MilvusCredentials:
        if self.api_key is None and (self.username is None or self.password is None):
            raise ValueError(gettext('Either `api_key` or `username` and `password` must be present in the credentials.'))
        return self

class DRMilvusVectorStore(MilvusVectorStore):

    async def aadd_texts(self, texts: Iterable[str], metadatas: list[dict] | None=None, ids: list[str] | None=None, **kwargs: Any) -> list[str]:  # type: ignore[override]
        texts = list(texts)
        if not texts:
            return []
        if self.embeddings:
            embeddings = await self.embeddings.aembed_documents(texts)
        text_embeddings = list(zip(texts, embeddings))
        return await self.aadd_embeddings(text_embeddings=text_embeddings, metadatas=metadatas, ids=ids, **kwargs)

    async def aadd_embeddings(self, text_embeddings: Iterable[tuple[str, list[float]]], metadatas: list[dict] | None=None, ids: list[str] | None=None, create_index_if_not_exists: bool=True, **kwargs: Any) -> list[str]:  # type: ignore[override]
        texts, embeddings = map(list, zip(*text_embeddings))
        texts, embeddings, metadatas = filter_out_big_chunks(texts, embeddings, metadatas)  # type: ignore[arg-type]
        return await super().aadd_embeddings(texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, batch_size=32, **kwargs)

    async def asimilarity_search_by_vector_with_scores_and_vectors(self, embedding: list[float], k: int=4, filter: dict[str, Any] | None=None) -> list[tuple[Document, float, list[float]]]:
        milvus_filter = translate_filter_to_milvus(filter)
        hits = await self.aclient.search(collection_name=self.collection_name, data=[embedding], filter=milvus_filter, limit=k, output_fields=['*'])
        return self._get_docs_scores_vectors_from_hits(hits[0])

    async def ammr_search_by_vector_with_scores_and_vectors(self, embedding: list[float], k: int=4, fetch_k: int=20, lambda_mult: float=0.5, filter: dict[str, Any] | None=None) -> list[tuple[Document, float, list[float]]]:
        milvus_filter = translate_filter_to_milvus(filter)
        hits = await self.aclient.search(collection_name=self.collection_name, data=[embedding], filter=milvus_filter, limit=fetch_k, output_fields=['*'])
        embeddings = [hit[self._vector_field] for hit in hits[0]]
        selected_indices = maximal_marginal_relevance(embedding, embeddings, lambda_mult=lambda_mult, k=k)
        selected_hits = [hits[0][i] for i in selected_indices]
        return self._get_docs_scores_vectors_from_hits(selected_hits)

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        documents = []
        for doc in await self.aclient.get(self.collection_name, ids):
            doc_id = doc.pop(self._primary_field)
            text = doc.pop(self._text_field, '')
            doc.pop(self._vector_field)
            doc.pop('source')
            documents.append(Document(id=doc_id, page_content=text, metadata=doc))
        return documents

    def _get_docs_scores_vectors_from_hits(self, hits: list[Hit]) -> list[tuple[Document, float, list[float]]]:
        docs_scores_vectors = []
        for hit in hits:
            fields = hit.fields
            fields.pop(self._primary_field)
            vector = fields.pop(self._vector_field)
            if not (text := fields.pop(self._text_field, '')):
                logger.warning('Found document with no text. Skipping.')
                continue
            docs_scores_vectors.append((Document(id=hit.pk, page_content=text, metadata=fields), hit.score, vector))
        return docs_scores_vectors

    @property
    def embeddings(self) -> LangChainEmbeddings | None:  # type: ignore[override]
        '\n        Get embedding function.\n\n        This property is used to narrow down return types to LangChainEmbeddings | None.\n        Otherwise, it creates a mess with mypy.\n        '
        return cast(LangChainEmbeddings | None, self.embedding_func)

def get_milvus_vector_store_from_credentials(credentials: MilvusCredentials, external_vector_database_connection: MilvusConnectionParams, index_name: str, embedder: Embeddings) -> DRMilvusVectorStore:
    token = credentials.api_key or f'{credentials.username}:{credentials.password}'  # type: ignore[union-attr]
    try:
        return DRMilvusVectorStore(enable_dynamic_field=True, collection_name=f'_{index_name}', embedding_function=ConnectedVectorStoreEmbeddings(embedder), connection_args={'uri': external_vector_database_connection.uri, 'token': token})
    except RpcError as e:
        if e.code() == StatusCode.UNAUTHENTICATED:
            if 'cluster does not exist' in e.details():
                raise ValueError('Failed to connect to external vector database')
            raise ValueError('Connected vector database credentials are not correct.')
        if e.code() == StatusCode.PERMISSION_DENIED:
            raise ValueError('You do not have permission to access the connected vector database.')
        raise
    except MilvusException as e:
        logger.bind(message=str(e)).warning('Failed to connect to external vector database')
        raise ValueError('Could not connect to the external vector database.')