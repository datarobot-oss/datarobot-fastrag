# Copyright 2023 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
import asyncio
import itertools
import json
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from typing import Any
from typing import Final
from typing import Mapping
from typing import Sequence
from typing import Type

import aiohttp
import backoff
import faiss
import numpy as np
import simsimd as simd
from aiohttp import ClientResponse
from aiohttp import ClientResponseError
from aiohttp import ClientSession
from aiohttp import ClientTimeout
from constants import DATAROBOT_IDENTITY_HEADER_NAME
from deployment_access import DataRobotDeploymentCredentials
from deployment_access import construct_headers
from deployment_access import construct_payload_from_dict
from deployment_access import parse_documents_from_response_data
from deployment_access import parse_query_embeddings_from_response_data
from i18n import gettext
from jinja2 import Environment
from langchain.schema import Document
from language_models.language_model_interface import LanguageModelInterface
from language_models.language_model_interface import LanguageModelInvalidRequestError
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionUserMessageParam
from sentence_transformers import quantize_embeddings
from vector_database.custom_model_embeddings import RetryableCustomModelError
from vector_database.custom_model_embeddings import get_factor
from vector_database.custom_model_embeddings import get_max_tries
from vector_database.dr_embeddings import Embeddings
from vector_database.inference.base_chunk_repository import BaseChunkRepository
from vector_database.inference.entities import DEFAULT_MAXIMAL_MARGINAL_RELEVANCE_LAMBDA
from vector_database.inference.entities import DEFAULT_RETRIEVAL_MODE
from vector_database.inference.entities import MetadataColumnNames
from vector_database.inference.entities import QueryEmbeddings
from vector_database.inference.entities import RetrievalMode
from vector_database.inference.entities import SearchResult
from vector_database.inference.entities import VectorStore
from vector_database.inference.prompts import CONVERSATIONAL_RETRIEVER_PROMPT
from vector_database.inference.prompts import MULTI_STEP_RETRIEVER_PROMPT

DEFAULT_K = 10
DEFAULT_FILTER = None
DEFAULT_ADD_NEIGHBOR_CHUNKS = False
RESCORE_MULTIPLIER = 10
MAX_SEARCH_K = 1000000
QUANTIZE_PRECISION: Final = "ubinary"


def _default_relevance_score_fn(score: float) -> float:
    """Return a relevance score based on the similarity score."""
    # We round to 2 digits because similarity scores are flaky starting from the 3rd digit
    return float(round(score, 2))


def maximal_marginal_relevance(
    query_embedding: np.ndarray | list[float],
    embedding_list: np.ndarray | list[list[float]],
    lambda_mult: float,
    k: int,
) -> list[int]:
    """Calculate maximal marginal relevance.

    Adapted from elasticsearch.helpers.vectorstore._utils and langchain_pinecone._utilities.

    Parameters
    ----------
    query_embedding
        The embedding of the query.
    embedding_list
        The embeddings of the retrieved documents.
    lambda_mult
        Lambda parameter for MMR. Between 0 and 1, inclusive.
        Smaller favors diversity, larger favors similarity.

    Returns
    -------
    idxs
        List of indexes in descending order of marginal relevance.
    """
    if min(k, len(embedding_list)) <= 0:
        return []
    embedding_list = np.array(embedding_list)
    query_embedding = np.array(query_embedding)
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices.

    Adapted from elasticsearch.helpers.vectorstore._utils and langchain_pinecone._utilities.

    Parameters
    ----------
    x
        The first array.
    y
        The second array.

    Returns
    -------
    z
        The row-wise cosine similarities of the two arrays.

    """
    if len(x) == 0 or len(y) == 0:
        return np.array([])
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    z = 1 - np.array(simd.cdist(x, y, metric="cosine"))
    return z


class FaissVectorStore(VectorStore):
    def __init__(
        self,
        index: faiss.Index,
        embedder: Embeddings,
        chunk_repository: BaseChunkRepository,
        relevance_score_fn: Callable = _default_relevance_score_fn,
    ):
        """Initialize the Faiss vector store.

        Parameters
        ----------
        index
            The faiss index
        embedder
            The embedder implementing aembed_query to generate embeddings
        chunk_repository
            The document storage implementing retrieve_text_chunks to retrieve documents
        relevance_score_fn
            The function to postprocess and or normalize similarity scores
        """
        self.index = index
        self.embedder = embedder
        self.chunk_repository = chunk_repository
        self.relevance_score_fn = relevance_score_fn
        self.logger = logger

    async def _search(
        self, query_embeddings: np.ndarray, search_k: int, params: faiss.SearchParameters | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger = self.logger.bind(num_embeddings=len(query_embeddings), search_k=search_k)
        logger.debug("Searching in the VDB")
        if isinstance(self.index, faiss.IndexBinaryFlat):
            rescore_embeddings = query_embeddings.copy()
            query_embeddings = quantize_embeddings(query_embeddings, precision=QUANTIZE_PRECISION)
            # retrieve more records for binary rescoring
            rescore_k = min(search_k * RESCORE_MULTIPLIER, self.index.ntotal)
            scores, indices = self.index.search(query_embeddings, rescore_k, params=params)
            # Adapted version from sentence-transformers
            # https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/quantization.py
            # we just want valid indices
            vectors = np.asarray(
                [self.index.reconstruct(i) for i in indices[indices != -1].tolist()]
            )
            top_k_embeddings = np.unpackbits(vectors, axis=-1).astype(int)
            rescored_scores = rescore_embeddings @ top_k_embeddings.T
            rescored_indices = np.argsort(-rescored_scores)[:, :search_k]
            indices = indices[0, rescored_indices]
            scores = rescored_scores[0, rescored_indices]
            vectors = vectors[rescored_indices]
        else:
            scores, indices, vectors = self.index.search_and_reconstruct(
                query_embeddings, search_k, params=params
            )
        logger.debug("Finished searching in the VDB")
        return (scores, indices, vectors, query_embeddings)

    async def _retrieve_and_filter_documents(
        self,
        scores: list[float],
        indices: list[int],
        vectors: list[QueryEmbeddings],
        filter: dict[str, Any] | None,
        add_neighbor_chunks: bool,
    ) -> list[Document]:
        self.logger.debug("Retrieving and filtering documents")
        # we cast indices to int because faiss returns np.int64
        # which is not JSON serializable by pydantic
        # faiss returns -1 if not enough indices are found
        ids_to_scores_and_vectors_map = {
            int(i): (self.relevance_score_fn(score), vector)
            for i, score, vector in zip(indices, scores, vectors)
            if i != -1
        }
        doc_ids = list(ids_to_scores_and_vectors_map.keys())
        if add_neighbor_chunks:
            doc_ids = [
                *dict.fromkeys(
                    [
                        j
                        for i in doc_ids
                        for j in range(i - 1, i + 2)
                        if j >= 0 and j <= self.index.ntotal - 1
                    ]
                )
            ]
        docs = await self.chunk_repository.retrieve_text_chunks(doc_ids)
        ids_to_consider = (
            (await self.chunk_repository.get_indices_matching_filters(filter)).tolist()
            if filter
            else None
        )
        filtered_docs = []
        for i, doc in zip(doc_ids, docs):
            if not ids_to_consider or i in ids_to_consider:
                # Neighbor documents do not have similarity scores or vectors
                score, vector = ids_to_scores_and_vectors_map.get(i, (None, None))
                doc.metadata[MetadataColumnNames.similarity_score.value] = score
                doc.metadata[MetadataColumnNames.doc_vector.value] = vector
                filtered_docs.append(doc)
        self.logger.debug("Finished retrieving and filtering documents")
        return filtered_docs

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

        License note: This function is a modified version of the langchain FAISS.asimilarity_search

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
            Retrieval mode to use, similarity or maximal marginal relevance (MMR).
        maximal_marginal_relevance_lambda
            Lambda parameter for MMR. Between 0 and 1, inclusive.
            Smaller favors diversity, larger favors similarity.

        Returns
        -------
            List of Documents most similar to the query and the query embeddings.
        """
        self.logger.debug("Running similarity search")
        self.logger.debug("Embedding the prompt")
        query_embeddings = await self.embedder.aembed_query(query)
        self.logger.debug("Finished embedding the prompt")
        # We need to increase dimensionality because we query just a single vector
        query_embeddings = np.expand_dims(query_embeddings, axis=0)
        top_k = k * 3 if add_neighbor_chunks else k
        params = None
        if filter:
            subset = await self.chunk_repository.get_indices_matching_filters(filter)
            if len(subset) == 0:
                return ([], query_embeddings[0].tolist())
            sel = faiss.IDSelectorBatch(subset)
            params = faiss.SearchParameters(sel=sel)
        if retrieval_mode == RetrievalMode.SIMILARITY:
            scores, indices, vectors, query_embeddings = await self._search(
                query_embeddings, k, params
            )
        elif retrieval_mode == RetrievalMode.MAXIMAL_MARGINAL_RELEVANCE:
            scores, indices, vectors, output_query_embeddings = await self._search(
                query_embeddings, 5 * k, params
            )
            embeddings = np.unpackbits(vectors, axis=-1).astype(int) * 2 - 1
            selected_indices = maximal_marginal_relevance(
                query_embeddings, embeddings[0], lambda_mult=maximal_marginal_relevance_lambda, k=k
            )
            query_embeddings = output_query_embeddings
            scores = scores[:, selected_indices]
            indices = indices[:, selected_indices]
            vectors = vectors[:, selected_indices]
        else:
            raise RuntimeError(f"Unknown retrieval_mode: {retrieval_mode}")
        documents = await self._retrieve_and_filter_documents(
            scores[0].tolist(),
            indices[0].tolist(),
            vectors[0].tolist(),
            filter,
            add_neighbor_chunks,
        )
        self.logger.debug("Finished running similarity search")
        return (documents[:top_k], query_embeddings[0].tolist())

    async def add_neighbor_chunks(
        self, docs: list[Document], filter: dict[str, Any] | None = DEFAULT_FILTER, **kwargs: Any
    ) -> list[Document]:
        scores = [doc.metadata[MetadataColumnNames.similarity_score.value] for doc in docs]
        indices = [doc.metadata[MetadataColumnNames.chunk_id.value] for doc in docs]
        vectors = [doc.metadata[MetadataColumnNames.doc_vector.value] for doc in docs]
        return await self._retrieve_and_filter_documents(
            scores, indices, vectors, filter, add_neighbor_chunks=True
        )


class ExternalVectorDatabaseRetrieveError(Exception):
    """Exception on retrieval from external vector database."""


class BaseCustomModelVectorDatabaseErrorHandler:
    async def handle(self) -> None:
        pass


@backoff.on_exception(
    backoff.expo,
    exception=RetryableCustomModelError,
    max_tries=get_max_tries,
    factor=get_factor,
    jitter=None,
)
async def _make_call_to_custom_vdb(
    session: ClientSession, prediction_api_url: str, payload: bytes
) -> ClientResponse:
    try:
        return await session.post(prediction_api_url, data=payload)
    except ClientResponseError as e:
        if e.status in [502, 503, 504]:
            raise RetryableCustomModelError() from e
        raise e


class CustomModelVectorDatabaseAsVectorStore(VectorStore):
    """Vector database implemented as deployed custom model."""

    def __init__(
        self,
        prediction_api_url: str,
        datarobot_key: str | None,
        authorization_header: str,
        model_type: str,
        input_type: str,
        prompt_column_name: str,
        target_column_name: str,
        association_id_column: str | None,
        error_handler: BaseCustomModelVectorDatabaseErrorHandler | None = None,
        identity_token_loader: Callable | None = None,
    ):
        self.prediction_api_url = prediction_api_url
        self.datarobot_key = datarobot_key
        self.authorization_header = authorization_header
        self.model_type = model_type
        self.input_type = input_type
        self.prompt_column_name = prompt_column_name
        self.target_column_name = target_column_name
        self.error_handler = error_handler
        self.association_id_column = association_id_column
        self.logger = logger
        self.identity_token_loader = identity_token_loader

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
        _ = k  # Ignored in custom models
        credentials = DataRobotDeploymentCredentials(
            datarobot_key=self.datarobot_key, authorization_header=self.authorization_header
        )
        query_embeddings = None
        headers = self.construct_headers(credentials)
        payload_dict = {
            self.prompt_column_name: query,
            "k": k,
            "metadata_filter": self.convert_metadata_filter_to_json(metadata_filter=filter),
            "add_neighbor_chunks": add_neighbor_chunks,
        }
        payload = construct_payload_from_dict(
            payload_dict, self.input_type, self.association_id_column
        )
        try:
            async with aiohttp.ClientSession(
                headers=headers, raise_for_status=True, timeout=ClientTimeout(total=60)
            ) as session:
                response = await _make_call_to_custom_vdb(session, self.prediction_api_url, payload)
                response_data = json.loads(await response.text())
        except Exception as e:
            if self.error_handler is not None:
                await self.error_handler.handle()
                return ([], query_embeddings)
            else:
                # This code runs inside custom models, where there is no diagnostic access to
                # pinpoint the exact reason of the error.
                self.logger.bind(prediction_api_url=self.prediction_api_url)
                error_message = repr(e)
                self.logger.error(f"Vector database request returned an error: {error_message}")
                msg = gettext(
                    "Vector database request returned an error. Try again, and if that does not resolve the issue, send the prompt to the associated LLM blueprint and a more detailed error will be returned in the playground."
                )
                # We don't want to log the original exception as it could contain PII
                raise ExternalVectorDatabaseRetrieveError(msg) from None
        # Validation confirmed this to be either string or list of strings
        docs = parse_documents_from_response_data(
            response_data, self.model_type, self.target_column_name
        )
        query_embeddings = parse_query_embeddings_from_response_data(response_data)
        return (docs, query_embeddings)

    def convert_metadata_filter_to_json(
        self, metadata_filter: dict[str, Any] | None = DEFAULT_FILTER
    ) -> dict[str, Any] | str | None:
        if metadata_filter:
            if self.input_type == "CSV":
                return json.dumps(metadata_filter)
            else:
                # we don't need to do anything if request will be JSON
                return metadata_filter
        else:
            return None

    async def add_neighbor_chunks(
        self, docs: list[Document], filter: dict[str, Any] | None = DEFAULT_FILTER, **kwargs: Any
    ) -> list[Document]:
        """Return documents with additional neighbor chunks asynchronously."""
        # Adding neighbor chunks separatetly from search is not supported for external VDBs
        # Make sure to call search with add_neighbor_chunks=True for adding neighbors.
        return docs

    def construct_headers(self, credentials: DataRobotDeploymentCredentials) -> dict[str, Any]:
        headers = construct_headers(self.model_type, credentials)
        if self.identity_token_loader:
            identity_token = self.identity_token_loader()
            if identity_token:
                headers.update({DATAROBOT_IDENTITY_HEADER_NAME: identity_token})
        return headers


DocsQueryResult = tuple[list[Document], str, Sequence[ChatCompletionMessageParam], QueryEmbeddings]


class BaseRetriever(ABC):
    """Base class for vector database retrievers."""

    prompt_template: str | None = None

    def __init__(
        self,
        vector_store: VectorStore,
        language_model: LanguageModelInterface,
        llm_settings: dict,
        number_of_documents_to_retrieve: int,
        add_neighbor_chunks: bool = False,
        prompt_template: str | None = None,
        retrieval_mode: RetrievalMode = DEFAULT_RETRIEVAL_MODE,
        maximal_marginal_relevance_lambda: float = DEFAULT_MAXIMAL_MARGINAL_RELEVANCE_LAMBDA,
    ):
        self.vector_store = vector_store
        self.language_model = language_model
        self.llm_settings = llm_settings
        self.number_of_documents_to_retrieve = number_of_documents_to_retrieve
        self.add_neighbor_chunks = add_neighbor_chunks
        self.prompt_template = prompt_template or self.prompt_template
        self.retrieval_mode = retrieval_mode
        self.maximal_marginal_relevance_lambda = maximal_marginal_relevance_lambda

    @abstractmethod
    async def _get_docs_and_query(
        self,
        query: str,
        history: Sequence[ChatCompletionMessageParam],
        metadata_filter: dict[str, Any] | None = None,
    ) -> DocsQueryResult:
        pass

    async def get_docs_and_query(
        self,
        query: str,
        history: Sequence[ChatCompletionMessageParam],
        metadata_filter: dict[str, Any] | None = None,
    ) -> DocsQueryResult:
        """Retrieve documents from the vector database and optionally rewrite the user query.

        Parameters
        ----------
        query
            The user query
        history
            Chat history messages
        metadata_filter
            The metadata filter to apply to the retrieved documents

        Returns
        -------
            The documents most similar to the user query and the rewrittern query
        """
        top_k = (
            self.number_of_documents_to_retrieve * 3
            if self.add_neighbor_chunks
            else self.number_of_documents_to_retrieve
        )
        slice = 2
        while True:
            try:
                documents, query, history, query_embeddings = await self._get_docs_and_query(
                    query, history, metadata_filter
                )
                break
            except LanguageModelInvalidRequestError:
                # we are not pruning history upfront but instead retry with growing reduced history
                if history:
                    history = history[slice:]
                    slice *= 2
                else:
                    raise
        return (documents[:top_k], query, history, query_embeddings)


class SingleLookUpRetriever(BaseRetriever):
    """Perform a single lookup in the vector database and retrieve the most similar documents."""

    async def _get_docs_and_query(
        self,
        query: str,
        history: Sequence[ChatCompletionMessageParam],
        metadata_filter: dict[str, Any] | None = None,
    ) -> DocsQueryResult:
        documents, query_emb = await self.vector_store.search(
            add_neighbor_chunks=self.add_neighbor_chunks,
            query=query,
            k=self.number_of_documents_to_retrieve,
            filter=metadata_filter,
            retrieval_mode=self.retrieval_mode,
            maximal_marginal_relevance_lambda=self.maximal_marginal_relevance_lambda,
        )
        return (documents, query, history, query_emb)


class ConversationalRetriever(BaseRetriever):
    """Rewrite user query in context of chat history before retrieving relevant documents."""

    prompt_template = CONVERSATIONAL_RETRIEVER_PROMPT

    async def _get_docs_and_query(
        self,
        query: str,
        history: Sequence[ChatCompletionMessageParam],
        metadata_filter: dict[str, Any] | None = None,
    ) -> DocsQueryResult:
        context = get_context_from_history(history)
        prompt = self.prompt_template.format(query=query, context=context)
        # Do not use the history because the rewritten query contains it in the context
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content=prompt)
        ]
        language_model_result = await self.language_model.submit_prompt(
            messages=messages, llm_settings=self.llm_settings
        )
        new_history: Sequence[ChatCompletionMessageParam] = []
        if language_model_result.result_text is None:
            # The content filter guards of the LLM has been triggered, so there is no response.
            # Thus, we default to the single lookup retriever here
            single_look_up_retriever = SingleLookUpRetriever(
                vector_store=self.vector_store,
                language_model=self.language_model,
                llm_settings=self.llm_settings,
                number_of_documents_to_retrieve=self.number_of_documents_to_retrieve,
                add_neighbor_chunks=self.add_neighbor_chunks,
                retrieval_mode=self.retrieval_mode,
                maximal_marginal_relevance_lambda=self.maximal_marginal_relevance_lambda,
            )
            (
                documents,
                query,
                new_history,
                query_emb,
            ) = await single_look_up_retriever._get_docs_and_query(
                query=query, history=history, metadata_filter=metadata_filter
            )
        else:
            documents, query_emb = await self.vector_store.search(
                query=language_model_result.result_text,
                k=self.number_of_documents_to_retrieve,
                add_neighbor_chunks=self.add_neighbor_chunks,
                filter=metadata_filter,
                retrieval_mode=self.retrieval_mode,
                maximal_marginal_relevance_lambda=self.maximal_marginal_relevance_lambda,
            )
            query = language_model_result.result_text
        return (documents, query, new_history, query_emb)


class MultiStepRetriever(BaseRetriever):
    """Rewrite user query and perform multiple lookups before retrieving relevant documents.

    It retrieves documents in two stages. The first stage is a conversational retrieval
    and the second stage generates multiple search queries based on the first stage results,
    zip merges and deduplicates the second stage results and returns the k most relevant documents.
    """

    prompt_template = MULTI_STEP_RETRIEVER_PROMPT

    async def _get_docs_and_query(
        self,
        query: str,
        history: Sequence[ChatCompletionMessageParam],
        metadata_filter: dict[str, Any] | None = None,
    ) -> DocsQueryResult:
        conversational_retriever = ConversationalRetriever(
            vector_store=self.vector_store,
            language_model=self.language_model,
            llm_settings=self.llm_settings,
            number_of_documents_to_retrieve=self.number_of_documents_to_retrieve,
            add_neighbor_chunks=False,
            retrieval_mode=self.retrieval_mode,
            maximal_marginal_relevance_lambda=self.maximal_marginal_relevance_lambda,
        )
        (
            documents,
            rewritten_query,
            new_history,
            query_emb,
        ) = await conversational_retriever._get_docs_and_query(
            query=query, history=history, metadata_filter=metadata_filter
        )
        context = get_context_from_documents(documents)
        prompt = self.prompt_template.format(query=query, context=context)
        messages = list(new_history) + [ChatCompletionUserMessageParam(role="user", content=prompt)]
        language_model_result = await self.language_model.submit_prompt(
            messages=messages, llm_settings=self.llm_settings
        )
        if language_model_result.result_text:
            can_add_neighbors_separately = isinstance(self.vector_store, FaissVectorStore)
            add_neighbors_in_search = self.add_neighbor_chunks and (
                not can_add_neighbors_separately
            )
            add_neighbors_after_search = self.add_neighbor_chunks and can_add_neighbors_separately
            search_queries = language_model_result.result_text.split("\n")
            tasks = [
                self.vector_store.search(
                    query=query,
                    k=self.number_of_documents_to_retrieve,
                    add_neighbor_chunks=add_neighbors_in_search,
                    filter=metadata_filter,
                    retrieval_mode=self.retrieval_mode,
                    maximal_marginal_relevance_lambda=self.maximal_marginal_relevance_lambda,
                )
                for query in search_queries
                if len(query) > 0
            ]
            _search_results = await asyncio.gather(*tasks)
            search_results = [v[0] for v in _search_results]
            final_documents = deduplicate_documents(zip_merge_lists(search_results))
            if add_neighbors_after_search:
                final_documents = await self.vector_store.add_neighbor_chunks(
                    final_documents, metadata_filter, apply_access_control_list=True
                )
        else:
            # Revert to ConversationalRetriever pattern if result_text is None,
            # which means that the content filters from the LLM Provider
            # got triggered midway through.
            final_documents = documents
        return (final_documents, rewritten_query, new_history, query_emb)


DEFAULT_RETRIEVER_NAME = "SINGLE_LOOKUP_RETRIEVER"
VECTOR_DATABASE_RETRIEVER_MAP: Mapping[str, Type[BaseRetriever]] = {
    "SINGLE_LOOKUP_RETRIEVER": SingleLookUpRetriever,
    "CONVERSATIONAL_RETRIEVER": ConversationalRetriever,
    "MULTI_STEP_RETRIEVER": MultiStepRetriever,
}


def get_retriever(
    retriever_name: str | None,
    vector_store: VectorStore,
    language_model: LanguageModelInterface | Any,
    llm_settings: dict,
    number_of_documents_to_retrieve: int,
    add_neighbor_chunks: bool,
    retrieval_mode: RetrievalMode = DEFAULT_RETRIEVAL_MODE,
    maximal_marginal_relevance_lambda: float = DEFAULT_MAXIMAL_MARGINAL_RELEVANCE_LAMBDA,
) -> BaseRetriever:
    retriever_name = retriever_name or DEFAULT_RETRIEVER_NAME
    retriever_class = VECTOR_DATABASE_RETRIEVER_MAP[retriever_name]
    if language_model is None and retriever_name != DEFAULT_RETRIEVER_NAME:
        raise ValueError(gettext("LLM is required for advanced retrievers."))
    # Remove tools from the language model settings
    llm_settings = dict(llm_settings)
    retriever = retriever_class(
        vector_store=vector_store,
        language_model=language_model,
        llm_settings=llm_settings,
        number_of_documents_to_retrieve=number_of_documents_to_retrieve,
        add_neighbor_chunks=add_neighbor_chunks,
        retrieval_mode=retrieval_mode,
        maximal_marginal_relevance_lambda=maximal_marginal_relevance_lambda,
    )
    return retriever


# This template deliberately ignores tool call and response messages
# as they are not considered important for retrieval context.
CONTEXT_HISTORY_TEMPLATE = '\n{% for message in messages %}\n{% if message["role"] == "user" %}\n\'user\': {{ message["content"] }}\n{% elif message["role"] == "assistant" and message.get("content") %}\n\'assistant\': {{ message["content"] }}\n{% endif %}\n{% endfor %}\n'


def get_context_from_history(messages: Sequence[ChatCompletionMessageParam]) -> str:
    """Extract the context from the chat history's in a single query."""
    jinja_env = Environment(trim_blocks=True)
    template = jinja_env.from_string(CONTEXT_HISTORY_TEMPLATE)
    result = template.render(messages=messages)
    return result


def get_context_from_documents(documents: list[Document]) -> str:
    """Extract the context from the retrieved documents."""
    return "".join((f"'doc{i}': {doc.page_content} \n" for i, doc in enumerate(documents)))


def zip_merge_lists(lists: list[list[Any]]) -> list[Any]:
    """Zip merge lists of lists into a single list."""
    return list(itertools.chain.from_iterable(zip(*lists)))


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """Deduplicate a list of langchain Documents based on the chunk id."""
    use_chunk_id = all((MetadataColumnNames.chunk_id.value in doc.metadata for doc in documents))
    if use_chunk_id:
        unique_docs = {doc.metadata[MetadataColumnNames.chunk_id.value]: doc for doc in documents}
    else:
        unique_docs = {doc.page_content: doc for doc in documents}
    return list(unique_docs.values())
