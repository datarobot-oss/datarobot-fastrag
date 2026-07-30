# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
import json
import os
import typing
from collections.abc import AsyncIterator
from collections.abc import Iterable
from collections.abc import Sequence
from typing import Any
import faiss
import pandas as pd
from fastrag import RuntimeParameters
from langchain_core.documents import Document
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionContentPartParam
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat import CompletionCreateParams
from sentence_transformers import SentenceTransformer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from custom_model_chat import generate_chat_completion
from custom_model_chat import get_language_model_citations_from_docs
from custom_model_chunk_repository import get_custom_model_chunk_repository
from custom_model_credentials import get_datarobot_authorization_header
from custom_model_entities import CustomModelLLMBlueprintConfig
from custom_model_entities import ExternalDeploymentParameters
from custom_model_enum import CustomModelNonEnglishEmbeddingModelPaths
from custom_model_enum import CustomModelRuntimeParameters
from custom_model_enum import VectorDatabaseCustomModelObjects
from custom_model_identity import get_identity_token
from deployment_access import DataRobotDeploymentCredentials
from language_models.helpers import DEFAULT_CONTEXT_SIZE
from language_models.helpers import get_excess_token_budget
from language_models.helpers import prune_docs
from language_models.language_model_interface import BaseLanguageModelResult
from language_models.language_model_interface import BaseProviderGuardsMetadata
from language_models.language_model_interface import LanguageModel
from language_models.language_model_interface import LLMBlueprintModel
from vector_database.connected_vector_store.vector_store import ConnectedVectorStore
from vector_database.custom_model_embeddings import CustomModelEmbeddingClient
from vector_database.dr_embeddings import get_embedder
from vector_database.dr_embeddings import get_sentence_transformer
from vector_database.inference.entities import MetadataColumnNames
from vector_database.inference.entities import MetadataFilterOperators
from vector_database.inference.entities import QueryEmbeddings
from vector_database.inference.retrieval import BaseRetriever
from vector_database.inference.retrieval import CustomModelVectorDatabaseAsVectorStore
from vector_database.inference.retrieval import FaissVectorStore
from vector_database.inference.retrieval import VectorStore
from vector_database.inference.retrieval import get_retriever
LLM_PROVIDER_GUARD_TRIGGERED_COL_NAME = 'LLM_PROVIDER_GUARD_TRIGGERED'
DATAROBOT_CONFIGURED_ON_PREM_ST_SAAS_URL = 'http://datarobot-nginx/api/v2'
DATAROBOT_ACTUAL_ON_PREM_ST_SAAS_URL = 'http://datarobot-prediction-server:80/predApi/v1.0/'
METADATA_FILTER_COLUMN_NAME = 'metadata_filter'

def get_external_embedding_client(external_embedding_model_parameters: ExternalDeploymentParameters) -> CustomModelEmbeddingClient:
    if not external_embedding_model_parameters.prompt_column_name:
        raise RuntimeError('Prompt column name is not set in the custom model embedding client.')
    if not external_embedding_model_parameters.target_column_name:
        raise RuntimeError('Target column name is not set in the custom model embedding client.')
    credentials = DataRobotDeploymentCredentials(datarobot_key=external_embedding_model_parameters.datarobot_key, authorization_header=get_datarobot_authorization_header())
    return CustomModelEmbeddingClient(credentials=credentials, prediction_api_url=external_embedding_model_parameters.prediction_api_url, model_type=external_embedding_model_parameters.model_type, input_type=external_embedding_model_parameters.input_type, prompt_column_name=external_embedding_model_parameters.prompt_column_name, target_column_name=external_embedding_model_parameters.target_column_name)

def read_faiss_index() -> faiss.Index:
    """Read both binary and non-binary faiss indexes.
    If the index is larger than 1GB, it is loaded memory-mapped.
    """
    index_size = os.path.getsize(VectorDatabaseCustomModelObjects.FAISS_INDEX)
    read_flag = faiss.IO_FLAG_MMAP if index_size > 1000000000 else 0  # 0 is the default flag
    try:
        index = faiss.read_index_binary(VectorDatabaseCustomModelObjects.FAISS_INDEX, read_flag)
    except RuntimeError:
        index = faiss.read_index(VectorDatabaseCustomModelObjects.FAISS_INDEX, read_flag)
    return index

def load_stored_vector_database_as_vector_store(config: CustomModelLLMBlueprintConfig) -> VectorStore | None:
    """Load stored vector database as vector store
    using VectorDatabaseCustomModelObjects.

    Parameters
    ----------
    config
        An instance CustomModelLLMBlueprintConfig

    Returns
    -------
        A loaded vector store, None if no vector database present.
    """
    # The DEVICE_FOR_NEURAL_NETWORK_COMPUTATIONS environment variable is used during the invocation
    # of  `get_sentence_transformer`. If it is not present, an exception will be raised.
    os.environ['DEVICE_FOR_NEURAL_NETWORK_COMPUTATIONS'] = RuntimeParameters.get(CustomModelRuntimeParameters.DEVICE_FOR_NEURAL_NETWORK_COMPUTATIONS.value)
    if config.vector_database_settings is None:
        return None
    vector_store: VectorStore
    if config.vector_database_settings.external_vector_database_parameters is None:
        if config.vector_database_settings.external_embedding_model_parameters:
            client: SentenceTransformer | CustomModelEmbeddingClient = get_external_embedding_client(config.vector_database_settings.external_embedding_model_parameters)
        else:
            client = get_sentence_transformer(VectorDatabaseCustomModelObjects.EMBEDDING_MODEL.value)
        embedder = get_embedder(client)
        index = read_faiss_index()
        chunk_repository = get_custom_model_chunk_repository()
        vector_store = FaissVectorStore(index, embedder, chunk_repository)
    else:
        external_params = config.vector_database_settings.external_vector_database_parameters
        if not external_params.prompt_column_name:
            raise RuntimeError('Prompt column name is not set for the custom model vector database.')
        if not external_params.target_column_name:
            raise RuntimeError('Target column name is not set for the custom model vector database.')
        prediction_api_url = _handle_prediction_api_url_substitutions(external_params)
        vector_store = CustomModelVectorDatabaseAsVectorStore(prediction_api_url=prediction_api_url, datarobot_key=external_params.datarobot_key, authorization_header=get_datarobot_authorization_header(), model_type=external_params.model_type, input_type=external_params.input_type, prompt_column_name=external_params.prompt_column_name, target_column_name=external_params.target_column_name, association_id_column=external_params.association_id_column, identity_token_loader=get_identity_token)
    return vector_store

def _handle_prediction_api_url_substitutions(external_params: ExternalDeploymentParameters | None) -> str:
    pred_api_url_overridden_deployment_id = _substitute_vdb_deployment_runtime_parameter(external_params)
    final_prediction_api_url = _substitute_prediction_api_url_for_nginx(pred_api_url_overridden_deployment_id)
    return final_prediction_api_url

def _substitute_vdb_deployment_runtime_parameter(external_params: ExternalDeploymentParameters | None) -> str:
    vector_database_deployment_id_param = RuntimeParameters.get(CustomModelRuntimeParameters.VECTOR_DATABASE_DEPLOYMENT_ID.value)
    if not vector_database_deployment_id_param:
        return external_params.prediction_api_url if external_params else ''
    prediction_api_components = external_params.prediction_api_url.split('/') if external_params else []
    prediction_api_url = '/'.join(prediction_api_components[:-2]) + '/' + vector_database_deployment_id_param + '/' + '/'.join(prediction_api_components[-1:])
    return prediction_api_url

def _substitute_prediction_api_url_for_nginx(prediction_api_url: str) -> str:
    prediction_api_components = prediction_api_url.split('/')
    os_datarobot_endpoint = os.environ.get('DATAROBOT_ENDPOINT')
    if os_datarobot_endpoint and os_datarobot_endpoint.startswith(DATAROBOT_CONFIGURED_ON_PREM_ST_SAAS_URL):
        return DATAROBOT_ACTUAL_ON_PREM_ST_SAAS_URL + '/'.join(prediction_api_components[-3:])
    return prediction_api_url

def get_excess_token_budget_for_custom_model_llm_blueprint_config(config: CustomModelLLMBlueprintConfig, user_prompt: str) -> int:
    """Get the overall excess token budget after accounting for the system prompt, user prompt,
    and output tokens.

    Parameters
    ----------
    llm_blueprint
        The LLMBlueprint used to be submit this comparison prompt
    user_prompt
        The prompt text from the ComparisonPrompt

    Returns
    -------
    excess_token_budget
        The total excess token budget
    """
    # Custom LLMs may set their context size in a LLM setting
    # so that they don't default to DEFAULT_CONTEXT_SIZE, which may be too low for certain LLMs.
    # This is not passed to the custom LLM, but it does affect context pruning of documents.
    # See the LLMSetting description to understand the implications of updating this setting.
    context_size = config.llm_settings.get('external_llm_context_size', config.context_size) or DEFAULT_CONTEXT_SIZE
    system_prompt = config.llm_settings.get('system_prompt')
    output_tokens = config.llm_settings.get('max_completion_length')
    return get_excess_token_budget(context_size, output_tokens, system_prompt, user_prompt)

def create_retriever_for_llm_blueprint(model: LLMBlueprintModel, config: CustomModelLLMBlueprintConfig) -> BaseRetriever | None:
    """
    Create a vector database retriever if the LLM blueprint requires it.

    Parameters
    ----------
    model
        A LanguageModel and optional VectorStore used by the LLM blueprint.
    config
        The configuration of the LLM blueprint.

    Returns
    -------
    An instance of BaseRetriever. None if the LLM blueprint does not use a vector database.
    """
    retriever = None
    if model.vector_store and config.vector_database_settings:
        retriever = get_retriever(retriever_name=config.vector_database_settings.retriever, vector_store=model.vector_store, language_model=model.language_model, llm_settings=config.llm_settings, number_of_documents_to_retrieve=config.vector_database_settings.number_of_documents_to_retrieve, add_neighbor_chunks=config.vector_database_settings.add_neighbor_chunks)
    return retriever

async def retrieve_and_prune_docs(query: str | Iterable[ChatCompletionContentPartParam], history: Sequence[ChatCompletionMessageParam], config: CustomModelLLMBlueprintConfig, retriever: BaseRetriever | None=None, metadata_filter: dict[str, Any] | None=None) -> tuple[list[Document], str | Iterable[ChatCompletionContentPartParam], list[ChatCompletionMessageParam], QueryEmbeddings]:
    """
    Retrieve the documents from the vector database and prune them to fit in the LLM's context
    window.

    Parameters
    ----------
    query
        The content of the user message.
    history
        Previous history in the
    config
        The configuration of the LLM blueprint.
    retriever
        An instance of BaseRetriever.
    metadata_filter
        The metadata filter for filtering the documents during retrieval.

    Returns
    -------
    tuple
        A pruned list of documents retrieved from the vector database.
        The final version of the query (possibly rewritten by the retrieval algorithm).
        The final version of the chat history (possibly rewritten by the retrieval algorithm).
        The embedding vector of the query.
    """
    if not retriever:
        return ([], query, [m for m in history], None)
    docs, rewritten_query, rewritten_history, query_emb = await retriever.get_docs_and_query(query=str(query), history=history, metadata_filter=metadata_filter)
    docs_token_budget = get_excess_token_budget_for_custom_model_llm_blueprint_config(config=config, user_prompt=rewritten_query)
    tokenizer = get_non_english_tokenizer_from_retriever(retriever)
    docs, _ = prune_docs(docs=docs, docs_token_budget=docs_token_budget, tokenizer=tokenizer)
    return (docs, rewritten_query, [m for m in rewritten_history], query_emb)

@typing.no_type_check
def get_non_english_tokenizer_from_retriever(retriever: BaseRetriever | None) -> PreTrainedTokenizerBase | None:
    """Safely get non-english tokenizers from retriever depending on the vector store type."""

    def _is_non_english(tokenizer: PreTrainedTokenizerBase) -> bool:
        return any((str(model) in tokenizer.name_or_path for model in CustomModelNonEnglishEmbeddingModelPaths))
    try:
        if isinstance(retriever.vector_store, FaissVectorStore):
            tokenizer = retriever.vector_store.embedder.client.tokenizer
            return tokenizer if _is_non_english(tokenizer) else None
        elif isinstance(retriever.vector_store, ConnectedVectorStore):
            tokenizer = retriever.vector_store.vector_store.embeddings.client.tokenizer
            return tokenizer if _is_non_english(tokenizer) else None
        else:
            return None
    except AttributeError:
        return None

async def submit_llm_blueprint_prompt(model: LanguageModel, query: str, config: CustomModelLLMBlueprintConfig, retriever: BaseRetriever | None=None, metadata_filter: dict[str, Any] | None=None) -> tuple[BaseLanguageModelResult, list[Document], QueryEmbeddings]:
    """
    Submit a new LLM prompt to the LLM blueprint.

    Parameters
    ----------
    model
        An instance of LanguageModel used by the LLM blueprint.
    query
        The text of the query (user prompt).
    config
        The configuration of the LLM blueprint used in this custom model.
    retriever
        (Optional) An instance of a vector database retriever
        (if the LLM blueprint uses a vector database).
    metadata_filter
        (Optional) The metadata filter for vector database retrieval.

    Returns
    -------
    The response of the LLM blueprint, along with the documents retrieved from the vector database
    (if using a vector database).
    """
    # When using the `score` hook, history is embedded in the prompt string itself either by the
    # `CustomModelLLM` client or by the user. We cannot reliably detect the history in the
    # prompt and parse it out.
    docs, rewritten_query, _, query_emb = await retrieve_and_prune_docs(query=query, history=[], config=config, retriever=retriever, metadata_filter=metadata_filter)
    messages: list[ChatCompletionMessageParam] = [ChatCompletionUserMessageParam(role='user', content=rewritten_query)]
    language_model_result = await model.submit_prompt(messages, config.llm_settings, docs=docs)
    return (language_model_result, docs, query_emb)

async def generate_llm_blueprint_chat_completion(model: LanguageModel, config: CustomModelLLMBlueprintConfig, completion_create_params: CompletionCreateParams, retriever: BaseRetriever | None=None, metadata_filter: dict[str, Any] | None=None) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
    """
    Generate the LLM blueprint's completion for the specified chat conversation.

    Parameters
    ----------
    model
        An instance of LanguageModel used by the LLM blueprint.
    config
        The configuration of the LLM blueprint.
    completion_create_params
        The parameters of the chat completion request.
    retriever
        (Optional) An instance of a vector database retriever.
    metadata_filter
        (Optional) The metadata filter for vector database retrieval.

    Returns
    -------
    For non-streaming responses, returns a ChatCompletion.
    For streaming responses, returns an iterator over ChatCompletionChunk objects.
    """
    messages = [m for m in completion_create_params['messages']]
    user_messages = [m for m in messages if m['role'] == 'user']
    if not user_messages:
        raise ValueError('User message not found in the chat completion request')
    # Collect all history before the last user message.
    next_user_prompt = user_messages[-1]
    next_user_prompt_index = messages.index(next_user_prompt)
    history_before_next_user_prompt = messages[:next_user_prompt_index]
    docs, rewritten_query, rewritten_history, query_emb = await retrieve_and_prune_docs(query=next_user_prompt['content'], history=history_before_next_user_prompt, config=config, retriever=retriever, metadata_filter=metadata_filter)
    # Recompose the messages using the rewritten history and the rewritten user prompt
    rewritten_messages = rewritten_history[:]
    next_user_prompt['content'] = rewritten_query
    rewritten_messages.append(next_user_prompt)
    rewritten_messages.extend(messages[next_user_prompt_index + 1:])
    completion_create_params['messages'] = rewritten_messages
    result = await generate_chat_completion(llm=model, config=config, completion_create_params=completion_create_params, docs=docs, prompt_vector=query_emb)
    return result

def parse_metadata_filter(data: pd.DataFrame, idx: int, config: CustomModelLLMBlueprintConfig) -> dict[str, str] | None:
    """Parse and validate the metadata filter for a specific row in the prediction dataframe."""
    if not config.vector_database_settings:
        return None
    if METADATA_FILTER_COLUMN_NAME not in data.columns:
        return None
    filter_str = data.iloc[idx][METADATA_FILTER_COLUMN_NAME]
    if not filter_str:
        return None
    if not isinstance(filter_str, str):
        if pd.isna(filter_str):
            return None
        else:
            raise ValueError(f'Invalid JSON for metadata filter in row {idx}')
    try:
        filter_dict = json.loads(filter_str)
    except json.JSONDecodeError:
        raise ValueError(f'Invalid JSON for metadata filter in row {idx}')
    if not isinstance(filter_dict, dict):
        raise ValueError(f'Metadata filter in row {idx} must be a dictionary')
    # For BYO VDB skip enabled citation validation
    if config.vector_database_settings.external_vector_database_parameters:
        return filter_dict
    allowed_keys = set(config.vector_database_settings.enabled_citations_fields + MetadataFilterOperators.all())
    for k in filter_dict:
        if k not in allowed_keys:
            raise ValueError(f"Metadata filter in row {idx} uses disallowed column or operator '{k}'")
    return filter_dict

async def llm_blueprint_score(data: pd.DataFrame, model: LLMBlueprintModel, config: CustomModelLLMBlueprintConfig) -> tuple[list[str | None], list[list[Document]], list[list[BaseProviderGuardsMetadata] | None], list[QueryEmbeddings]]:
    """
    Score the specified dataframe using the LLM blueprint in this custom model.

    Parameters
    ----------
    data
        The dataframe to score.
    LLMBlueprintModel
        A LanguageModel and optional VectorStore used by the LLM blueprint.
    config
        The configuration of the LLM blueprint.

    Returns
    -------
    A list of LLM responses.
    A list of documents retrieved from the vector database for each response.
    A list of metadata describing the LLM provider guards triggered by each prompt or response.
    A list of query embeddings, one for each response.
    """
    docs_list, result_list, provider_llm_guards_list, query_emb_list = ([], [], [], [])
    prompt_column_name = RuntimeParameters.get(CustomModelRuntimeParameters.PROMPT_COLUMN_NAME.value)
    retriever = create_retriever_for_llm_blueprint(model, config)
    for idx, query in enumerate(data[prompt_column_name].tolist()):
        metadata_filter = parse_metadata_filter(data, idx, config)
        language_model_result, docs, query_emb = await submit_llm_blueprint_prompt(model.language_model, query, config, retriever, metadata_filter)
        result_list.append(language_model_result.result_text)
        provider_llm_guards_list.append(language_model_result.result_metadata.provider_llm_guards)
        docs_list.append(docs)
        query_emb_list.append(query_emb)
    return (result_list, docs_list, provider_llm_guards_list, query_emb_list)

async def llm_blueprint_chat(model: LLMBlueprintModel, config: CustomModelLLMBlueprintConfig, completion_create_params: CompletionCreateParams) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
    """
    Generate the LLM blueprint's completion for the specified chat conversation.

    Parameters
    ----------
    model
        A LanguageModel and optional VectorStore used by the LLM blueprint.
    config
        The configuration of the LLM blueprint.
    completion_create_params
        The parameters of the chat completion request.

    Returns
    -------
    For non-streaming responses, returns a ChatCompletion.
    For streaming responses, returns an iterator over ChatCompletionChunk objects.
    """
    retriever = create_retriever_for_llm_blueprint(model, config)
    metadata_filter: dict[str, Any] | None = None
    metadata_filter_in_request = completion_create_params.pop(METADATA_FILTER_COLUMN_NAME, None)  # type: ignore[typeddict-item, misc]
    if not isinstance(metadata_filter_in_request, (dict, type(None))):
        raise ValueError('Metadata filter must be a dictionary.')
    metadata_filter = metadata_filter_in_request
    return await generate_llm_blueprint_chat_completion(model=model.language_model, config=config, completion_create_params=completion_create_params, retriever=retriever, metadata_filter=metadata_filter)

def populate_provider_llm_guard_columns(provider_llm_guards_list: list[list[BaseProviderGuardsMetadata] | None], full_result_dict: dict[str, list]) -> None:
    unique_metrics = set()
    # Collect all unique metric names
    for guards_list in provider_llm_guards_list:
        if guards_list is not None:
            for guard in guards_list:
                unique_metrics.add((guard.stage, guard.name))
    # Initialize columns in full_result_dict for each unique metric
    for metric in unique_metrics:
        full_result_dict[f'LLM_PROVIDER_{metric[0]}_GUARD_{metric[1]}_TRIGGERED'] = []
        full_result_dict[f'LLM_PROVIDER_{metric[0]}_GUARD_{metric[1]}_VALUE'] = []
    full_result_dict[LLM_PROVIDER_GUARD_TRIGGERED_COL_NAME] = []
    # Loop row by row
    for guards_list in provider_llm_guards_list:
        overall_guard_triggered = False
        # Populate a row of all the unique metrics and
        # guarantee only going through O(n) per row where n is the length of the
        # unique_metrics to populate all unique metric columns
        if guards_list is not None:
            row_unique_metric = set()
            for guard in guards_list:
                triggered_col = f'LLM_PROVIDER_{guard.stage}_GUARD_{guard.name}_TRIGGERED'
                value_col = f'LLM_PROVIDER_{guard.stage}_GUARD_{guard.name}_VALUE'
                full_result_dict[triggered_col].append(guard.satisfy_criteria)
                full_result_dict[value_col].append(guard.value)
                row_unique_metric.add((guard.stage, guard.name))
                if not overall_guard_triggered and guard.satisfy_criteria:
                    overall_guard_triggered = True
            # unique_metrics contains the entire list of things, row_unique_metric might
            # contain less than the unique_metrics, and thus not handling the difference
            # with unique_metrics will risk not having the appropriate length list in the
            # full_result_dict.
            left_over_unique_metrics = unique_metrics.difference(row_unique_metric)
        else:
            # Everything is None
            left_over_unique_metrics = unique_metrics
        # Populate an empty value for each of the required column for the left
        # over metrics
        for metric in left_over_unique_metrics:
            triggered_col = f'LLM_PROVIDER_{metric[0]}_GUARD_{metric[1]}_TRIGGERED'
            value_col = f'LLM_PROVIDER_{metric[0]}_GUARD_{metric[1]}_VALUE'
            full_result_dict[triggered_col].append(None)
            full_result_dict[value_col].append(None)
        # Populate the global guard triggered col name
        full_result_dict[LLM_PROVIDER_GUARD_TRIGGERED_COL_NAME].append(overall_guard_triggered)

def populate_monitoring_columns(full_result_dict: dict[str, list], docs_list: list[list[Document]], query_embeddings: list[QueryEmbeddings], config: CustomModelLLMBlueprintConfig) -> None:
    """Create columns for production LLM monitoring."""

    def _pack(docs: list[Document]) -> list[dict[str, Any]]:
        return get_language_model_citations_from_docs(docs)
    full_result_dict['_LLM_CONTEXT'] = [_pack(docs) for docs in docs_list]
    full_result_dict['_LLM_PROMPT_VECTOR'] = [json.dumps(emb) for emb in query_embeddings]
    enabled_citation_columns = json.loads(RuntimeParameters.get(CustomModelRuntimeParameters.ENABLE_CITATION_COLUMNS.value))
    citation_fields = config.vector_database_settings.enabled_citations_fields if config.vector_database_settings and config.vector_database_settings.enabled_citations_fields else MetadataColumnNames.custom_model_citation_fields()
    for i, llm_context_list in enumerate(full_result_dict['_LLM_CONTEXT']):
        for llm_context_item in llm_context_list:
            for citation_field in citation_fields:
                if not enabled_citation_columns.get(citation_field, False):
                    if citation_field in llm_context_item:
                        llm_context_item.pop(citation_field)
                    elif 'metadata' in llm_context_item and citation_field in llm_context_item['metadata']:
                        llm_context_item['metadata'].pop(citation_field)
        full_result_dict['_LLM_CONTEXT'][i] = json.dumps(llm_context_list)