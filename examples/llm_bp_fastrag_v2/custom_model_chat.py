# Copyright 2023 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
import time
import uuid
from collections.abc import AsyncIterator
from collections.abc import Sequence
from typing import Any
from langchain_core.documents import Document
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionAssistantMessageParam
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessageToolCallParam
from openai.types.chat import CompletionCreateParams
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call_param import Function as OpenAIFunction
from custom_model_entities import CustomModelLLMBlueprintConfig
from language_models.language_model_interface import MODEL_NAME_FOR_DEFAULT_CHAT_MODEL_ID_FALLBACK
from language_models.language_model_interface import BaseLanguageModelResult
from language_models.language_model_interface import LanguageModel
from vector_database.inference.entities import MetadataColumnNames
from vector_database.inference.entities import QueryEmbeddings

def translate_completion_create_params_to_llm_settings(completion_create_params: CompletionCreateParams) -> dict[str, Any]:
    """
    Transform the LLM settings submitted in the chat completion request to the schema
    understood by the target LLM.
    """
    llm_settings: dict[str, Any] = {}
    # Transfer all parameters from CompletionCreateParams
    llm_settings.update(completion_create_params)
    # Rename the `max_tokens` (deprecated in OpenAI) or `max_completion_tokens` parameter
    # to "max_completion_length" since the internals of `submit_prompt` expect this naming.
    if 'max_tokens' in llm_settings:
        llm_settings['max_completion_length'] = llm_settings.pop('max_tokens')
    if 'max_completion_tokens' in llm_settings:
        llm_settings['max_completion_length'] = llm_settings.pop('max_completion_tokens')
    if llm_settings.get('n', 1) != 1:
        msg = 'LLM blueprints support generating only one completion choice.'
        raise ValueError(msg)
    # extract system prompt
    system_prompt = next((m for m in completion_create_params['messages'] if m['role'] == 'system'), None)
    if system_prompt:
        llm_settings['system_prompt'] = system_prompt['content']
    # `messages` are parsed separately and organized into list[ChatCompletionMessageParam].
    llm_settings.pop('messages', None)
    # `model` is handled separately in `generate_chat_completion`.
    llm_settings.pop('model', None)
    # Response streaming is handled separately in `generate_chat_completion`.
    llm_settings.pop('stream', None)
    return llm_settings

def get_language_model_citations_from_docs(docs: Sequence[Document]) -> list[dict[str, Any]]:
    return [{'content': doc.page_content, 'link': f"{doc.metadata.get(MetadataColumnNames.source.value, '')}:{doc.metadata.get(MetadataColumnNames.page.value, '')}", 'vector': doc.metadata.pop(MetadataColumnNames.doc_vector.value, None), 'metadata': doc.metadata} for doc in docs]

def create_chat_completion_from_llm_result(llm: LanguageModel, config: CustomModelLLMBlueprintConfig, llm_result: BaseLanguageModelResult, docs: Sequence[Document] | None=None, model_id: str | None=None, prompt_vector: QueryEmbeddings=None) -> ChatCompletion:
    """Convert the result of LLM invocation into an OpenAI `chat.completion` object."""
    is_content_filtered = bool(llm_result.result_metadata.provider_llm_guards)
    message = ChatCompletionAssistantMessageParam(content=llm_result.result_text, role='assistant')
    if llm_result.tool_call_request:
        message['tool_calls'] = [ChatCompletionMessageToolCallParam(id=call.id, type='function', function=OpenAIFunction(name=call.tool_name, arguments=call.tool_arguments)) for call in llm_result.tool_call_request.tool_calls]  # type: ignore[arg-type]
    choice = Choice(index=0, message=message, finish_reason='content_filter' if is_content_filtered else 'stop')
    completion_id = str(uuid.uuid4())
    completion_timestamp = int(time.time())
    completion = ChatCompletion(id=completion_id, object='chat.completion', choices=[choice], created=completion_timestamp, model=model_id or llm.default_model_id, usage=CompletionUsage(completion_tokens=llm_result.result_metadata.output_token_count, prompt_tokens=llm_result.result_metadata.input_token_count, total_tokens=llm_result.result_metadata.total_token_count))
    # Include the extra metadata with citations.
    citations = get_language_model_citations_from_docs(docs) if docs else None
    # Include the extra metadata with provider LLM guards.
    provider_guards = None
    if llm_result.result_metadata.provider_llm_guards:
        provider_guards = [guard.model_dump(exclude_none=False, exclude_unset=False) for guard in llm_result.result_metadata.provider_llm_guards]
    # These are extra attributes that are not defined in ChatCompletion.
    # When using the `openai` library, they need `OpenAI(_strict_response_validation=False)`
    # to work properly, which is set to False by default.
    completion.citations = citations  # type: ignore[attr-defined]
    completion.llm_blueprint_id = config.llm_blueprint_id  # type: ignore[attr-defined]
    completion.llm_provider_guards = provider_guards  # type: ignore[attr-defined]
    completion.prompt_vector = prompt_vector  # type: ignore[attr-defined]
    return completion

async def generate_chat_completion(llm: LanguageModel, config: CustomModelLLMBlueprintConfig, completion_create_params: CompletionCreateParams, docs: Sequence[Document] | None=None, prompt_vector: QueryEmbeddings=None) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
    """
    Generate the LLM blueprint's completion for the specified chat conversation.

    Parameters
    ----------
    llm
        The language model the LLM blueprint uses.
    config
        The configuration of the LLM blueprint.
    completion_create_params
        The parameters of the chat completion request.
    docs
        The documents retrieved from the vector database.
    prompt_vector
        The embedding vector of the user prompt.

    Returns
    -------
    ChatCompletion when a non-streaming chat completion is requested.
    Iterator[ChatCompletionChunk] when a streaming chat completion is requested.
    """
    llm_settings_from_blueprint = config.llm_settings
    llm_settings_from_request = translate_completion_create_params_to_llm_settings(completion_create_params)
    llm_settings = {}
    llm_settings.update(llm_settings_from_blueprint)
    llm_settings.update(llm_settings_from_request)
    # We pass messages as is except for the system message which is passed in llm_settings.
    messages = [m for m in completion_create_params['messages'] if m['role'] != 'system']
    # Prefer the user-specified value for the `model` parameter, but allow an empty value
    # or a special value that falls back to the LLM client's default model ID.
    model_id = completion_create_params.get('model')
    if model_id == MODEL_NAME_FOR_DEFAULT_CHAT_MODEL_ID_FALLBACK:
        # Passing `model_id=None` allows the LLM client to fall back to its default model ID.
        model_id = None
    use_streaming = completion_create_params.get('stream')
    if use_streaming:
        if not llm.supports_response_streaming:
            raise ValueError('This LLM blueprint does not support response streaming')

        def attach_extra_metadata_to_final_chunk(chunk: ChatCompletionChunk) -> None:
            # Include the extra metadata with citations.
            citations = get_language_model_citations_from_docs(docs) if docs else None
            # Include the extra metadata with provider LLM guards.
            provider_guards = None
            # These are extra attributes that are not defined in ChatCompletionChunk. When using
            # the `openai` library, they need `OpenAI(_strict_response_validation=False)`
            # to work properly, which is set to False by default.
            chunk.citations = citations  # type: ignore[attr-defined]
            chunk.llm_blueprint_id = config.llm_blueprint_id  # type: ignore[attr-defined]
            chunk.llm_provider_guards = provider_guards  # type: ignore[attr-defined]
            chunk.prompt_vector = prompt_vector  # type: ignore[attr-defined]
        return llm.submit_prompt_with_response_streaming(messages=messages, llm_settings=llm_settings, docs=docs, model_id=model_id, final_chunk_hook=attach_extra_metadata_to_final_chunk)
    else:
        llm_result = await llm.submit_prompt(messages=messages, llm_settings=llm_settings, docs=docs, model_id=model_id)
        completion = create_chat_completion_from_llm_result(llm=llm, config=config, llm_result=llm_result, docs=docs, model_id=model_id, prompt_vector=prompt_vector)
        return completion