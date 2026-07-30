# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from __future__ import annotations
import json
from collections.abc import AsyncIterator
from collections.abc import Sequence
from enum import StrEnum
from time import time
from typing import Any
import backoff
import httpx
from langchain_core.documents import Document
from loguru import logger
from openai import APIConnectionError
from openai import APIStatusError
from openai import AsyncOpenAI
from openai import RateLimitError
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessageParam
from language_models.helpers import get_settings_and_system_prompt
from language_models.helpers import get_token_count
from language_models.helpers import get_tool_settings
from language_models.helpers import prepend_system_message_to_messages
from language_models.language_model_interface import BaseLanguageModelResult
from language_models.language_model_interface import BaseResultMetadata
from language_models.language_model_interface import BaseToolCall
from language_models.language_model_interface import BaseToolCallRequest
from language_models.language_model_interface import LanguageModel
from language_models.language_model_interface import LanguageModelCredentials
from language_models.language_model_interface import LanguageModelError

class LLMGatewayError(LanguageModelError):
    """An error that occurs when submitting a prompt to the LLM gateway."""
    pass

class RetryableLLMGatewayError(LLMGatewayError):
    pass

class LLMGatewayRateLimitError(LLMGatewayError):
    """A 429 rate limit error returned by the LLM gateway."""

    def __init__(self, message: str, retry_after: str | None=None):
        super().__init__(message)
        self.retry_after = retry_after

def extract_error_detail(exception: Exception) -> str:
    """
    Extract the detail message from the client error responses.

    Args:
        exception: The exception object from the client

    Returns
    -------
        The extracted detail message, or the complete exception if extraction fails
    """
    try:
        return exception.body['detail']  # type: ignore[attr-defined]
    except (AttributeError, KeyError, TypeError):
        return str(exception)

class LLMWorkloads(StrEnum):
    """Workload codes used by the metering project."""
    PLAYGROUND = 'genai-playground'
    CUSTOM_MODEL = 'custom-model'

class LLMGatewayCredentials(LanguageModelCredentials):
    """Credentials for the LLM gateway."""
    base_url: str
    api_token: str
    client_id: LLMWorkloads
    llm_id: str
    user_credentials: LanguageModelCredentials | None = None

class LLMGatewayLanguageModel(LanguageModel):
    """Language model that uses the LLM gateway to submit a prompt."""
    default_model_id = 'llm-gateway-model'

    def __init__(self, credentials: LLMGatewayCredentials):
        super().__init__(credentials)
        self.credentials = credentials
        self.base_url = self.credentials.base_url
        self.default_headers = {'User-Agent': 'Buzok', 'Client-Id': self.credentials.client_id.value}
        self.supports_response_streaming = True
        self._client: AsyncOpenAI | None = None

    def _create_client(self) -> AsyncOpenAI:
        """Create an async OpenAI client for communicating with the LLM gateway."""
        # OpenAI client needs the token value only
        # it constructs the Bearer header on its own.
        return AsyncOpenAI(base_url=self.base_url, api_key=self.credentials.api_token, default_headers=self.default_headers, timeout=httpx.Timeout(5 * 60, connect=5), max_retries=0)

    @property
    def client(self) -> AsyncOpenAI:
        """Get or create a reusable async OpenAI client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_request(self, messages: Sequence[ChatCompletionMessageParam], llm_settings: dict, docs: Sequence[Document] | None=None, model_id: str | None=None, stream: bool=False) -> dict[str, Any]:
        settings, system_prompt = get_settings_and_system_prompt(llm_settings)
        tool_settings = get_tool_settings(settings)
        request_messages = prepend_system_message_to_messages(messages=messages, system_prompt=system_prompt, docs=docs or [])
        settings.update(tool_settings)
        # pass user credentials if they are provided
        # pass deployment_id if it is provided, note that this will overwrite model and llm_id
        # We don't set llm_id if the model_id is provided
        # because the LLM gateway prioritizes llm_id over model
        # and model must always be set because the openai client requires it
        # By passing llm settings as top level keys in extra_body, they are directly
        # passed to the litellm completion call unvalidated by openai client and gatway api
        # Furthermore by unpacking them last, we ensure user can override default settings
        extra_body = {'credential_json': self.credentials.user_credentials.model_dump_json(exclude_none=True) if self.credentials.user_credentials else None, 'deployment_id': getattr(self.credentials.user_credentials, 'deployment_id', None), 'llm_id': None if model_id else self.credentials.llm_id, **settings}
        request = {'messages': request_messages, 'model': model_id or self.default_model_id, 'extra_body': extra_body, 'stream': stream}
        return request

    @backoff.on_exception(backoff.expo, exception=RetryableLLMGatewayError, max_tries=3)
    async def _submit_request(self, request: dict[str, Any]) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Make chat completion request to LLM gateway, retrying the request in case of
        connection issues with connection or proxies.
        """
        # We don't want to log user credentials
        extra_body = request['extra_body'].copy()
        extra_body.pop('credential_json', None)
        request_logger = logger.bind(model=request['model'], extra_body=extra_body, stream=request['stream'], base_url=self.base_url)
        try:
            completion_or_stream = await self.client.chat.completions.create(**request)
            return completion_or_stream
        except APIConnectionError as e:
            request_logger.warning('Failed to establish connection to LLM gateway')
            raise RetryableLLMGatewayError(e) from e
        except RateLimitError as e:
            # Rate limited, either in LLM Gateway or upsteam (no difference the client).
            # Not retryable internally, let client decide if request has to be retried and if
            # that retry should respect Retry-After header.
            request_logger.warning('LLM gateway rate limited request')
            retry_after = e.response.headers.get('retry-after')
            error_detail = extract_error_detail(e)
            raise LLMGatewayRateLimitError(error_detail, retry_after) from e
        except APIStatusError as e:
            request_logger.warning('LLM gateway chat completion errored', extra={'status_code': e.status_code})
            error_detail = extract_error_detail(e)
            if e.status_code in {502, 503, 504}:
                raise RetryableLLMGatewayError(error_detail) from e
            else:
                raise LLMGatewayError(error_detail) from e
        except Exception as e:
            request_logger.exception('LLM gateway chat completion failed')
            error_detail = extract_error_detail(e)
            raise LLMGatewayError(error_detail) from e

    async def _submit_prompt(self, messages: Sequence[ChatCompletionMessageParam], llm_settings: dict, docs: Sequence[Document] | None=None, model_id: str | None=None) -> BaseLanguageModelResult:
        """Submit a prompt to the LLM gateway, requesting a completion."""
        request = self._create_request(messages=messages, llm_settings=llm_settings, docs=docs, model_id=model_id, stream=False)
        time_start = time()
        completion: ChatCompletion = await self._submit_request(request)  # type: ignore[assignment]
        time_end = time()
        return get_language_model_result_from_completion(completion, request, time_end - time_start)

    async def _submit_prompt_with_response_streaming(self, messages: Sequence[ChatCompletionMessageParam], llm_settings: dict, docs: Sequence[Document] | None=None, model_id: str | None=None) -> AsyncIterator[ChatCompletionChunk]:
        """Submit a prompt to the LLM gateway, requesting a streaming response."""
        request = self._create_request(messages=messages, llm_settings=llm_settings, docs=docs, model_id=model_id, stream=True)
        response_stream: AsyncIterator[ChatCompletionChunk] = await self._submit_request(request)  # type: ignore[assignment]
        async for chunk in response_stream:
            yield chunk

def get_language_model_result_from_completion(completion: ChatCompletion, request_parameters: dict[str, Any], time_delta: float) -> BaseLanguageModelResult:
    if completion.usage:
        input_token_count = completion.usage.prompt_tokens
        output_token_count = completion.usage.completion_tokens
        total_token_count = completion.usage.total_tokens
    else:
        result_text = completion.choices[0].message.content
        input_token_count = get_token_count(json.dumps(request_parameters['messages']))
        output_token_count = get_token_count(result_text or '')
        total_token_count = input_token_count + output_token_count
    tool_call_request = None
    if completion.choices[0].message.tool_calls:
        tool_calls = []
        for tool_call in completion.choices[0].message.tool_calls:
            if tool_call.type == 'function':
                name = tool_call.function.name
                arguments = tool_call.function.arguments
            else:
                name = tool_call.custom.name
                arguments = tool_call.custom.input
            tool_calls.append(BaseToolCall(tool_name=name, tool_arguments=arguments, id=tool_call.id))
        tool_call_request = BaseToolCallRequest(tool_calls=tool_calls)  # type: ignore[arg-type]
    return BaseLanguageModelResult(result_text=completion.choices[0].message.content, tool_call_request=tool_call_request, result_metadata=BaseResultMetadata(final_prompt=request_parameters['messages'], input_token_count=input_token_count, output_token_count=output_token_count, total_token_count=total_token_count, latency_milliseconds=int(time_delta * 1000)))