# Copyright 2023 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
import json
from collections.abc import AsyncIterator
from collections.abc import Sequence
from time import time
from typing import Any
import backoff
from aiohttp import ClientResponse
from aiohttp import ClientResponseError
from aiohttp import ClientSession
from aiohttp.client import DEFAULT_TIMEOUT
from jinja2 import Environment
from langchain_core.documents import Document
from loguru import logger
from openai import APIStatusError
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from deployment_access import DataRobotDeploymentCredentials
from deployment_access import construct_headers
from deployment_access import construct_payload
from deployment_access import parse_response_data
from i18n import gettext
from language_models.helpers import get_settings_and_system_prompt
from language_models.helpers import get_token_count
from language_models.helpers import prepend_system_message_to_messages
from language_models.language_model_interface import MODEL_NAME_FOR_DEFAULT_CHAT_MODEL_ID_FALLBACK
from language_models.language_model_interface import BaseLanguageModelResult
from language_models.language_model_interface import BaseResultMetadata
from language_models.language_model_interface import LanguageModel
from language_models.language_model_interface import LanguageModelError
from vector_database.custom_model_embeddings import RetryableCustomModelError
from vector_database.custom_model_embeddings import get_factor
from vector_database.custom_model_embeddings import get_max_tries
# The template used to format the complete prompt when using the predictions API.
PREDICTIONS_PROMPT_TEMPLATE = '{% if system_prompt %}\n{{ system_prompt }}\n\n{% endif %}\n{% if docs %}\nContext:\n{% for doc in docs %}\n - {{ doc.page_content }}\n\n{% endfor %}\n{% endif %}\n{% if messages|length > 1 %}\nPrevious conversation:\n{% for message in messages[:-1] %}\n{% if message["role"] == "user" %}\nPrompt: {{ message["content"] }}\n{% elif message["role"] == "assistant" %}\nResponse: {{ message["content"] }}\n{% endif %}\n\n{% endfor %}\n{% endif %}\nPrompt: {{ messages[-1]["content"] }}\n\n'
DEPLOYMENT_TEMPORARILY_UNAVAILABLE_MESSAGE = 'The deployment returned a server error. The error code indicates it may be temporarily unavailable to serve requests. Will retry the chat completion request.'

@backoff.on_exception(backoff.expo, exception=RetryableCustomModelError, max_tries=get_max_tries, factor=get_factor, jitter=None)
async def make_prompt_call_to_custom_llm(session: ClientSession, inference_api_url: str, payload: bytes) -> ClientResponse:
    response = await session.post(inference_api_url, data=payload)
    if response.status in [502, 503, 504]:
        logger.bind(base_url=inference_api_url, status_code=response.status).warning(DEPLOYMENT_TEMPORARILY_UNAVAILABLE_MESSAGE)  # type: ignore[arg-type]
        raise RetryableCustomModelError() from ClientResponseError(response.request_info, response.history, status=response.status, message=response.reason, headers=response.headers)
    elif response.status == 422:
        error_message = await response.json()
        raise ClientResponseError(response.request_info, response.history, status=response.status, message=error_message.get('message', response.reason), headers=response.headers)
    else:
        response.raise_for_status()
    return response

@backoff.on_exception(backoff.expo, exception=RetryableCustomModelError, max_tries=get_max_tries, factor=get_factor, jitter=None)
async def make_openai_chat_api_call_to_custom_llm(client: AsyncOpenAI, **kwargs: Any) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
    """
    Make an OpenAI chat completion request to the custom model LLM deployment,
    retrying the request if the deployment reports it is unavailable.
    """
    try:
        completion_or_stream = await client.chat.completions.create(**kwargs)
        return completion_or_stream
    except APIStatusError as e:
        if e.status_code in [502, 503, 504]:
            logger.bind(base_url=client.base_url, status_code=e.status_code).warning(DEPLOYMENT_TEMPORARILY_UNAVAILABLE_MESSAGE)
            raise RetryableCustomModelError() from e
        else:
            raise

class BaseCustomModelLLMErrorHandler:

    async def handle(self) -> None:
        pass

class CustomModelLLM(LanguageModel):
    """LLM implemented as deployed custom model."""

    def __init__(self, credentials: BaseModel, prediction_api_url: str, chat_api_url: str | None, model_type: str, input_type: str, prompt_column_name: str | None, target_column_name: str | None, association_id_column: str | None, supports_chat_api: bool=False, chat_model_id: str | None=None, error_handler: BaseCustomModelLLMErrorHandler | None=None):
        super().__init__()
        self.credentials = DataRobotDeploymentCredentials.model_validate(credentials, from_attributes=True)
        self.prediction_api_url = prediction_api_url
        self.chat_api_url = chat_api_url
        self.model_type = model_type
        self.input_type = input_type
        self.prompt_column_name = prompt_column_name
        self.target_column_name = target_column_name
        self.association_id_column = association_id_column
        self.supports_chat_api = supports_chat_api
        self.default_model_id = chat_model_id or MODEL_NAME_FOR_DEFAULT_CHAT_MODEL_ID_FALLBACK
        self.error_handler = error_handler
        self.supports_response_streaming = True
        self._client: AsyncOpenAI | None = None

    def _create_openai_client(self) -> AsyncOpenAI:
        """Create an async OpenAI client for communicating with the DataRobot deployment."""
        if not self.chat_api_url:
            raise CustomModelLLMError(gettext("The URL of the deployment's chat endpoint was not specified."))
        # OpenAI client needs the base URL only; it appends the API path on its own.
        base_url = self.chat_api_url
        completions_suffix = '/chat/completions'
        if base_url.endswith(completions_suffix):
            base_url = base_url[:-len(completions_suffix)]
        # OpenAI client needs the token value only; it constructs the Bearer header on its own.
        api_key = self.credentials.authorization_header
        bearer_prefix = 'Bearer '
        if api_key.startswith(bearer_prefix):
            api_key = api_key[len(bearer_prefix):]
        return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=DEFAULT_TIMEOUT.total, max_retries=0)

    @property
    def client(self) -> AsyncOpenAI:
        """Get or create a reusable async OpenAI client."""
        if self._client is None:
            self._client = self._create_openai_client()
        return self._client

    async def _submit_prompt(self, messages: Sequence[ChatCompletionMessageParam], llm_settings: dict, docs: Sequence[Document] | None=None, model_id: str | None=None) -> BaseLanguageModelResult:
        """Submit a prompt to a custom model LLM with the specified settings, chat history, and
        retrieved vector database documents, if any.

        Parameters
        ----------
        messages
            The chat messages to submit to the model, current prompt and history
        llm_settings
            The DataRobot LLM settings specified by the user
        docs
            Documents retrieved from a vector database
        model_id
            The model ID to use for the chat completion request (only for deployments that support
            the chat completion API).

        Returns
        -------
        BaseLanguageModelResult
            The result text and metadata
        """
        settings, system_prompt = get_settings_and_system_prompt(llm_settings)
        if self.supports_chat_api and self.chat_api_url:
            return await self._submit_prompt_using_chat_api(system_prompt=system_prompt, messages=messages, llm_settings=settings, docs=docs or [], model_id=model_id)
        else:
            return await self._submit_prompt_using_predictions_api(system_prompt=system_prompt, messages=messages, llm_settings=settings, docs=docs or [])

    async def _submit_prompt_with_response_streaming(self, messages: Sequence[ChatCompletionMessageParam], llm_settings: dict, docs: Sequence[Document] | None=None, model_id: str | None=None) -> AsyncIterator[ChatCompletionChunk]:
        """
        Submit a prompt to a custom model LLM with the specified settings, chat history, and
        retrieved vector database documents, if any. Request a streaming response.

        Parameters
        ----------
        messages
            The chat messages to submit to the model, current prompt and history
        llm_settings
            The DataRobot LLM settings specified by the user
        docs
            Documents retrieved from a vector database
        model_id
            The model ID to use for the chat completion request.
            If not specified, will use default model ID for this LLM.

        Yields
        ------
        OpenAI `chat.completion.chunk` objects.
        """
        settings, system_prompt = get_settings_and_system_prompt(llm_settings)
        request_messages = prepend_system_message_to_messages(messages=messages, system_prompt=system_prompt, docs=docs or [])  # type: ignore[assignment]
        response_stream: AsyncIterator[ChatCompletionChunk] = await make_openai_chat_api_call_to_custom_llm(client=self.client, model=model_id or self.default_model_id, messages=request_messages, stream=True, **settings)
        async for chunk in response_stream:
            yield chunk

    async def _submit_prompt_using_chat_api(self, system_prompt: str, messages: Sequence[ChatCompletionMessageParam], llm_settings: dict, docs: Sequence[Document], model_id: str | None=None) -> BaseLanguageModelResult:
        """Submit a prompt to the custom model LLM deployment using the /chat/completions API."""
        request_messages = prepend_system_message_to_messages(messages=messages, system_prompt=system_prompt, docs=docs)
        # Submit the request to the deployment.
        time_start = time()
        try:  # type: ignore[assignment]
            completion: ChatCompletion = await make_openai_chat_api_call_to_custom_llm(client=self.client, model=model_id or self.default_model_id, messages=request_messages, **llm_settings)
            choice = completion.choices[0]
        except Exception as e:
            if self.error_handler is not None:
                await self.error_handler.handle()  # type: ignore[arg-type]
                return BaseLanguageModelResult(result_text='', result_metadata=BaseResultMetadata(output_token_count=0, input_token_count=0, total_token_count=0, latency_milliseconds=0, error_message=gettext('Custom model LLM request returned an error.'), final_prompt=request_messages))
            else:
                # This code runs inside custom models, where there is no diagnostic access to
                # pinpoint the exact reason of the error.
                msg = gettext('Custom model LLM request returned an error. Try again, and if that does not resolve the issue, send the prompt to the associated LLM blueprint and a more detailed error will be returned in the playground.')
                raise CustomModelLLMError(msg) from e
        time_end = time()
        # Parse the completion object
        result_text = choice.message.content
        if completion.usage:
            input_token_count = completion.usage.prompt_tokens
            output_token_count = completion.usage.completion_tokens
            total_token_count = completion.usage.total_tokens
        else:
            input_token_count = get_token_count(json.dumps(request_messages))
            output_token_count = get_token_count(result_text or '')
            total_token_count = input_token_count + output_token_count  # type: ignore[arg-type]
        return BaseLanguageModelResult(result_text=result_text, result_metadata=BaseResultMetadata(output_token_count=output_token_count, input_token_count=input_token_count, total_token_count=total_token_count, latency_milliseconds=int((time_end - time_start) * 1000), final_prompt=request_messages))

    async def _submit_prompt_using_predictions_api(self, system_prompt: str, messages: Sequence[ChatCompletionMessageParam], llm_settings: dict, docs: Sequence[Document]) -> BaseLanguageModelResult:
        """Submit a prompt to the custom model LLM deployment using the /predictions/ API."""
        # Currently no usable settings here, but there may be in the future.
        _ = llm_settings
        if not self.prompt_column_name:
            raise RuntimeError('Prompt column name was not specified for the custom model LLM.')
        if not self.target_column_name:
            raise RuntimeError('Target column name was not specified for the custom model LLM.')
        # Build the prompt body
        jinja_env = Environment(trim_blocks=True)
        template = jinja_env.from_string(PREDICTIONS_PROMPT_TEMPLATE)
        input_text = template.render(system_prompt=system_prompt, docs=docs, messages=messages)
        headers = construct_headers(model_type=self.model_type, credentials=self.credentials)
        payload = construct_payload(self.input_type, [self.prompt_column_name], [input_text], self.association_id_column)
        # Submit the request to the deployed model
        time_start = time()
        try:
            async with ClientSession(headers=headers, timeout=DEFAULT_TIMEOUT) as session:
                response = await make_prompt_call_to_custom_llm(session=session, inference_api_url=self.prediction_api_url, payload=payload)
                response_data = await response.json()
        except Exception as e:
            if self.error_handler is not None:
                await self.error_handler.handle()
                return BaseLanguageModelResult(result_text='', result_metadata=BaseResultMetadata(output_token_count=0, input_token_count=0, total_token_count=0, latency_milliseconds=0, error_message=gettext('Custom model LLM request returned an error.'), final_prompt=input_text))
            else:
                # This code runs inside custom models, where there is no diagnostic access to
                # pinpoint the exact reason of the error.
                msg = gettext('Custom model LLM request returned an error. Try again, and if that does not resolve the issue, send the prompt to the associated LLM blueprint and a more detailed error will be returned in the playground.')
                raise CustomModelLLMError(msg) from e
        # Parse the response
        # Validation of the deployment confirmed this to be string
        result_text = parse_response_data(response_data, self.model_type, self.target_column_name)
        time_end = time()
        input_token_count = get_token_count(input_text)
        output_token_count = get_token_count(result_text)
        return BaseLanguageModelResult(result_text=result_text, result_metadata=BaseResultMetadata(output_token_count=output_token_count, input_token_count=input_token_count, total_token_count=input_token_count + output_token_count, latency_milliseconds=int((time_end - time_start) * 1000), final_prompt=input_text))

    async def ping(self) -> bool:
        """
        Check whether a model is accessible. This is not really testable for a generic
        custom model LLM endpoint since it depends on the deployment credentials for
        a specific deployment.

        Returns
        -------
        True if accessible otherwise False.
        """
        return True

class CustomModelLLMError(LanguageModelError):
    """Raised when the custom model is unusable for some reason."""