# Copyright 2023 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from __future__ import annotations

import asyncio
import time
import uuid
from abc import abstractmethod
from collections import namedtuple
from collections.abc import AsyncIterator
from collections.abc import Sequence
from enum import StrEnum
from typing import Any
from typing import Callable
from typing import TypeAlias

from i18n import gettext
from i18n import gettext_noop
from langchain.schema import Document
from loguru import logger
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_chunk import Choice as StreamingChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import BaseModel

# The reserved model name that makes the LLM client use its default model ID when submitting
# prompts to 3rd-party LLM services instead of the user-supplied model name.
MODEL_NAME_FOR_DEFAULT_CHAT_MODEL_ID_FALLBACK = "datarobot-deployed-llm"
# NOTE: Declaring GPTFinalPromptType as list[ChatCompletionMessageParam] would be more accurate, but
# it can be dangerous. OpenAI message types like ChatCompletionUserMessageParam are TypedDicts with
# Iterable[] values, and GPTFinalPromptType is used in pydantic field declarations.
# Currently, pydantic has an open issue where assigning values to Iterable fields coerces them to
# ValidationIterator, so the value that is read back is not equal to the value that was assigned:
# https://github.com/pydantic/pydantic/issues/9467
MultipartMessageContentType: TypeAlias = list[dict[str, Any]]
GPTFinalPromptType: TypeAlias = list[dict[str, str | MultipartMessageContentType | None]]
BedrockFinalPromptType: TypeAlias = list[dict[str, str | MultipartMessageContentType]]
GoogleFinalPromptType: TypeAlias = dict[str, str | list[dict[str, str]]]
FinalPromptType: TypeAlias = (
    str | GPTFinalPromptType | BedrockFinalPromptType | GoogleFinalPromptType | None
)
LLMBlueprintModel = namedtuple("LLMBlueprintModel", ["language_model", "vector_store"])


class LanguageModelCredentials(BaseModel):
    """Base class for language model credentials required to submit an LLM prompt."""

    pass


class CustomModelLLMCredentialType(StrEnum):
    AZURE = "azure_openai"
    GOOGLE = "google_vertex_ai"
    AMAZON = "amazon_bedrock"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    OPENAI = "openai"
    TOGETHERAI = "togetherai"
    GROQ = "groq"
    CEREBRAS = "cerebras"
    DATAROBOT = "datarobot"


class Tool(BaseModel):
    """A tool definition that can be called by an LLM."""

    name: str
    description: str
    # JSON schema object: https://json-schema.org/understanding-json-schema/basics
    parameters: dict[str, Any]
    required: bool = False

    def to_openai_tool_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_google_tool_dict(self) -> dict[str, Any]:
        # Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/Tool
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def to_bedrock_tool_dict(self) -> dict[str, Any]:
        # Reference and examples:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
        # https://docs.anthropic.com/en/docs/build-with-claude/tool-use
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {"json": self.parameters},
        }

    @staticmethod
    def from_openai_tool_dict(tool_dict: dict[str, Any]) -> Tool:
        return Tool(
            name=tool_dict["function"]["name"],
            description=tool_dict["function"]["description"],
            parameters=tool_dict["function"]["parameters"],
        )


class BaseToolCall(BaseModel):
    """A tool call requested by an LLM."""

    id: str
    tool_name: str
    # JSON string of the tool arguments
    tool_arguments: str


class BaseToolCallRequest(BaseModel):
    tool_calls: list[BaseToolCall]


class BaseProviderGuardStage(StrEnum):
    PROMPT = "prompt"
    RESPONSE = "response"


class BaseProviderGuardsMetadata(BaseModel):
    """Info on the provider guard metrics."""

    satisfy_criteria: bool
    name: str
    value: str | float | int | None
    stage: BaseProviderGuardStage


class BasePipelineStage(StrEnum):
    """Enum that describes at which stage the metric may be calculated."""

    PROMPT_PIPELINE = "prompt_pipeline"
    RESPONSE_PIPELINE = "response_pipeline"


class BaseExecutionStatus(StrEnum):
    """Job and entity execution status."""

    NEW = "NEW"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    REQUIRES_USER_INPUT = "REQUIRES_USER_INPUT"
    SKIPPED = "SKIPPED"  # Not executed and can't be executed
    ERROR = "ERROR"


class BaseMetricMetadata(BaseModel):
    name: str
    value: Any
    stage: BasePipelineStage | None = None
    execution_status: BaseExecutionStatus | None = None
    error_message: str | None = None
    custom_model_guard_id: str | None = None


class BaseResultMetadata(BaseModel):
    """Metadata for an LLM response."""

    cost: float | None = None
    output_token_count: int
    input_token_count: int
    total_token_count: int
    latency_milliseconds: int
    error_message: str | None = None
    final_prompt: FinalPromptType = None
    blocked_result_text: str | None = None
    provider_llm_guards: list[BaseProviderGuardsMetadata] | None = None
    pipeline_interactions: str | None = None
    metrics: list[BaseMetricMetadata] = []

    def merge_token_counts(self, other: BaseResultMetadata) -> BaseResultMetadata:
        """Merge the token counts from another BaseResultMetadata instance."""
        self.output_token_count += other.output_token_count
        self.input_token_count += other.input_token_count
        self.total_token_count += other.total_token_count
        self.latency_milliseconds += other.latency_milliseconds
        return self


class BasePlaygroundContext(BaseModel):
    """Context for a playground LLM request."""

    chat_prompt_id: str
    playground_id: str
    use_case_id: str
    playground_type: str = "rag"


class BaseLanguageModelResult(BaseModel):
    """The response text and metadata for an LLM response."""

    # None result_text is the indicator that no response got generated
    result_text: str | None
    tool_call_request: BaseToolCallRequest | None = None
    result_metadata: BaseResultMetadata


class LanguageModelError(Exception):
    """A generic exception class for LLM errors."""

    pass


class LanguageModelInvalidRequestError(LanguageModelError):
    """
    A generic exception class for LLM errors due to invalid requests.

    Most commonly, this would be caused by exceeding the LLM's context length. However, it is also
    possible for the user to supply an invalid parameter to the deployment's chat API or configure
    the wrong region in the custom model.
    """

    def __init__(
        self,
        msg: str = gettext_noop(
            "The LLM has received an invalid request. Possible causes include exceeding the LLM's context length or an invalid parameter in the chat completion request. Try reducing the user or system prompt, max completion length, or vector database retrieval limits. If calling the chat API of a DataRobot deployment, check the syntax of the parameters in the request. If the issue persists, contact DataRobot."
        ),
        internal_error_msg: str = gettext_noop("No detailed internal error message was supplied."),
        *args: Any,
        **kwargs: Any,
    ):
        self.internal_error_msg = internal_error_msg
        combined_msg = gettext("{msg} Details: {internal_error_msg}").format(
            msg=gettext(msg), internal_error_msg=gettext(internal_error_msg)
        )
        super().__init__(combined_msg, *args, **kwargs)


class LanguageModelInterface:
    """Shared interface for all language model implementations."""

    @abstractmethod
    async def submit_prompt(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        llm_settings: dict | None = None,
        docs: Sequence[Document] | None = None,
        model_id: str | None = None,
    ) -> BaseLanguageModelResult:
        pass


class LanguageModel(LanguageModelInterface):
    """Defines the common interface for all language models."""

    # The model ID used in the requests to the LLM service (DataRobot or 3rd-party).
    # Can be overridden by the user when calling the chat completion API of the deployment.
    default_model_id: str

    def __init__(self, credentials: BaseModel | None = None) -> None:
        self.collected_result_metadata = BaseResultMetadata(
            output_token_count=0, input_token_count=0, total_token_count=0, latency_milliseconds=0
        )
        # Test mode disables any interaction with the LLM API and returns mock responses instead.
        self.enable_test_mode = False
        self.supports_response_streaming = False

    @abstractmethod
    async def _submit_prompt(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        llm_settings: dict,
        docs: Sequence[Document] | None = None,
        model_id: str | None = None,
    ) -> BaseLanguageModelResult:
        """Submit a prompt to a DataRobot deployment or an external LLM service."""
        pass

    @abstractmethod
    def _submit_prompt_with_response_streaming(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        llm_settings: dict,
        docs: Sequence[Document] | None = None,
        model_id: str | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Submit a prompt to an external or internal LLM, requesting a streaming response."""
        pass

    async def submit_prompt(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        llm_settings: dict | None = None,
        docs: Sequence[Document] | None = None,
        model_id: str | None = None,
    ) -> BaseLanguageModelResult:
        """
        Submit a prompt to a DataRobot deployment or an external LLM service.

        Merge the token counts from the result metadata into `collected_result_metadata`.

        Parameters
        ----------
        messages
            The chat messages to submit to the model, current prompt and history.
        llm_settings
            The DataRobot LLM settings specified by the user.
        docs
            The documents retrieved from a vector database.
        model_id
            The model ID to use for the LLM request.
            If not specified, will use default model ID for this LLM.

        Returns
        -------
        LLM invocation result with the merged token counts.
        """
        llm_settings = llm_settings or {}
        # Remove null tools from llm settings.
        if "tools" in llm_settings and llm_settings["tools"] is None:
            llm_settings.pop("tools")
        if self.enable_test_mode:
            # Return a mock response that echoes the contents of the request.
            llm_result = generate_test_llm_result(messages, llm_settings, docs)
        else:
            # Use the LLM client to submit the prompt to the LLM service.
            llm_result = await self._submit_prompt(
                messages=messages, llm_settings=llm_settings, docs=docs, model_id=model_id
            )
        logger.bind(
            model_class_name=self.__class__.__name__,
            model_module_name=self.__module__.split(".")[-1],
            input_token_count=llm_result.result_metadata.input_token_count,
            output_token_count=llm_result.result_metadata.output_token_count,
            latency_milliseconds=llm_result.result_metadata.latency_milliseconds,
        ).info("LLM prompt completed")
        llm_result.result_metadata.merge_token_counts(self.collected_result_metadata)
        self.collected_result_metadata = llm_result.result_metadata
        return llm_result

    async def submit_prompt_with_response_streaming(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        llm_settings: dict | None = None,
        docs: Sequence[Document] | None = None,
        model_id: str | None = None,
        final_chunk_hook: Callable[[ChatCompletionChunk], None] | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """
        Submit a prompt to a DataRobot deployment or an external LLM service,
        requesting a streaming response.
        """
        llm_settings = llm_settings or {}
        # Remove null tools from llm settings
        if "tools" in llm_settings and llm_settings["tools"] is None:
            llm_settings.pop("tools")
        if self.enable_test_mode:
            # Return a mock response that echoes the contents of the request.
            response_stream = generate_test_completion_streaming(
                messages=messages,
                llm_settings=llm_settings,
                docs=docs,
                model_id=model_id or "llm-blueprint",
            )
        else:
            # Use the LLM client to submit the prompt to the LLM service.
            response_stream = self._submit_prompt_with_response_streaming(
                messages, llm_settings=llm_settings, docs=docs, model_id=model_id
            )
        async for chunk in response_stream:
            # If the chunk contains a finish reason, invoke the final chunk hook.
            is_final_chunk = chunk.choices and chunk.choices[0].finish_reason is not None
            if is_final_chunk and final_chunk_hook:
                final_chunk_hook(chunk)
            yield chunk

    async def ping(self) -> bool:
        """
        Check whether a model is accessible.

        Returns
        -------
        True if accessible otherwise False
        """
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(content="Why is the sky blue?", role="user")
        ]
        result = await self.submit_prompt(messages, llm_settings={})
        return bool(result.result_text)


def generate_test_llm_result(
    messages: Sequence[ChatCompletionMessageParam],
    llm_settings: dict | None = None,
    docs: Sequence[Document] | None = None,
) -> BaseLanguageModelResult:
    """Generate a mock LLM result to be used in `LanguageModel.enable_test_mode`."""
    return BaseLanguageModelResult(
        result_text=f"This is a mock LLM response. Prompt: '{messages[-1]['content']}'. Settings: {llm_settings}. Number of messages: {len(messages)}. Number of documents: {(len(docs) if docs else 0)}.",
        result_metadata=BaseResultMetadata(
            input_token_count=10,
            output_token_count=20,
            total_token_count=30,
            latency_milliseconds=1000,
        ),
    )


async def generate_test_completion_streaming(
    messages: Sequence[ChatCompletionMessageParam],
    llm_settings: dict | None = None,
    docs: Sequence[Document] | None = None,
    model_id: str = "llm-blueprint",
    chunk_delay_seconds: float = 0.1,
) -> AsyncIterator[ChatCompletionChunk]:
    """Generate a mock streaming completion to be used in `LanguageModel.enable_test_mode`."""
    completion_id = str(uuid.uuid4())
    completion_timestamp = int(time.time())
    result_text = f"This is a mock LLM response. Prompt: '{messages[-1]['content']}'. Settings: {llm_settings}. Number of messages: {len(messages)}. Number of documents: {(len(docs) if docs else 0)}."
    # Yield the initial chunk.
    await asyncio.sleep(chunk_delay_seconds)
    yield ChatCompletionChunk(
        id=completion_id,
        choices=[
            StreamingChoice(
                delta=ChoiceDelta(role="assistant", content=None), finish_reason=None, index=0
            )
        ],
        created=completion_timestamp,
        model=model_id,
        object="chat.completion.chunk",
    )
    # Yield the chunk with the response text.
    await asyncio.sleep(chunk_delay_seconds)
    yield ChatCompletionChunk(
        id=completion_id,
        choices=[
            StreamingChoice(
                delta=ChoiceDelta(role="assistant", content=result_text),
                finish_reason=None,
                index=0,
            )
        ],
        created=completion_timestamp,
        model=model_id,
        object="chat.completion.chunk",
    )
    # Yield the final chunk with the finish reason.
    await asyncio.sleep(chunk_delay_seconds)
    yield ChatCompletionChunk(
        id=completion_id,
        choices=[
            StreamingChoice(
                delta=ChoiceDelta(role="assistant", content=None), finish_reason="stop", index=0
            )
        ],
        created=completion_timestamp,
        model=model_id,
        object="chat.completion.chunk",
    )
