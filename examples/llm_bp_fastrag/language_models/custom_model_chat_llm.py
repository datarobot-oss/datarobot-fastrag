# Copyright 2023 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
import asyncio
import os
from collections.abc import AsyncIterator
from collections.abc import Sequence
from time import time
from typing import Any
from uuid import uuid4

from aiohttp import ClientSession
from aiohttp import ClientTimeout
from i18n import gettext
from langchain.schema import Document
from language_models.custom_model_chat_llm_helper import get_guard_configuration_for_custom_model
from language_models.custom_model_chat_llm_helper import get_metrics_metadata
from language_models.helpers import get_settings_and_system_prompt
from language_models.language_model_interface import BaseLanguageModelResult
from language_models.language_model_interface import BaseMetricMetadata
from language_models.language_model_interface import BasePlaygroundContext
from language_models.language_model_interface import BaseResultMetadata
from language_models.language_model_interface import LanguageModel
from language_models.language_model_interface import LanguageModelError
from loguru import logger
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessageParam

MAX_WAIT = 1200  # 20 minutes
TIMEOUT_INDIVIDUAL_REQUEST = 10
POLLING_INTERVAL = 1.0


class CustomModelChatLLMError(LanguageModelError):
    """Base error for custom model chat llm."""

    @property
    def user_message(self) -> str:
        return str(self)


class CustomModelChatLLMSetupError(CustomModelChatLLMError):
    """Error raised when setup / job submission fails for custom model chat llm."""

    pass


class CustomModelChatLLMPromptError(CustomModelChatLLMError):
    """Error raised when chat completion job fails."""

    def __init__(self, error_message: str, error_details: str) -> None:
        super().__init__("Custom model failed to execute chat completion prompt.")
        # Those values contain sensitive information, so we don't want
        # to log them, but we want to pass them back to the user
        self.error_message = error_message
        self.error_details = error_details

    @property
    def user_message(self) -> str:
        return gettext(
            "Custom model failed to execute chat completion prompt.\nError: {error_message}.\n{error_details}"
        ).format(error_message=self.error_message, error_details=self.error_details)


class CustomModelChatLLM(LanguageModel):
    """LLM implemented as deployed custom model."""

    def __init__(
        self,
        datarobot_endpoint: str | None = None,
        authorization_header: str | None = None,
        user_agent_header: str | None = None,
        playground_context: BasePlaygroundContext | None = None,
    ):
        super().__init__()
        # This will make it work without any arguments in custom model environment
        self.datarobot_endpoint = datarobot_endpoint or os.environ["DATAROBOT_ENDPOINT"]
        authorization_header = authorization_header or f"Bearer {os.environ['DATAROBOT_API_TOKEN']}"
        user_agent_header = user_agent_header or "custom-model"
        self.headers = {"User-Agent": user_agent_header, "Authorization": authorization_header}
        self.playground_context = playground_context

    async def _submit_prompt(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        llm_settings: dict,
        docs: Sequence[Document] | None = None,
        model_id: str | None = None,
    ) -> BaseLanguageModelResult:
        custom_model_id = llm_settings["custom_model_id"]
        with logger.contextualize(custom_model_id=custom_model_id):
            logger.info("Submitting prompt to custom model chat LLM")
            settings, system_prompt = get_settings_and_system_prompt(llm_settings)
            custom_model_id = settings["custom_model_id"]
            custom_model_version_id = settings.get("custom_model_version_id")
            url = os.path.join(
                self.datarobot_endpoint, f"genai/agents/fromCustomModel/{custom_model_id}/chat/"
            )
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}, *messages]
            payload: dict[str, Any] = {"messages": messages}
            if model_id:
                payload["model"] = model_id
            if custom_model_version_id:
                payload["customModelVersionId"] = str(custom_model_version_id)
            if self.playground_context:
                payload["tracingContext"] = {
                    "entityId": self.playground_context.use_case_id,
                    "entityType": "use_case",
                    "attributes": {
                        "chat_prompt_id": self.playground_context.chat_prompt_id,
                        "playground_id": self.playground_context.playground_id,
                    },
                }
            async with ClientSession(
                headers=self.headers, timeout=ClientTimeout(total=TIMEOUT_INDIVIDUAL_REQUEST)
            ) as session:
                time_start = time()
                response_data = await self._chat_completion_async(session, url, payload)
                time_end = time()
                latency_milliseconds = int((time_end - time_start) * 1000)
                usage = response_data.get("usage", {}) or {}
                metrics: list[BaseMetricMetadata] = []
                moderation_lib_results = response_data.get("datarobot_moderations")
                if moderation_lib_results:
                    guard_configurations = await get_guard_configuration_for_custom_model(
                        session, self.datarobot_endpoint, custom_model_id
                    )
                    metrics = get_metrics_metadata(
                        guard_configurations, response_data["datarobot_moderations"]
                    )
            result_metadata = BaseResultMetadata(
                output_token_count=usage.get("completion_tokens", 0),
                input_token_count=usage.get("prompt_tokens", 0),
                total_token_count=usage.get("total_tokens", 0),
                latency_milliseconds=latency_milliseconds,
                pipeline_interactions=response_data.get("pipeline_interactions", None),
                metrics=metrics,
            )
            return BaseLanguageModelResult(
                result_text=response_data["choices"][0]["message"]["content"],
                result_metadata=result_metadata,
            )

    def _submit_prompt_with_response_streaming(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        llm_settings: dict,
        docs: Sequence[Document] | None = None,
        model_id: str | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Submit a prompt to an external or internal LLM, requesting a streaming response."""
        raise NotImplementedError("Streaming is not supported for custom model chat completion")

    async def _chat_completion_async(
        self, session: ClientSession, url: str, payload: dict
    ) -> dict[str, Any]:
        response = await session.post(url, json=payload)
        if not response.ok or not response.headers.get("Location"):
            status = response.status
            try:
                response_data = await response.json()
                error = response_data.get("detail")
            except Exception:
                error = await response.text()
            # Here it's error from our API, so it's okay to log it.
            # There shouldn't be any sensitive info.
            logger.bind(response_status=status, response_text=error).error(
                "Custom model chat failed"
            )
            raise CustomModelChatLLMSetupError(error)
        chat_completion_location = await self._wait_for_async_resolution(
            session, response.headers["Location"]
        )
        response = await session.get(chat_completion_location)
        chat_completion = await response.json()
        if chat_completion.get("errorMessage"):
            error_message = chat_completion.get("errorMessage")
            error_details = chat_completion.get("errorDetails")
            raise CustomModelChatLLMPromptError(error_message, error_details)
        # Cast it to ChatCompletion
        chat_completion.setdefault("id", str(uuid4()))
        chat_completion.setdefault("created", int(time()))
        chat_completion.setdefault("model", "unknown")
        chat_completion["object"] = "chat.completion"
        return chat_completion

    async def _wait_for_async_resolution(
        self,
        session: ClientSession,
        async_location: str,
        max_wait: int = MAX_WAIT,
        polling_interval: float = POLLING_INTERVAL,
    ) -> str:
        """Wait for async response to resolve."""
        ctx_logger = logger.bind(
            async_location=async_location, max_wait=max_wait, polling_interval=polling_interval
        )
        start_time = time()
        while time() < start_time + max_wait:
            response = await session.get(async_location, allow_redirects=False)
            if response.status == 303:
                return response.headers["Location"]
            data = await response.json()
            if data["status"] in {"ERROR", "ABORTED"}:
                ctx_logger.bind(status=data["status"]).error("Async request failed")
                # If job has failed, it's most likely a problem with codespace and
                # should be raised to the user.
                raise CustomModelChatLLMSetupError(
                    data.get("message") or gettext("Failed to run custom model in a codespace.")
                )
            await asyncio.sleep(polling_interval)
        ctx_logger.error("Timeout waiting for chat completion async job to resolve")
        raise TimeoutError(
            gettext("Client timed out waiting for chat completion async job to resolve")
        )
