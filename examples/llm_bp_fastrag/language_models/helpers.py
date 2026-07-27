# Copyright 2023 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from collections.abc import Sequence
from copy import copy
from typing import Any

import tiktoken
from i18n import gettext
from jinja2 import Environment
from langchain.schema import Document
from language_models.language_model_interface import Tool
from openai.types.chat import ChatCompletionMessageParam
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# This is the default that we use for calculation of the token budget when pruning
# history and docs from chat prompts and comparison prompts when max_completion_length is
# not defined.
DEFAULT_CONTEXT_TO_COMPLETION_SIZE_RATIO = 4
# If not defined, i.e. custom model LLM, we assume the context size is 4096 for purposes of
# pruning docs and history.
DEFAULT_CONTEXT_SIZE = 4096
SYSTEM_PROMPT_TEMPLATE = "{% if system_prompt %}\n{{ system_prompt }}\n{%- endif %}\n{% if system_prompt and docs %}\n\n\n{% endif %}\n{% if docs %}\nContext:\n{% for doc in docs %}\n - {{ doc.page_content }}\n\n{% endfor %}\n{% endif %}\n"


def add_docs_to_system_prompt(system_prompt: str, docs: Sequence[Document]) -> str:
    """Add retrieved vector database documents, if any, to the system prompt.

    Parameters
    ----------
    system_prompt
        The system prompt specified by the user
    docs
        List of documents retrieved from a vector database

    Returns
    -------
    system_prompt
        The system prompt with document text appended
    """
    jinja_env = Environment(trim_blocks=True)
    template = jinja_env.from_string(SYSTEM_PROMPT_TEMPLATE)
    return template.render(system_prompt=system_prompt, docs=docs)


def get_token_count(input: str, tokenizer: PreTrainedTokenizerBase | None = None) -> int:
    """
    Get the token count for the input. If tokenizer provided, i.e. pre-trained
    tokenizer from embedding model used in vector database,
    then use tokenizer instead of tiktoken.
    """
    if tokenizer:
        return len(tokenizer.encode(input))
    else:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(input, disallowed_special=()))


def prune_docs(
    docs: list[Document], docs_token_budget: int, tokenizer: PreTrainedTokenizerBase | None = None
) -> tuple[list[Document], int]:
    """Prune the docs to stay within the docs_token_budget.

    Parameters
    ----------
    docs
        The list of documents retrieved
    docs_token_budget
        The number of tokens to allow the docs to consume
    tokenizer
        The tokenizer to use to calculate the token count of the docs

    Returns
    -------
    docs
        The pruned list of docs
    docs_token_count
        The token count used by the docs
    """
    remaining_docs_token_budget = docs_token_budget
    num_docs_to_keep = 0
    for doc in docs:
        doc_token_count = get_token_count(doc.page_content, tokenizer)
        if doc_token_count > remaining_docs_token_budget:
            break
        remaining_docs_token_budget -= doc_token_count
        num_docs_to_keep += 1
    docs = docs[:num_docs_to_keep]
    docs_token_count = docs_token_budget - remaining_docs_token_budget
    return (docs, docs_token_count)


def get_excess_token_budget(
    context_size: int | None,
    output_tokens: int | None,
    system_prompt: str | None,
    user_prompt: str,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> int:
    context_size = context_size or DEFAULT_CONTEXT_SIZE
    if system_prompt is None:
        system_prompt = ""
    input_tokens_excluding_docs = get_token_count(
        "\n".join([system_prompt, user_prompt]), tokenizer=tokenizer
    )
    output_tokens = output_tokens or context_size // DEFAULT_CONTEXT_TO_COMPLETION_SIZE_RATIO
    excess_token_budget = context_size - output_tokens - input_tokens_excluding_docs
    # This can be negative because the user is requesting too large max_completion_length for
    # their prompt given the context size of the LLM. We will prune out all history except the most
    # recent message and all docs in this case. The LLM will likely return an error message and
    # the user can reduce the max_completion_length and retry. Some LLMs may just limit the
    # output tokens. Not much we can do to fix this case. We could try to automatically reduce
    # the max_completion_length but it may be better to transparently error and let the user
    # decide what to do.
    return max(excess_token_budget, 0)


def get_tool_settings(llm_settings: dict[str, Any]) -> dict[str, Any]:
    openai_settings: dict[str, Any] = {}
    if llm_settings.get("tools"):
        # Parse to Tool objects once (handles both OpenAI and DataRobot formats)
        tools = []
        for t in llm_settings.pop("tools"):
            if "function" in t and isinstance(t["function"], dict):
                # OpenAI format: {'type': 'function', 'function': {...}}
                tools.append(Tool.from_openai_tool_dict(t))
            else:
                # DataRobot format: {'name': '...', 'description': '...', 'parameters': {...}}
                tools.append(Tool(**t))
        # Convert all to OpenAI format
        openai_settings["tools"] = [t.to_openai_tool_dict() for t in tools]
        # Set tool_choice based on required tools
        required_tools = [t for t in tools if t.required]
        if len(required_tools) > 1:
            raise ValueError(gettext("Only one required tool is allowed for OpenAI models."))
        elif len(required_tools) == 1:
            openai_settings["tool_choice"] = {
                "type": "function",
                "function": {"name": required_tools[0].name},
            }
        else:
            openai_settings["tool_choice"] = "auto"
    return openai_settings


def prepend_system_message_to_messages(
    messages: Sequence[ChatCompletionMessageParam], system_prompt: str, docs: Sequence[Document]
) -> list[ChatCompletionMessageParam]:
    """
    Given an existing message payload, insert a system message in the beginning.
    If vector database documents are specified, incorporate them in the system message as well.

    Parameters
    ----------
    messages
        The chat messages to submit to the LLM, including the chat history so far
        and the new user prompt.
    system_prompt
        The DataRobot LLM settings specified by the user.
    docs
        The documents retrieved from a vector database.

    Returns
    -------
    The updated message payload, including the system message.
    """
    system_prompt = add_docs_to_system_prompt(system_prompt, docs)
    result = [m for m in messages]
    if system_prompt:
        result.insert(0, {"role": "system", "content": system_prompt})
    return result


def get_settings_and_system_prompt(llm_settings: dict[str, Any]) -> tuple[dict[str, Any], str]:
    # Map llm_settings to the custom model LLM settings.
    llm_settings = copy(llm_settings)
    custom_model_settings = {}
    system_prompt = ""
    # Settings that require custom handling.
    if "system_prompt" in llm_settings:
        system_prompt = llm_settings.pop("system_prompt") or ""
    # Translate the DataRobot internal naming back to OpenAI naming.
    if "max_completion_length" in llm_settings:
        llm_settings["max_tokens"] = llm_settings.pop("max_completion_length")
    # Already used to load the custom model information when initializing the model
    llm_settings.pop("validation_id", None)
    # This setting is only used for pruning documents supplied to the LLM when
    # a vector database is associated with the LLM blueprint.
    llm_settings.pop("external_llm_context_size", None)
    # Transfer other llm_settings "as is", if specified.
    custom_model_settings.update(llm_settings)
    return (custom_model_settings, system_prompt)
