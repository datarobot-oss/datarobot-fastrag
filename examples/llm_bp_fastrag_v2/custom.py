# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
import os
from collections.abc import AsyncIterator
from typing import Any
import pandas as pd
from fastrag import RuntimeParameters
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import CompletionCreateParams
from custom_model_enum import CustomModelRuntimeParameters
from custom_model_inference import llm_blueprint_chat
from custom_model_inference import llm_blueprint_score
from custom_model_inference import populate_monitoring_columns
from custom_model_inference import populate_provider_llm_guard_columns
from custom_model_loader import load_custom_model_llm_blueprint_config
from custom_model_loader import load_llm_blueprint_model
from language_models.language_model_interface import LLMBlueprintModel

async def load_model(*args: Any, **kwargs: Any) -> LLMBlueprintModel:
    """Load the model object to be used in `score`. This custom model hook is called at runtime."""
    return load_llm_blueprint_model()

async def score(data: pd.DataFrame, model: LLMBlueprintModel, **kwargs: Any) -> pd.DataFrame:
    config = load_custom_model_llm_blueprint_config()
    result_list, docs_list, provider_llm_guards_list, query_embeddings = await llm_blueprint_score(data, model, config)
    target_name: str = (kwargs.get('target_name') or '').replace('"', '')
    full_result_dict: dict[str, list] = {target_name: result_list}
    if config.vector_database_settings is not None:
        populate_monitoring_columns(full_result_dict, docs_list, query_embeddings, config)
    populate_provider_llm_guard_columns(provider_llm_guards_list=provider_llm_guards_list, full_result_dict=full_result_dict)
    llm_blueprint_id_column_name = RuntimeParameters.get(CustomModelRuntimeParameters.LLM_BLUEPRINT_ID_COLUMN_NAME.value)
    if llm_blueprint_id_column_name is not None and RuntimeParameters.get(CustomModelRuntimeParameters.ENABLE_LLM_BLUEPRINT_ID_COLUMN.value):
        full_result_dict[llm_blueprint_id_column_name] = [config.llm_blueprint_id] * len(result_list)
    return pd.DataFrame(full_result_dict)

async def chat(completion_create_params: CompletionCreateParams, model: LLMBlueprintModel, **kwargs: Any) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
    config = load_custom_model_llm_blueprint_config()
    return await llm_blueprint_chat(model, config, completion_create_params)