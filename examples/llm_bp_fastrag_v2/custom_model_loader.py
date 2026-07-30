# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
import datarobot as dr
from custom_model_credentials import get_datarobot_authorization_header
from custom_model_credentials import load_amazon_credentials
from custom_model_credentials import load_anthropic_credentials
from custom_model_credentials import load_cerebras_credentials
from custom_model_credentials import load_cohere_credentials
from custom_model_credentials import load_google_credentials
from custom_model_credentials import load_groq_credentials
from custom_model_credentials import load_openai_credentials
from custom_model_credentials import load_togetherai_credentials
from custom_model_entities import CustomModelLLMBlueprintConfig
from custom_model_enum import CUSTOM_MODEL_LLM_BLUEPRINT_CONFIG_DOC
from custom_model_inference import load_stored_vector_database_as_vector_store
from deployment_access import DataRobotDeploymentCredentials
from language_models.custom_model_llm import CustomModelLLM
from language_models.custom_model_llm import CustomModelLLMError
from language_models.language_model_interface import CustomModelLLMCredentialType
from language_models.language_model_interface import LanguageModelCredentials
from language_models.language_model_interface import LLMBlueprintModel
from language_models.llm_gateway import LLMGatewayCredentials
from language_models.llm_gateway import LLMGatewayLanguageModel
from language_models.llm_gateway import LLMWorkloads

def get_llm_credentials(config: CustomModelLLMBlueprintConfig) -> LanguageModelCredentials | DataRobotDeploymentCredentials:
    """Get the LLM credentials.

    We try to load user credentials from runtime parameters,
    and always use the LLM gateway if it is enabled (with or without user credentials).
    If it's disabled, then credentials are required.
    Exception: DR deployed LLMs are not supported by the LLM Gateway and are handled separately.
    """
    if config.llm_credential_type == CustomModelLLMCredentialType.DATAROBOT:
        if config.external_custom_model_llm_settings is None:
            raise CustomModelLLMError('Custom Model LLM settings are not available')
        return DataRobotDeploymentCredentials(datarobot_key=config.external_custom_model_llm_settings.datarobot_key, authorization_header=get_datarobot_authorization_header())
    user_credentials: LanguageModelCredentials | None = None
    try:
        match config.llm_credential_type:
            case CustomModelLLMCredentialType.OPENAI:
                user_credentials = load_openai_credentials()
            case CustomModelLLMCredentialType.GOOGLE:
                user_credentials = load_google_credentials()
            case CustomModelLLMCredentialType.AMAZON:
                user_credentials = load_amazon_credentials()
            case CustomModelLLMCredentialType.ANTHROPIC:
                user_credentials = load_anthropic_credentials()
            case CustomModelLLMCredentialType.COHERE:
                user_credentials = load_cohere_credentials()
            case CustomModelLLMCredentialType.TOGETHERAI:
                user_credentials = load_togetherai_credentials()
            case CustomModelLLMCredentialType.GROQ:
                user_credentials = load_groq_credentials()
            case CustomModelLLMCredentialType.CEREBRAS:
                user_credentials = load_cerebras_credentials()
            case _:
                raise Exception(f'Credential type {config.llm_credential_type} is not supported.')
    except (CustomModelLLMError, ValueError):
        pass
    client = dr.Client()
    credentials = LLMGatewayCredentials(base_url=client.endpoint.rstrip('/') + '/genai/llmgw', api_token=client.token, client_id=LLMWorkloads.CUSTOM_MODEL, llm_id=str(config.llm_id), user_credentials=user_credentials)
    return credentials

def load_llm_blueprint_model() -> LLMBlueprintModel:
    config = load_custom_model_llm_blueprint_config()
    vector_store = load_stored_vector_database_as_vector_store(config)
    credentials = get_llm_credentials(config)
    if isinstance(credentials, DataRobotDeploymentCredentials) and config.external_custom_model_llm_settings is not None:
        return LLMBlueprintModel(language_model=CustomModelLLM(credentials=credentials, prediction_api_url=config.external_custom_model_llm_settings.prediction_api_url, chat_api_url=config.external_custom_model_llm_settings.chat_api_url, model_type=config.external_custom_model_llm_settings.model_type, input_type=config.external_custom_model_llm_settings.input_type, prompt_column_name=config.external_custom_model_llm_settings.prompt_column_name, target_column_name=config.external_custom_model_llm_settings.target_column_name, association_id_column=config.external_custom_model_llm_settings.association_id_column, supports_chat_api=config.external_custom_model_llm_settings.supports_chat_api, chat_model_id=config.external_custom_model_llm_settings.chat_model_id), vector_store=vector_store)  # type: ignore[arg-type]
    return LLMBlueprintModel(language_model=LLMGatewayLanguageModel(credentials), vector_store=vector_store)

def load_custom_model_llm_blueprint_config() -> CustomModelLLMBlueprintConfig:
    with open(CUSTOM_MODEL_LLM_BLUEPRINT_CONFIG_DOC) as llm_blueprint_config_handle:
        config_file_contents = llm_blueprint_config_handle.read()
        return CustomModelLLMBlueprintConfig.model_validate_json(config_file_contents)