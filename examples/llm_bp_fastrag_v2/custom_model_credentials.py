# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
import os
from fastrag import RuntimeParameters
from pydantic import AnyHttpUrl
from custom_model_enum import CustomModelRuntimeParameters
from language_models.custom_model_llm import CustomModelLLMError
from language_models.language_model_interface import CustomModelLLMCredentialType
from language_models.language_model_interface import LanguageModelCredentials

class AzureOpenAICredentials(LanguageModelCredentials):
    """Credentials necessary to access Azure OpenAI models."""
    credential_type: CustomModelLLMCredentialType = CustomModelLLMCredentialType.AZURE
    api_type: str = 'azure'
    api_base: AnyHttpUrl
    api_version: str
    api_key: str
    deployment_id: str | None = None
    region: str | None = None

class GoogleVertexAICredentials(LanguageModelCredentials):
    """Credentials necessary to access Vertex AI models."""
    credential_type: CustomModelLLMCredentialType = CustomModelLLMCredentialType.GOOGLE
    region: str
    service_account_info: dict[str, str]

class AmazonBedrockCredentials(LanguageModelCredentials):
    """Credentials necessary to access Amazon Bedrock models."""
    credential_type: CustomModelLLMCredentialType = CustomModelLLMCredentialType.AMAZON
    region: str
    access_key_id: str
    secret_access_key: str
    session_token: str | None = None

class AnthropicCredentials(LanguageModelCredentials):
    """Credentials necessary to access Anthropic (first-party) models."""
    credential_type: CustomModelLLMCredentialType = CustomModelLLMCredentialType.ANTHROPIC
    api_key: str

class CohereCredentials(LanguageModelCredentials):
    """Credentials necessary to access Cohere (first-party) models."""
    credential_type: CustomModelLLMCredentialType = CustomModelLLMCredentialType.COHERE
    api_key: str

class TogetherAICredentials(LanguageModelCredentials):
    """Credentials necessary to access TogetherAI models."""
    credential_type: CustomModelLLMCredentialType = CustomModelLLMCredentialType.TOGETHERAI
    api_key: str

class GroqCredentials(LanguageModelCredentials):
    """Credentials necessary to access Groq models."""
    credential_type: CustomModelLLMCredentialType = CustomModelLLMCredentialType.GROQ
    api_key: str

class CerebrasCredentials(LanguageModelCredentials):
    """Credentials necessary to access Cerebras models."""
    credential_type: CustomModelLLMCredentialType = CustomModelLLMCredentialType.CEREBRAS
    api_key: str

def ensure_required_runtime_parameter_set(parameter_name: str) -> None:
    """Ensure that the specified runtime parameter exists and is not empty."""
    parameter = RuntimeParameters.get(parameter_name)
    if not parameter:
        raise CustomModelLLMError(f'{parameter_name} runtime parameter is required and cannot be empty')

def load_openai_credentials() -> AzureOpenAICredentials:
    """Load OpenAI-compatible credentials from the runtime parameters of the model.

    Azure OpenAI and every other OpenAI-compatible provider (Nebius, Groq, TogetherAI, self-hosted
    vLLM, ...) share the same `OPENAI_API_*` parameters. Azure is not a separate credential; it is
    the OpenAI-compatible variant that additionally carries a `deployment_id`. litellm treats a
    non-null `deployment_id` as a directive to rewrite the request as an Azure
    `/openai/deployments/<id>/chat/completions?api-version=` call, which other providers 404, so the
    presence of a deployment id is exactly what distinguishes Azure from generic OpenAI here.
    """
    api_key = RuntimeParameters.get(CustomModelRuntimeParameters.OPENAI_API_KEY.value)
    if not api_key or 'apiToken' not in api_key:
        raise CustomModelLLMError('OPENAI_API_KEY runtime parameter is missing or invalid. Ensure that the parameter is assigned a credential of type "API Token" as its value.')
    ensure_required_runtime_parameter_set(CustomModelRuntimeParameters.OPENAI_API_BASE.value)
    deployment_id = RuntimeParameters.get(CustomModelRuntimeParameters.OPENAI_API_DEPLOYMENT_ID.value)
    if deployment_id:
        # Azure needs a real API version to route the deployment request.
        ensure_required_runtime_parameter_set(CustomModelRuntimeParameters.OPENAI_API_VERSION.value)
    # A blank version is dropped downstream; non-Azure providers ignore it anyway.
    # Only Azure carries a deployment id; normalise an unset/blank value to None.
    credentials = AzureOpenAICredentials(api_type='azure', api_base=RuntimeParameters.get(CustomModelRuntimeParameters.OPENAI_API_BASE.value), api_version=RuntimeParameters.get(CustomModelRuntimeParameters.OPENAI_API_VERSION.value) or '', deployment_id=deployment_id or None, api_key=api_key['apiToken'])
    return credentials

def load_google_credentials() -> GoogleVertexAICredentials:
    """Load Google credentials from the runtime parameters of the model."""
    service_account = RuntimeParameters.get(CustomModelRuntimeParameters.GOOGLE_SERVICE_ACCOUNT.value)
    if not service_account or 'gcpKey' not in service_account:
        raise CustomModelLLMError('GOOGLE_SERVICE_ACCOUNT runtime parameter is missing or invalid. Ensure that the parameter is assigned a credential of type "Google Cloud Service Account" as its value.')
    ensure_required_runtime_parameter_set(CustomModelRuntimeParameters.GOOGLE_REGION.value)
    credentials = GoogleVertexAICredentials(region=RuntimeParameters.get(CustomModelRuntimeParameters.GOOGLE_REGION.value), service_account_info=service_account['gcpKey'])
    return credentials

def load_amazon_credentials() -> AmazonBedrockCredentials:
    """Load Amazon credentials from the runtime parameters of the model."""
    aws_account = RuntimeParameters.get(CustomModelRuntimeParameters.AWS_ACCOUNT.value)
    if not aws_account:
        raise CustomModelLLMError('AWS_ACCOUNT runtime parameter is missing or invalid. Ensure that the parameter is assigned a credential of type "AWS" as its value.')
    ensure_required_runtime_parameter_set(CustomModelRuntimeParameters.AWS_REGION.value)
    credentials = AmazonBedrockCredentials(access_key_id=aws_account['awsAccessKeyId'], secret_access_key=aws_account['awsSecretAccessKey'], session_token=aws_account.get('awsSessionToken'), region=RuntimeParameters.get(CustomModelRuntimeParameters.AWS_REGION.value))
    return credentials

def load_anthropic_credentials() -> AnthropicCredentials:
    """Load Anthropic (first-party) credentials from the runtime parameters of the model."""
    api_key = RuntimeParameters.get(CustomModelRuntimeParameters.ANTHROPIC_API_KEY.value)
    if not api_key or 'apiToken' not in api_key:
        raise CustomModelLLMError('ANTHROPIC_API_KEY runtime parameter is missing or invalid. Ensure that the parameter is assigned a credential of type "API Token" as its value.')
    return AnthropicCredentials(api_key=api_key['apiToken'])

def load_cohere_credentials() -> CohereCredentials:
    """Load Cohere (first-party) credentials from the runtime parameters of the model."""
    api_key = RuntimeParameters.get(CustomModelRuntimeParameters.COHERE_API_KEY.value)
    if not api_key or 'apiToken' not in api_key:
        raise CustomModelLLMError('COHERE_API_KEY runtime parameter is missing or invalid. Ensure that the parameter is assigned a credential of type "API Token" as its value.')
    return CohereCredentials(api_key=api_key['apiToken'])

def load_togetherai_credentials() -> TogetherAICredentials:
    """Load TogetherAI credentials from the runtime parameters of the model."""
    api_key = RuntimeParameters.get(CustomModelRuntimeParameters.TOGETHERAI_API_KEY.value)
    if not api_key or 'apiToken' not in api_key:
        raise CustomModelLLMError('TOGETHERAI_API_KEY runtime parameter is missing or invalid. Ensure that the parameter is assigned a credential of type "API Token" as its value.')
    return TogetherAICredentials(api_key=api_key['apiToken'])

def load_groq_credentials() -> GroqCredentials:
    """Load Groq credentials from the runtime parameters of the model."""
    api_key = RuntimeParameters.get(CustomModelRuntimeParameters.GROQ_API_KEY.value)
    if not api_key or 'apiToken' not in api_key:
        raise CustomModelLLMError('GROQ_API_KEY runtime parameter is missing or invalid. Ensure that the parameter is assigned a credential of type "API Token" as its value.')
    return GroqCredentials(api_key=api_key['apiToken'])

def load_cerebras_credentials() -> CerebrasCredentials:
    """Load Cerebras credentials from the runtime parameters of the model."""
    api_key = RuntimeParameters.get(CustomModelRuntimeParameters.CEREBRAS_API_KEY.value)
    if not api_key or 'apiToken' not in api_key:
        raise CustomModelLLMError('CEREBRAS_API_KEY runtime parameter is missing or invalid. Ensure that the parameter is assigned a credential of type "API Token" as its value.')
    return CerebrasCredentials(api_key=api_key['apiToken'])

def get_datarobot_api_token() -> str:
    """Retrieve the DataRobot API token of the model's creator."""
    result = os.environ.get('DATAROBOT_API_TOKEN') or os.environ.get('API_TOKEN')
    if not result:
        raise CustomModelLLMError('DataRobot API token not found')
    return result

def get_datarobot_authorization_header() -> str:
    """Retrieve the value of the `Authorization` header for the DataRobot API."""
    token = get_datarobot_api_token()
    header = f'Bearer {token}'
    return header