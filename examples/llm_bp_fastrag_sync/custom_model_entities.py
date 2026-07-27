# Copyright 2023 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from typing import Any

from custom_model_enum import CustomModelVectorDatabaseRetrievers
from language_models.language_model_interface import CustomModelLLMCredentialType
from pydantic import BaseModel
from pydantic import ConfigDict


class ExternalDeploymentParameters(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prompt_column_name: str | None
    target_column_name: str | None
    prediction_api_url: str
    chat_api_url: str | None
    datarobot_key: str | None
    input_type: str
    model_type: str
    association_id_column: str | None
    supports_chat_api: bool
    chat_model_id: str | None


class CustomModelVectorDatabaseSettings(BaseModel):
    number_of_documents_to_retrieve: int
    retriever: CustomModelVectorDatabaseRetrievers = (
        CustomModelVectorDatabaseRetrievers.SINGLE_LOOKUP_RETRIEVER
    )
    add_neighbor_chunks: bool = False
    # Hold the list of columns(including VDB's metadata) defined
    # as CustomModels citation fields
    enabled_citations_fields: list[str] = []
    # Only defined for external VD
    external_vector_database_parameters: ExternalDeploymentParameters | None = None
    # Only defined for internal VDB using external custom model
    external_embedding_model_parameters: ExternalDeploymentParameters | None = None


class CustomModelLLMBlueprintConfig(BaseModel):
    llm_blueprint_id: str
    llm_id: str
    llm_settings: dict[str, Any]
    context_size: int
    llm_credential_type: CustomModelLLMCredentialType
    external_custom_model_llm_settings: ExternalDeploymentParameters | None = None
    vector_database_settings: CustomModelVectorDatabaseSettings | None = None
