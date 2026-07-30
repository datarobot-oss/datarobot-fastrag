# Copyright 2023 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from enum import StrEnum
VECTOR_DATABASE_ASSETS_KEY = 'vector_database_assets'
CUSTOM_MODEL_LLM_BLUEPRINT_CONFIG_DOC = 'custom_model_llm_blueprint_config.json'
LLM_PROVIDER_GUARD_METRIC_NAME_COL_NAME = 'LLM_PROVIDER_GUARD_METRIC_{col_idx}_NAME'
LLM_PROVIDER_GUARD_METRIC_TRIGGERED_COL_NAME = 'LLM_PROVIDER_GUARD_METRIC_{col_idx}_TRIGGERED'
LLM_PROVIDER_GUARD_METRIC_VALUE_COL_NAME = 'LLM_PROVIDER_GUARD_METRIC_{col_idx}_VALUE'

class VectorDatabaseCustomModelObjects(StrEnum):
    # Contains concatenated texts of chunks
    TEXT_CHUNKS = f'{VECTOR_DATABASE_ASSETS_KEY}/chunks.txt'
    # Contains concatenated texts of chunks compressed with zlib
    TEXT_CHUNKS_ZLIB = f'{VECTOR_DATABASE_ASSETS_KEY}/chunks.zlib'
    # Metadata for chunks as pandas categorical fields + chunk sizes
    TEXT_METADATA = f'{VECTOR_DATABASE_ASSETS_KEY}/chunks.pkl'
    TEXT_METADATA_SQLITE = f'{VECTOR_DATABASE_ASSETS_KEY}/chunks.db'
    # Faiss serialized index file
    FAISS_INDEX = f'{VECTOR_DATABASE_ASSETS_KEY}/index.faiss'
    # Embedding model folder
    EMBEDDING_MODEL = f'{VECTOR_DATABASE_ASSETS_KEY}/embedding_model'

class CustomModelVectorDatabaseRetrievers(StrEnum):
    SINGLE_LOOKUP_RETRIEVER = 'SINGLE_LOOKUP_RETRIEVER'
    CONVERSATIONAL_RETRIEVER = 'CONVERSATIONAL_RETRIEVER'
    MULTI_STEP_RETRIEVER = 'MULTI_STEP_RETRIEVER'

class CustomModelRuntimeParameters(StrEnum):
    OPENAI_API_KEY = 'OPENAI_API_KEY'
    OPENAI_API_BASE = 'OPENAI_API_BASE'
    OPENAI_API_DEPLOYMENT_ID = 'OPENAI_API_DEPLOYMENT_ID'
    OPENAI_API_VERSION = 'OPENAI_API_VERSION'
    GOOGLE_SERVICE_ACCOUNT = 'GOOGLE_SERVICE_ACCOUNT'
    GOOGLE_REGION = 'GOOGLE_REGION'
    AWS_ACCOUNT = 'AWS_ACCOUNT'
    AWS_REGION = 'AWS_REGION'
    PROMPT_COLUMN_NAME = 'PROMPT_COLUMN_NAME'
    VECTOR_DATABASE_ID = 'VECTOR_DATABASE_ID'
    VECTOR_DATABASE_FAMILY_ID = 'VECTOR_DATABASE_FAMILY_ID'
    LLM_BLUEPRINT_ID = 'LLM_BLUEPRINT_ID'
    LLM_BLUEPRINT_ID_COLUMN_NAME = 'LLM_BLUEPRINT_ID_COLUMN_NAME'
    ENABLE_LLM_BLUEPRINT_ID_COLUMN = 'ENABLE_LLM_BLUEPRINT_ID_COLUMN'
    ENABLE_CITATION_COLUMNS = 'ENABLE_CITATION_COLUMNS'
    LLM_ID = 'LLM_ID'
    LLM_TEST_SUITE_ID = 'LLM_TEST_SUITE_ID'
    DEVICE_FOR_NEURAL_NETWORK_COMPUTATIONS = 'DEVICE_FOR_NEURAL_NETWORK_COMPUTATIONS'
    PLAYGROUND_ID = 'PLAYGROUND_ID'
    ANTHROPIC_API_KEY = 'ANTHROPIC_API_KEY'
    COHERE_API_KEY = 'COHERE_API_KEY'
    TOGETHERAI_API_KEY = 'TOGETHERAI_API_KEY'
    GROQ_API_KEY = 'GROQ_API_KEY'
    CEREBRAS_API_KEY = 'CEREBRAS_API_KEY'
    VECTOR_DATABASE_DEPLOYMENT_ID = 'VECTOR_DATABASE_DEPLOYMENT_ID'
    CUSTOM_MODEL_WORKERS = 'CUSTOM_MODEL_WORKERS'
    DRUM_SERVER_TYPE = 'DRUM_SERVER_TYPE'
    DRUM_GUNICORN_WORKER_CLASS = 'DRUM_GUNICORN_WORKER_CLASS'
    DRUM_WORKER_CONNECTIONS = 'DRUM_WORKER_CONNECTIONS'
    DRUM_CLIENT_REQUEST_TIMEOUT = 'DRUM_CLIENT_REQUEST_TIMEOUT'

    @classmethod
    def do_not_override_parameters(cls) -> set[str]:
        """Parameters that are expected to be freshly set by the custom model version creation job
        because they identify the new VDB version and LLM blueprint and should not be overridden
        when performing a version update.
        """
        return {cls.VECTOR_DATABASE_ID.value, cls.VECTOR_DATABASE_FAMILY_ID.value, cls.VECTOR_DATABASE_DEPLOYMENT_ID.value, cls.LLM_BLUEPRINT_ID.value, cls.LLM_TEST_SUITE_ID.value, cls.LLM_ID.value, cls.PLAYGROUND_ID.value}

class CustomModelNonEnglishEmbeddingModelPaths(StrEnum):
    MULTILINGUAL_E5_BASE = 'intfloat/multilingual-e5-base-fp16'
    MULTILINGUAL_E5_SMALL = 'intfloat/multilingual-e5-small-fp16'
    SUP_SIMCSE_JA_BASE = 'cl-nagoya/sup-simcse-ja-base-fp16'