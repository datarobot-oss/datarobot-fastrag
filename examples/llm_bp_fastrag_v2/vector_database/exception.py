# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.

class EmbeddingModelLoadingError(Exception):
    """Error loading the embedding model from model path."""

class ExternalEmbeddingError(Exception):
    """Exception to indicate that external embedding (OpenAI or Custom Model) is not working"""

class CustomModelEmbeddingError(ExternalEmbeddingError):
    """Raised when the custom model is unusable."""

class IncorrectResponseFromCustomModelError(ExternalEmbeddingError):
    """Raised when the custom model returned an incorrect response."""

class OpenAIEmbeddingModelError(ExternalEmbeddingError):
    """Raised when the OpenAI embedding model is unusable."""

class OpenAIEmbeddingModelCredentialError(OpenAIEmbeddingModelError):
    """Raised when OpenAI embedding model credential can't be loaded."""