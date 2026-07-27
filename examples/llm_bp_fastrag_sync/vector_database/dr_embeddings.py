# Copyright 2023 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
# Released under the terms of DataRobot Tool and Utility Agreement.
from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np
import sentence_transformers  # noqa: F401
import torch
from langchain_core.runnables.config import run_in_executor
from sentence_transformers.models import Transformer
from vector_database.custom_model_embeddings import CustomModelEmbeddingClient
from vector_database.custom_model_embeddings import EmbeddingStage
from vector_database.entities import DeviceForNNComputationSetting

DEFAULT_BATCH_SIZE = 128


def get_device() -> str:
    device_setting = DeviceForNNComputationSetting()
    return device_setting.device.value


def _detect_tokenizer_architecture_mismatch(
    model_name_or_path: str, architectures: list[str], tokenizer_class: str
) -> bool:
    """Detect if there is a mismatch between architecture and tokenizer."""
    if not model_name_or_path or not architectures or (not tokenizer_class):
        return False
    if "multilingual-e5-small" in model_name_or_path:
        return True
    # Check for BERT model with XLMRoberta tokenizer mismatch
    is_bert_model = any(("BertModel" in arch for arch in architectures))
    is_xlm_tokenizer = tokenizer_class == "XLMRobertaTokenizer"
    return is_bert_model and is_xlm_tokenizer


def patch_sentence_transformer() -> None:

    def forward(cls: Any, features: dict[str, Any]) -> dict[str, Any]:
        """Return token_embeddings, cls_token."""
        trans_features = {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
        }
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]
        # Patch multilingual-e5-small model as e5-team messed tokenizer and model configs
        # and token_type_ids is required
        if _detect_tokenizer_architecture_mismatch(
            cls.auto_model.config._name_or_path,
            cls.auto_model.config.architectures,
            cls.auto_model.config.tokenizer_class,
        ):
            trans_features["token_type_ids"] = features["attention_mask"]
        output_states = cls.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]
        features.update(
            {"token_embeddings": output_tokens, "attention_mask": features["attention_mask"]}
        )
        if cls.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if (
                len(output_states) < 3
            ):  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1
            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})
        return features

    # Originally this patch was added because the multilingual-e5-small model resulted in errors.
    # With an upgrade of sentence-transformers, the model just runs fine without the patch. The
    # embeddings change however, so the patch is preserved.
    Transformer.forward = forward  # type: ignore[method-assign, assignment]


patch_sentence_transformer()


class EmbeddingModelLoadingError(Exception):
    """Error loading the embedding model from model path."""


def get_sentence_transformer(model_path: str) -> sentence_transformers.SentenceTransformer:
    device = torch.device(get_device())
    onnx_provider = "CUDAExecutionProvider" if device.type == "cuda" else "CPUExecutionProvider"
    return sentence_transformers.SentenceTransformer(
        model_path,
        cache_folder=model_path,
        backend="onnx",
        model_kwargs={"provider": onnx_provider, "file_name": "model_quantized.onnx"},
    )


def get_embedder(
    client: sentence_transformers.SentenceTransformer | CustomModelEmbeddingClient,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Embeddings:
    try:
        if isinstance(client, sentence_transformers.SentenceTransformer):
            # Embedding L2 normalization is hardcoded to True.
            # Making it configurable would involve storing it in the database
            # to ensure consistency between ingest-time and inference-time embeddings.
            return DataRobotEmbeddings(
                client=client,
                encode_kwargs={
                    "batch_size": batch_size,
                    "normalize_embeddings": True,
                    "show_progress_bar": False,
                    "device": torch.device(get_device()).type,
                },
            )
        elif isinstance(client, CustomModelEmbeddingClient):
            return CustomModelEmbeddings(client=client)
        else:
            raise EmbeddingModelLoadingError("Invalid client")
    except Exception as e:
        raise EmbeddingModelLoadingError from e


class Embeddings(ABC):
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError

    async def aembed_documents(self, texts: list[str]) -> np.ndarray:
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> np.ndarray:
        return await run_in_executor(None, self.embed_query, text)


class DataRobotEmbeddings(Embeddings):
    """Langchain interface to DataRobot embedding models."""

    def __init__(
        self,
        client: sentence_transformers.SentenceTransformer,
        encode_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.client = client
        self.encode_kwargs = encode_kwargs or {}

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Compute doc embeddings using provided client.

        Parameters
        ----------
        texts
            List of chunks of text to embed

        Returns
        -------
            List of embedded documents
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        return self.client.encode(texts, **self.encode_kwargs)

    def embed_query(self, text: str) -> np.ndarray:
        """Compute query embeddings using provided client.

        Parameters
        ----------
        text
            Text of embed via featurization

        Returns
        -------
            The embedded query
        """
        text = text.replace("\n", " ")
        return self.client.encode(text, **self.encode_kwargs)

    async def __aenter__(self) -> Embeddings:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        pass


class CustomModelEmbeddings(Embeddings):
    """Langchain interface to DataRobot embedding models."""

    def __init__(self, client: CustomModelEmbeddingClient, **kwargs: Any):
        self.client = client

    async def aembed_documents(self, texts: list[str]) -> np.ndarray:
        """Asynchronous Embed search docs."""
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        return await self.client.encode(texts, EmbeddingStage.indexing)

    async def aembed_query(self, text: str) -> np.ndarray:
        """Asynchronous Embed query text."""
        text = text.replace("\n", " ")
        response_2d_array = await self.client.encode([text], EmbeddingStage.prompting)
        return response_2d_array.flatten()

    async def __aenter__(self) -> Embeddings:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        await self.client.close()
