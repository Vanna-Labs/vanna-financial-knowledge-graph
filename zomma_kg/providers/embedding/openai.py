"""
OpenAI Embedding Provider (LangChain-based)

Implements EmbeddingProvider interface using LangChain's OpenAIEmbeddings.

Supports:
    - Batch embedding generation (embed)
    - Single text embedding (embed_single)

Models:
    - text-embedding-3-large: 3072 dimensions, best quality
    - text-embedding-3-small: 1536 dimensions, faster/cheaper

Example:
    >>> provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
    >>> vectors = await provider.embed(["Hello world", "Goodbye world"])
    >>> print(len(vectors[0]))  # 3072 for large model
    3072

    >>> single = await provider.embed_single("Test text")
    >>> print(len(single))
    3072
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from zomma_kg.providers.base import EmbeddingProvider

if TYPE_CHECKING:
    from langchain_openai import OpenAIEmbeddings


# Model dimensions mapping
MODEL_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

DEFAULT_MODEL = "text-embedding-3-large"


def _get_openai_embeddings(
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> "OpenAIEmbeddings":
    """
    Get an OpenAIEmbeddings instance.

    Uses lazy import to avoid requiring langchain-openai unless actually used.

    Args:
        api_key: Optional API key. If not provided, uses OPENAI_API_KEY env var.
        model: Model name to use.

    Returns:
        OpenAIEmbeddings instance

    Raises:
        ImportError: If langchain-openai package is not installed
    """
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        raise ImportError(
            "OpenAI embedding provider requires the 'langchain-openai' package. "
            "Install with: pip install zomma-kg[openai]"
        )

    if api_key:
        from pydantic import SecretStr
        return OpenAIEmbeddings(model=model, api_key=SecretStr(api_key))
    return OpenAIEmbeddings(model=model)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider implementation using LangChain.

    Provides batch and single text embedding generation using
    OpenAI's embedding models via LangChain's OpenAIEmbeddings.

    Args:
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
        model: Model to use (default: "text-embedding-3-large")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = MODEL_DIMENSIONS.get(model, 3072)
        # Lazy initialization
        self._client: OpenAIEmbeddings | None = None

    def _get_client(self) -> "OpenAIEmbeddings":
        """Get or create the OpenAIEmbeddings client."""
        if self._client is None:
            self._client = _get_openai_embeddings(
                api_key=self._api_key,
                model=self._model,
            )
        return self._client

    @property
    def dimensions(self) -> int:
        """Embedding dimensions for the current model."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Current model name."""
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (same order as input)
        """
        if not texts:
            return []

        client = self._get_client()

        # LangChain's embed_documents is synchronous, run in thread pool
        embeddings = await asyncio.to_thread(client.embed_documents, texts)
        return embeddings

    async def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        client = self._get_client()

        # LangChain's embed_query is synchronous, run in thread pool
        embedding = await asyncio.to_thread(client.embed_query, text)
        return embedding

    def with_model(self, model: str) -> "OpenAIEmbeddingProvider":
        """
        Return a new provider instance with a different model.

        Args:
            model: New model name to use

        Returns:
            New OpenAIEmbeddingProvider with the specified model
        """
        return OpenAIEmbeddingProvider(api_key=self._api_key, model=model)
