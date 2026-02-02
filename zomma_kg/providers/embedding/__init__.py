"""
Embedding Provider Implementations

Modules:
    openai: OpenAI embeddings (text-embedding-3-large)
    voyage: Voyage embeddings (voyage-finance-2)
    local: Local embeddings (sentence-transformers)

Each provider implements the EmbeddingProvider interface with:
    - embed(): Batch embedding generation
    - embed_single(): Single text embedding
    - dimensions: Vector dimensionality
    - model_name: Current model identifier

Batch Processing:
    - All providers support batching
    - Rate limit handling with exponential backoff
    - Sync methods wrapped with asyncio.to_thread for async compatibility

Example:
    >>> from zomma_kg.providers.embedding import OpenAIEmbeddingProvider
    >>> provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
    >>> vectors = await provider.embed(["Hello", "World"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zomma_kg.providers.embedding.openai import OpenAIEmbeddingProvider


def __getattr__(name: str):
    """Lazy import of providers to avoid requiring all dependencies."""
    if name == "OpenAIEmbeddingProvider":
        from zomma_kg.providers.embedding.openai import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OpenAIEmbeddingProvider"]
