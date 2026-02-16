"""
Abstract Provider Interfaces

Base classes for LLM and embedding providers.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a completion."""
        ...

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
    ) -> T:
        """Generate a structured response matching the schema."""
        ...

    @abstractmethod
    def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion. Returns an async iterator."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Current model name."""
        ...


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        ...

    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding dimensions."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Current model name."""
        ...
