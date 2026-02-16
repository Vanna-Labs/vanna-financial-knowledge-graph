"""
LLM and Embedding Providers

Provider-agnostic interfaces for LLM and embedding operations.

Modules:
    base: Abstract provider interfaces
    llm/: LLM provider implementations
    embedding/: Embedding provider implementations

Supported LLM Providers:
    - OpenAI (gpt-4o, gpt-4o-mini) via LangChain

Supported Embedding Providers:
    - OpenAI (text-embedding-3-large) via LangChain

Design:
    - All providers implement abstract interfaces (LLMProvider, EmbeddingProvider)
    - Lazy import to avoid requiring all dependencies
    - Structured output support via LangChain's with_structured_output

Example:
    >>> from vanna_kg.providers import LLMProvider, EmbeddingProvider
    >>> from vanna_kg.providers.llm import OpenAILLMProvider
    >>> from vanna_kg.providers.embedding import OpenAIEmbeddingProvider

See: docs/architecture/PYTHON_PACKAGE_DESIGN.md Section 7
"""

from vanna_kg.providers.base import EmbeddingProvider, LLMProvider

__all__ = ["LLMProvider", "EmbeddingProvider"]
