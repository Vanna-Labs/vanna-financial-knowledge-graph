"""
LLM Provider Implementations

Modules:
    openai: OpenAI provider (gpt-4o, gpt-4o-mini)
    anthropic: Anthropic provider (claude-sonnet-4-5, claude-haiku-3-5)
    google: Google provider (gemini-2.5-flash)
    local: Local provider (Ollama, llama.cpp)

Each provider implements the LLMProvider interface with:
    - generate(): Text completion
    - generate_structured(): Structured output (JSON schema)
    - stream(): Streaming completion

Structured Output Implementation:
    - OpenAI: LangChain with_structured_output
    - Anthropic: tool_use with input_schema
    - Google: response_schema
    - Local: JSON mode with schema in prompt

Example:
    >>> from zomma_kg.providers.llm import OpenAILLMProvider
    >>> provider = OpenAILLMProvider(model="gpt-4o")
    >>> response = await provider.generate("Hello!")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zomma_kg.providers.llm.openai import OpenAILLMProvider


def __getattr__(name: str):
    """Lazy import of providers to avoid requiring all dependencies."""
    if name == "OpenAILLMProvider":
        from zomma_kg.providers.llm.openai import OpenAILLMProvider
        return OpenAILLMProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OpenAILLMProvider"]
