"""
OpenAI LLM Provider (LangChain-based)

Implements LLMProvider interface using LangChain's ChatOpenAI.

Supports:
    - Text generation (generate)
    - Structured output with Pydantic schemas (generate_structured)
    - Streaming responses (stream)

Models:
    - gpt-4o: Best quality, recommended for extraction
    - gpt-4o-mini: Fast and cheap, good for verification

Example:
    >>> provider = OpenAILLMProvider(api_key="sk-...", model="gpt-4o")
    >>> response = await provider.generate("What is 2+2?")
    >>> print(response)
    "4"

    >>> from pydantic import BaseModel
    >>> class Answer(BaseModel):
    ...     value: int
    ...     explanation: str
    >>> result = await provider.generate_structured("What is 2+2?", Answer)
    >>> print(result.value)
    4
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, TypeVar

from vanna_kg.providers.base import LLMProvider

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel

T = TypeVar("T", bound="BaseModel")


def _get_chat_openai(
    api_key: str | None = None,
    model: str = "gpt-5.1",
    temperature: float = 0.0,
) -> "ChatOpenAI":
    """
    Get a ChatOpenAI instance.

    Uses lazy import to avoid requiring langchain-openai unless actually used.

    Args:
        api_key: Optional API key. If not provided, uses OPENAI_API_KEY env var.
        model: Model name to use.
        temperature: Sampling temperature.

    Returns:
        ChatOpenAI instance

    Raises:
        ImportError: If langchain-openai package is not installed
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "OpenAI provider requires the 'langchain-openai' package. "
            "Install with: pip install vanna-kg[openai]"
        )

    kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
    if api_key:
        kwargs["api_key"] = api_key

    return ChatOpenAI(**kwargs)


class OpenAILLMProvider(LLMProvider):
    """
    OpenAI LLM provider implementation using LangChain.

    Provides text generation, structured output, and streaming
    using LangChain's ChatOpenAI wrapper.

    Args:
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
        model: Model to use (default: "gpt-4o")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.1",
    ) -> None:
        self._api_key = api_key
        self._model = model
        # Lazy initialization - create client on first use
        self._client: ChatOpenAI | None = None

    def _get_client(self, temperature: float = 0.0) -> "ChatOpenAI":
        """Get or create the ChatOpenAI client."""
        if self._client is None or self._client.temperature != temperature:
            self._client = _get_chat_openai(
                api_key=self._api_key,
                model=self._model,
                temperature=temperature,
            )
        return self._client

    @property
    def model_name(self) -> str:
        """Current model name."""
        return self._model

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """
        Generate a text completion.

        Args:
            prompt: User prompt/question
            system: Optional system message for context
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response
        """
        from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

        base_client = _get_chat_openai(
            api_key=self._api_key,
            model=self._model,
            temperature=temperature,
        )
        client = base_client.bind(max_tokens=max_tokens)

        messages: list[BaseMessage] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        response = await client.ainvoke(messages)
        return str(response.content)

    async def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
    ) -> T:
        """
        Generate a structured response matching a Pydantic schema.

        Uses LangChain's with_structured_output for reliable
        structured output that conforms to the provided schema.

        Args:
            prompt: User prompt/question
            schema: Pydantic model class defining expected structure
            system: Optional system message

        Returns:
            Instance of schema class populated with generated values
        """
        from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

        client = _get_chat_openai(
            api_key=self._api_key,
            model=self._model,
            temperature=0.0,  # Deterministic for structured output
        )

        # Use LangChain's structured output feature
        structured_client = client.with_structured_output(schema)

        messages: list[BaseMessage] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        result = await structured_client.ainvoke(messages)
        return result  # type: ignore[return-value]

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a text completion token by token.

        Args:
            prompt: User prompt/question
            system: Optional system message

        Yields:
            Text chunks as they're generated
        """
        from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

        client = _get_chat_openai(
            api_key=self._api_key,
            model=self._model,
            temperature=0.0,
        )

        messages: list[BaseMessage] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        async for chunk in client.astream(messages):
            if chunk.content:
                yield str(chunk.content)

    def with_model(self, model: str) -> "OpenAILLMProvider":
        """
        Return a new provider instance with a different model.

        Useful for switching between quality tiers (e.g., gpt-4o vs gpt-4o-mini).

        Args:
            model: New model name to use

        Returns:
            New OpenAILLMProvider with the specified model
        """
        return OpenAILLMProvider(api_key=self._api_key, model=model)
