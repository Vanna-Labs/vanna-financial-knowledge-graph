"""
Model pricing metadata for cost telemetry.

All prices are estimated USD per 1M tokens and may change over time.
Unknown models default to 0.0 cost but still report token usage.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICING_VERSION = "2026-02-estimate-v1"


@dataclass(frozen=True)
class LLMPrice:
    """Input/output pricing for chat models (USD per 1M tokens)."""

    input_per_million: float
    output_per_million: float


@dataclass(frozen=True)
class EmbeddingPrice:
    """Input pricing for embedding models (USD per 1M tokens)."""

    input_per_million: float


LLM_PRICING: dict[str, LLMPrice] = {
    "gpt-5.1": LLMPrice(input_per_million=1.25, output_per_million=10.0),
    "gpt-5": LLMPrice(input_per_million=1.25, output_per_million=10.0),
    "gpt-5-mini": LLMPrice(input_per_million=0.25, output_per_million=2.0),
    "gpt-4o": LLMPrice(input_per_million=5.0, output_per_million=15.0),
    "gpt-4o-mini": LLMPrice(input_per_million=0.15, output_per_million=0.6),
}

EMBEDDING_PRICING: dict[str, EmbeddingPrice] = {
    "text-embedding-3-large": EmbeddingPrice(input_per_million=0.13),
    "text-embedding-3-small": EmbeddingPrice(input_per_million=0.02),
    "text-embedding-ada-002": EmbeddingPrice(input_per_million=0.10),
}


def estimate_llm_cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
) -> tuple[float, bool]:
    """
    Estimate LLM cost in USD.

    Returns:
        (cost_usd, priced) where priced=False means model was unknown.
    """
    price = LLM_PRICING.get(model)
    if price is None:
        return 0.0, False
    cost = (
        (input_tokens / 1_000_000.0) * price.input_per_million
        + (output_tokens / 1_000_000.0) * price.output_per_million
    )
    return cost, True


def estimate_embedding_cost_usd(
    model: str,
    *,
    input_tokens: int,
) -> tuple[float, bool]:
    """
    Estimate embedding cost in USD.

    Returns:
        (cost_usd, priced) where priced=False means model was unknown.
    """
    price = EMBEDDING_PRICING.get(model)
    if price is None:
        return 0.0, False
    cost = (input_tokens / 1_000_000.0) * price.input_per_million
    return cost, True
