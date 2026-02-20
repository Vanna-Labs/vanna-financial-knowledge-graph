"""
Token counting helpers for telemetry fallback paths.

Uses tiktoken when available and a conservative character heuristic otherwise.
"""

from __future__ import annotations

from collections.abc import Iterable


def _count_with_tiktoken(text: str, model: str) -> int | None:
    """Count tokens using tiktoken, returning None if unavailable."""
    try:
        import tiktoken
    except ImportError:
        return None

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def count_text_tokens(text: str, model: str) -> int:
    """
    Estimate tokens for plain text.

    Falls back to a char-based heuristic when tokenizer is unavailable.
    """
    tk_count = _count_with_tiktoken(text, model)
    if tk_count is not None:
        return tk_count

    # Simple fallback: ~4 chars/token
    return max(1, (len(text) + 3) // 4) if text else 0


def count_chat_tokens(messages: Iterable[str], model: str) -> int:
    """
    Estimate tokens for chat-style inputs.

    Adds a small fixed overhead per message for role/control tokens.
    """
    total = 0
    message_count = 0
    for message in messages:
        total += count_text_tokens(message, model)
        message_count += 1

    # Approximate role/message framing overhead.
    return total + (message_count * 4)
