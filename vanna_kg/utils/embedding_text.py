"""
Helpers for deterministic embedding input text formatting.
"""

from __future__ import annotations


def format_canonical_entity_text(name: str, summary: str) -> str:
    """
    Build canonical embedding text for entities.

    Keeping this format centralized ensures dedup, registry, and assembly
    embed the exact same text representation.
    """
    return f"{name}: {summary}"
