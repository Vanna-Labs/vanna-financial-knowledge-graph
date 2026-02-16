"""
Text Processing Utilities

Functions for text normalization and transformation.
"""

from __future__ import annotations

import re


def normalize_relationship_type(description: str) -> str:
    """
    Normalize a free-form relationship description to UPPER_SNAKE_CASE.

    Args:
        description: e.g., "acquired a majority stake in"

    Returns:
        Normalized type e.g., "ACQUIRED_A_MAJORITY_STAKE_IN"
    """
    # Remove parentheses and contents
    text = re.sub(r"\([^)]*\)", "", description)
    # Replace non-alphanumeric with spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Split, uppercase, limit to 8 words
    words = text.upper().split()[:8]
    return "_".join(words) if words else "RELATED_TO"


def clean_entity_name(name: str) -> str:
    """
    Clean an entity name by removing qualifiers.

    Args:
        name: Raw entity name from extraction

    Returns:
        Cleaned entity name
    """
    # Remove parentheses and contents
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def generate_chunk_id(doc_id: str, sequence: int) -> str:
    """Generate chunk ID: {doc_id}_chunk_{sequence:04d}"""
    return f"{doc_id}_chunk_{sequence:04d}"
