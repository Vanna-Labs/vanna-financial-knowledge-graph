"""
In-Document Entity Deduplication

Phase 2a-c of the VannaKG ingestion pipeline.

Merges entity mentions that refer to the same real-world entity within
a single document using embedding similarity + LLM verification.

Example:
    >>> from vanna_kg.ingestion.resolution import deduplicate_entities
    >>> result = await deduplicate_entities(entities, llm, embeddings)
    >>> print(f"Reduced to {len(result.canonical_entities)} canonical entities")
"""

from __future__ import annotations

import asyncio
import re
import uuid as uuid_module

import numpy as np
from scipy.spatial.distance import cdist

from typing import TYPE_CHECKING

from vanna_kg.types.entities import EnumeratedEntity
from vanna_kg.types.results import (
    CanonicalEntity,
    EntityDedupeResult,
    EntityDeduplicationOutput,
    MergeRecord,
)
from vanna_kg.utils.clustering import union_find_components

if TYPE_CHECKING:
    from vanna_kg.providers.base import EmbeddingProvider, LLMProvider


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _embedding_text(entity: EnumeratedEntity) -> str:
    """
    Generate embedding text for an entity.

    Format: "{name}: {definition}" if definition exists,
    else "{name}: {summary}" if summary exists,
    else just "{name}".

    Uses definition (stable meaning) rather than summary (context-specific)
    to ensure entities with the same name get similar embeddings.
    """
    # Prefer definition (stable) over summary (context-specific)
    definition = entity.definition.strip() if entity.definition else ""
    if definition:
        return f"{entity.name}: {definition}"

    # Fall back to summary for backwards compatibility
    summary = entity.summary.strip() if entity.summary else ""
    if summary:
        return f"{entity.name}: {summary}"

    return entity.name


def _compute_similarity_matrix(vectors: list[list[float]]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Uses scipy's optimized cdist function for performance.

    Args:
        vectors: List of embedding vectors (n x d)

    Returns:
        n x n similarity matrix where S[i,j] = cosine_sim(v[i], v[j])
    """
    arr = np.array(vectors, dtype=np.float64)
    # cdist returns distance (1 - similarity), so we subtract from 1
    similarity = 1 - cdist(arr, arr, metric="cosine")
    # Handle potential floating point issues (ensure diagonal is exactly 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _similarity_order(n: int, similarity_matrix: np.ndarray) -> list[int]:
    """
    Order entities so similar ones are adjacent (greedy BFS).

    Starts at entity 0, then repeatedly picks the most similar unvisited
    entity to any of the last few visited entities. This ensures that
    when we slice into batches, entities likely to be duplicates end up
    in the same batch (or in the overlap region).

    Args:
        n: Number of entities
        similarity_matrix: n x n cosine similarity matrix

    Returns:
        List of entity indices in similarity-traversal order
    """
    if n <= 1:
        return list(range(n))

    visited = [False] * n
    order: list[int] = []

    # Start at entity 0
    visited[0] = True
    order.append(0)

    while len(order) < n:
        best_next = -1
        best_sim = -1.0

        # Check last 10 visited for efficiency (or all if fewer)
        check_count = min(10, len(order))
        for visited_idx in order[-check_count:]:
            for j in range(n):
                if not visited[j] and similarity_matrix[visited_idx, j] > best_sim:
                    best_sim = similarity_matrix[visited_idx, j]
                    best_next = j

        visited[best_next] = True
        order.append(best_next)

    return order


# -----------------------------------------------------------------------------
# LLM Prompts
# -----------------------------------------------------------------------------

_DEDUP_SYSTEM_PROMPT = """\
You are deduplicating entities for a knowledge graph.

Your task: Given a list of entities, identify which ones refer to the SAME
real-world entity and should be merged.

## MERGE - same real-world entity, different names:
- Ticker and company: "AAPL" = "Apple Inc." = "Apple"
- Abbreviations: "Fed" = "Federal Reserve"
- Person name variants: "Tim Cook" = "Timothy D. Cook" = "Mr. Cook"
- Full and short names: "Amazon.com, Inc." = "Amazon"

## DO NOT MERGE - related but distinct entities:
- Parent and subsidiary: "Alphabet" ≠ "Google" ≠ "YouTube" ≠ "Waymo"
- Person and company: "Tim Cook" ≠ "Apple"
- Product and company: "iPhone" ≠ "Apple"
- Competitors: "Goldman Sachs" ≠ "Morgan Stanley"
- Different people with same name: Context matters!

## Decision Test
Ask yourself: "In a knowledge graph, would these be the same node?"
- Would ALL facts about entity A also apply to entity B?
- If someone asks about A, should they get information about B?

## Output Rules
- Group entities that are THE SAME real-world entity
- Choose the most formal/complete name as canonical
- Merge summaries to combine information from all mentions
- Entities with no duplicates should NOT appear in output (omit singletons)"""

_DEDUP_USER_TEMPLATE = """\
ENTITIES TO ANALYZE:
{entity_list}

Group entities that refer to the SAME real-world entity.
Omit entities that have no duplicates (singletons)."""


async def _verify_cluster(
    entities: list[EnumeratedEntity],
    indices: list[int],
    llm: "LLMProvider",
) -> EntityDedupeResult:
    """
    Use LLM to verify which entities in a cluster are truly the same.

    Args:
        entities: Full list of entities
        indices: Indices of entities in this cluster
        llm: LLM provider for verification

    Returns:
        EntityDedupeResult with groups of entities to merge
    """
    # Build entity list for prompt
    entity_lines = []
    for i, idx in enumerate(indices, 1):
        e = entities[idx]
        summary_part = f": {e.summary}" if e.summary.strip() else ""
        entity_lines.append(f"{i}. {e.name} ({e.entity_type}){summary_part}")

    entity_list = "\n".join(entity_lines)
    prompt = _DEDUP_USER_TEMPLATE.format(entity_list=entity_list)

    return await llm.generate_structured(
        prompt,
        EntityDedupeResult,
        system=_DEDUP_SYSTEM_PROMPT,
    )


def _create_overlapping_batches(
    indices: list[int],
    max_batch_size: int = 15,
    overlap: int = 5,
) -> list[list[int]]:
    """
    Split indices into overlapping batches for LLM processing.

    Args:
        indices: List of entity indices (already in similarity order)
        max_batch_size: Maximum entities per batch
        overlap: Number of overlapping entities between batches

    Returns:
        List of batches, where each batch is a list of indices
    """
    if len(indices) <= max_batch_size:
        return [indices]

    batches: list[list[int]] = []
    step = max_batch_size - overlap
    start = 0

    while start < len(indices):
        end = min(start + max_batch_size, len(indices))
        batches.append(indices[start:end])

        # Move to next batch start
        start += step

        # If remaining elements are less than batch size, we've already captured them
        if end == len(indices):
            break

    return batches


def _merge_batch_results(
    batch_results: list[tuple[list[int], EntityDedupeResult]],
    entities: list[EnumeratedEntity],
) -> list[dict]:
    """
    Merge results from overlapping batches.

    When an entity appears in multiple batches with different group assignments,
    we keep it in its first-assigned group.

    Note:
        Name matching is case-insensitive (e.g., "Apple Inc." matches "APPLE INC.").

    Args:
        batch_results: List of (batch_indices, EntityDedupeResult) tuples
        entities: Full entity list for name lookups

    Returns:
        List of merged group dicts with canonical, members, entity_type, reasoning
    """
    # Track which original indices are assigned to which group
    index_to_group: dict[int, int] = {}
    groups: list[dict] = []

    for batch_indices, result in batch_results:
        # Create name-to-batch-indices mapping (case-insensitive)
        name_to_batch_indices: dict[str, list[int]] = {}
        for batch_idx, orig_idx in enumerate(batch_indices):
            name_key = entities[orig_idx].name.lower().strip()
            if name_key not in name_to_batch_indices:
                name_to_batch_indices[name_key] = []
            name_to_batch_indices[name_key].append(batch_idx)

        for group in result.groups:
            # Get all names in this group, stripping any entity type suffix
            all_names = [group.canonical] + group.members
            cleaned_names = [_strip_entity_type_suffix(name) for name in all_names]
            # Deduplicate cleaned names (case-insensitive) to avoid double-counting
            seen_keys: set[str] = set()
            unique_cleaned_names: list[str] = []
            for name in cleaned_names:
                key = name.lower().strip()
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_cleaned_names.append(name)
            cleaned_canonical = _strip_entity_type_suffix(group.canonical)

            # Convert names to original indices (case-insensitive lookup)
            orig_indices: list[int] = []
            for name in unique_cleaned_names:
                name_key = name.lower().strip()
                if name_key in name_to_batch_indices:
                    for batch_idx in name_to_batch_indices[name_key]:
                        orig_indices.append(batch_indices[batch_idx])

            # Check if any of these indices are already assigned
            existing_group_idx = None
            for idx in orig_indices:
                if idx in index_to_group:
                    existing_group_idx = index_to_group[idx]
                    break

            if existing_group_idx is not None:
                # Merge into existing group
                existing = groups[existing_group_idx]
                existing_names_lower = {existing["canonical"].lower().strip()}
                existing_names_lower.update(m.lower().strip() for m in existing["members"])
                for idx in orig_indices:
                    name = entities[idx].name
                    name_lower = name.lower().strip()
                    if name_lower not in existing_names_lower:
                        existing["members"].append(name)
                        existing_names_lower.add(name_lower)
                    index_to_group[idx] = existing_group_idx
            else:
                # Create new group with cleaned canonical name
                group_idx = len(groups)
                cleaned_members = [_strip_entity_type_suffix(m) for m in group.members]
                groups.append({
                    "canonical": cleaned_canonical,
                    "members": cleaned_members,
                    "entity_type": group.entity_type,
                    "reasoning": group.reasoning,
                })
                for idx in orig_indices:
                    index_to_group[idx] = group_idx

    return groups


def _strip_entity_type_suffix(name: str) -> str:
    """
    Extract clean entity name from LLM response.

    The LLM sometimes returns full entity lines like:
    - "Apple Inc. (Company)" -> "Apple Inc."
    - "Apple Inc. (Company): summary text" -> "Apple Inc."
    - "Federal Reserve" -> "Federal Reserve" (no change)

    This extracts just the entity name.
    """
    # First, strip everything after ": " (the summary part)
    if ': ' in name:
        name = name.split(': ')[0]
    # Then strip trailing " (Type)" where Type is a capitalized word
    pattern = r'\s*\([A-Z][a-zA-Z]*\)$'
    return re.sub(pattern, '', name)


def _process_dedup_result(
    result: EntityDedupeResult,
    component: list[int],
    entities: list[EnumeratedEntity],
) -> list[tuple[str, str, str, list[int]]]:
    """
    Process LLM dedup result into merge groups.

    Note:
        Name matching is case-insensitive (e.g., "Apple Inc." matches "APPLE INC.").

    Returns:
        List of (canonical_name, entity_type, merged_summary, indices) tuples.
    """
    groups: list[tuple[str, str, str, list[int]]] = []

    # Build name -> list of indices mapping (case-insensitive to handle casing variations)
    name_to_indices: dict[str, list[int]] = {}
    for idx in component:
        name_key = entities[idx].name.lower().strip()
        if name_key not in name_to_indices:
            name_to_indices[name_key] = []
        name_to_indices[name_key].append(idx)

    grouped_indices: set[int] = set()

    for group in result.groups:
        all_names = [group.canonical] + group.members
        # Strip entity type suffix that LLM may have included (e.g., "Beige Book (Product)")
        cleaned_names = [_strip_entity_type_suffix(name) for name in all_names]
        # Deduplicate cleaned names (case-insensitive) to avoid double-counting
        seen_keys: set[str] = set()
        unique_cleaned_names: list[str] = []
        for name in cleaned_names:
            key = name.lower().strip()
            if key not in seen_keys:
                seen_keys.add(key)
                unique_cleaned_names.append(name)
        # Collect all indices for all matching names (case-insensitive lookup)
        indices: list[int] = []
        for name in unique_cleaned_names:
            name_key = name.lower().strip()
            if name_key in name_to_indices:
                indices.extend(name_to_indices[name_key])

        if indices:
            grouped_indices.update(indices)
            # Merge summaries
            summaries = [entities[idx].summary for idx in indices if entities[idx].summary.strip()]
            merged_summary = " ".join(summaries) if summaries else ""
            # Use entity_type from the canonical entity (first in indices that matches canonical name)
            # Fall back to first entity's type if canonical name not found
            cleaned_canonical = _strip_entity_type_suffix(group.canonical)
            canonical_key = cleaned_canonical.lower().strip()
            canonical_indices = name_to_indices.get(canonical_key, [])
            canonical_idx = canonical_indices[0] if canonical_indices else indices[0]
            entity_type = entities[canonical_idx].entity_type
            # Use the cleaned canonical name (without type suffix)
            groups.append((cleaned_canonical, entity_type, merged_summary, indices))

    # Add singletons (entities not in any group)
    for idx in component:
        if idx not in grouped_indices:
            e = entities[idx]
            groups.append((e.name, e.entity_type, e.summary, [idx]))

    return groups


async def deduplicate_entities(
    entities: list[EnumeratedEntity],
    llm: "LLMProvider",
    embeddings: "EmbeddingProvider",
    *,
    similarity_threshold: float = 0.70,
    max_batch_size: int = 25,
) -> EntityDeduplicationOutput:
    """
    Deduplicate entities extracted from a single document.

    Uses embedding similarity to find candidate clusters, then LLM verification
    to make final merge decisions.

    Args:
        entities: Entities from extraction (Phase 1 output)
        llm: LLM provider for verification decisions
        embeddings: Embedding provider for similarity computation
        similarity_threshold: Minimum cosine similarity for clustering (default 0.70)
        max_batch_size: Max entities per LLM verification call (default 15)

    Returns:
        EntityDeduplicationOutput with canonical entities, index mapping, and merge history
    """
    # Handle empty input
    if not entities:
        return EntityDeduplicationOutput(
            canonical_entities=[],
            index_to_canonical={},
            merge_history=[],
        )

    n = len(entities)

    # Step 1: Generate embeddings
    texts = [_embedding_text(e) for e in entities]
    vectors = await embeddings.embed(texts)

    # Step 2: Compute similarity matrix and find clusters
    similarity_matrix = _compute_similarity_matrix(vectors)

    # Step 2a: Boost similarity for same-name entities
    # This ensures they're placed adjacent in batch ordering so LLM sees them together
    # We use 0.99 (not 1.0) to preserve self-similarity semantics
    name_to_indices: dict[str, list[int]] = {}
    for i in range(n):
        normalized = entities[i].name.lower().strip()
        if normalized not in name_to_indices:
            name_to_indices[normalized] = []
        name_to_indices[normalized].append(i)

    for indices in name_to_indices.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx_a, idx_b = indices[i], indices[j]
                    similarity_matrix[idx_a, idx_b] = 0.99
                    similarity_matrix[idx_b, idx_a] = 0.99

    # Build edges where similarity >= threshold
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= similarity_threshold:
                edges.append((i, j))

    # Find connected components via Union-Find
    # Note: Same-name entities are already connected via the 0.99 similarity boost above
    components = union_find_components(n, edges)

    # Step 3: LLM verification for multi-entity clusters (PARALLELIZED)
    # Maps original index -> (canonical_name, merged_summary, group_indices)
    merge_groups: list[tuple[str, str, str, list[int]]] = []  # (name, type, summary, indices)

    # Separate singletons (no LLM needed) from clusters (need LLM verification)
    singletons: list[list[int]] = []
    small_clusters: list[list[int]] = []
    large_clusters: list[tuple[list[int], list[list[int]]]] = []  # (ordered_indices, batches)

    for component in components:
        if len(component) == 1:
            singletons.append(component)
        elif len(component) <= max_batch_size:
            small_clusters.append(component)
        else:
            # Prepare overlapping batches for large cluster
            ordered = _similarity_order(len(component), similarity_matrix[np.ix_(component, component)])
            ordered_indices = [component[i] for i in ordered]
            batches = _create_overlapping_batches(ordered_indices, max_batch_size)
            large_clusters.append((ordered_indices, batches))

    # Add singletons immediately (no LLM call needed)
    for component in singletons:
        idx = component[0]
        e = entities[idx]
        merge_groups.append((e.name, e.entity_type, e.summary, [idx]))

    # Collect all LLM verification coroutines for parallel execution
    # Small clusters: one verification per cluster
    small_cluster_coros = [
        _verify_cluster(entities, component, llm)
        for component in small_clusters
    ]

    # Large clusters: multiple batches per cluster
    # Track (cluster_idx, batch_indices) for each coroutine
    large_batch_metadata: list[tuple[int, list[int]]] = []
    large_batch_coros = []
    for cluster_idx, (ordered_indices, batches) in enumerate(large_clusters):
        for batch in batches:
            large_batch_metadata.append((cluster_idx, batch))
            large_batch_coros.append(_verify_cluster(entities, batch, llm))

    # Run ALL LLM calls in parallel (small clusters + large cluster batches)
    all_coros = small_cluster_coros + large_batch_coros
    if all_coros:
        all_results = await asyncio.gather(*all_coros)

        # Split results back into small and large
        small_results = all_results[:len(small_cluster_coros)]
        large_results = all_results[len(small_cluster_coros):]

        # Process small cluster results
        for component, result in zip(small_clusters, small_results):
            groups = _process_dedup_result(result, component, entities)
            merge_groups.extend(groups)

        # Group large cluster batch results by cluster index
        cluster_batch_results: dict[int, list[tuple[list[int], EntityDedupeResult]]] = {}
        for (cluster_idx, batch), result in zip(large_batch_metadata, large_results):
            if cluster_idx not in cluster_batch_results:
                cluster_batch_results[cluster_idx] = []
            cluster_batch_results[cluster_idx].append((batch, result))

        # Process each large cluster's results
        for cluster_idx, (ordered_indices, _) in enumerate(large_clusters):
            batch_results = cluster_batch_results.get(cluster_idx, [])
            merged = _merge_batch_results(batch_results, entities)

            # Build name -> indices mapping for this cluster (case-insensitive)
            cluster_name_to_indices: dict[str, list[int]] = {}
            for idx in ordered_indices:
                name_key = entities[idx].name.lower().strip()
                if name_key not in cluster_name_to_indices:
                    cluster_name_to_indices[name_key] = []
                cluster_name_to_indices[name_key].append(idx)

            grouped_indices: set[int] = set()
            for group in merged:
                indices = []
                for name in [group["canonical"]] + group["members"]:
                    name_key = name.lower().strip()
                    if name_key in cluster_name_to_indices:
                        for idx in cluster_name_to_indices[name_key]:
                            if idx not in indices:
                                indices.append(idx)
                if not indices:
                    continue
                grouped_indices.update(indices)
                # Merge summaries
                summaries = [entities[idx].summary for idx in indices if entities[idx].summary.strip()]
                merged_summary = " ".join(summaries) if summaries else ""
                # Use entity_type from the canonical entity
                canonical_key = group["canonical"].lower().strip()
                canonical_entity_type = entities[indices[0]].entity_type
                for idx in indices:
                    if entities[idx].name.lower().strip() == canonical_key:
                        canonical_entity_type = entities[idx].entity_type
                        break
                merge_groups.append((group["canonical"], canonical_entity_type, merged_summary, indices))

            # Handle singletons not in any group
            for idx in ordered_indices:
                if idx not in grouped_indices:
                    e = entities[idx]
                    merge_groups.append((e.name, e.entity_type, e.summary, [idx]))

    # Step 4: Build output
    canonical_entities: list[CanonicalEntity] = []
    index_to_canonical: dict[int, int] = {}
    merge_history: list[MergeRecord] = []

    for canonical_idx, (name, entity_type, summary, indices) in enumerate(merge_groups):
        entity_uuid = str(uuid_module.uuid4())

        # Collect aliases (names other than canonical)
        aliases = [entities[idx].name for idx in indices if entities[idx].name != name]

        canonical_entities.append(CanonicalEntity(
            uuid=entity_uuid,
            name=name,
            entity_type=entity_type,
            summary=summary,
            source_indices=indices,
            aliases=aliases,
        ))

        for idx in indices:
            index_to_canonical[idx] = canonical_idx

        # Record merge if multiple entities
        if len(indices) > 1:
            merge_history.append(MergeRecord(
                canonical_uuid=entity_uuid,
                canonical_name=name,
                merged_indices=indices,
                merged_names=[entities[idx].name for idx in indices],
                original_summaries=[entities[idx].summary for idx in indices],
                final_summary=summary,
            ))

    return EntityDeduplicationOutput(
        canonical_entities=canonical_entities,
        index_to_canonical=index_to_canonical,
        merge_history=merge_history,
    )
