"""
Topic Resolver - Topic Ontology Resolution

Resolves extracted topics against a curated ontology using:
1. Vector similarity search (LanceDB)
2. Batched LLM verification

The ontology is loaded from a JSON file and cached in LanceDB.
Hash-based change detection triggers automatic reload when JSON changes.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from zomma_kg.config import KGConfig
from zomma_kg.providers.base import EmbeddingProvider, LLMProvider
from zomma_kg.storage.lancedb.indices import LanceDBIndices
from zomma_kg.types.topics import (
    BatchTopicMatchResponse,
    TopicDefinition,
    TopicMatchDecision,
    TopicResolution,
    TopicResolutionResult,
)

logger = logging.getLogger(__name__)


class TopicResolver:
    """
    Resolves extracted topics against the curated ontology.

    Usage:
        resolver = TopicResolver(lancedb_indices, llm, embeddings)
        result = await resolver.resolve(topics)

        # result.resolved_topics - matched to ontology
        # result.uuid_remap - {topic_name: ontology_uuid}
        # result.new_topics - unmatched (if collect_unmatched=True)
    """

    def __init__(
        self,
        indices: LanceDBIndices,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        config: KGConfig | None = None,
        # Behavior toggles
        collect_unmatched: bool = True,
        # Batching & concurrency
        batch_size: int = 10,
        concurrency: int = 3,  # -1 = unlimited
        # Search tuning
        candidate_limit: int = 15,
        similarity_threshold: float = 0.40,
        high_similarity_flag: float = 0.85,
        # Ontology
        ontology_group_id: str = "ontology",
    ):
        self.indices = indices
        self.llm = llm_provider
        self.embeddings = embedding_provider
        self.config = config or KGConfig()

        # Behavior toggles
        self.collect_unmatched = collect_unmatched

        # Batching & concurrency
        self.batch_size = batch_size
        self.concurrency = concurrency

        # Search tuning
        self.candidate_limit = candidate_limit
        self.similarity_threshold = similarity_threshold
        self.high_similarity_flag = high_similarity_flag

        # Ontology
        self.ontology_group_id = ontology_group_id

        # Internal state
        self._ontology_loaded = False

    @staticmethod
    def _normalize_topic_name(name: str) -> str:
        """Normalize topic names for case/whitespace-insensitive matching."""
        return name.strip().lower()

    def _get_ontology_path(self) -> Path:
        """Get path to the ontology JSON file."""
        # Use package data path
        import zomma_kg
        package_dir = Path(zomma_kg.__file__).parent
        return package_dir / "data" / "topics" / "financial_topics.json"

    def _get_hash_file_path(self) -> Path:
        """Get path to the ontology hash file."""
        return self.indices.path / ".ontology_hash"

    def _load_ontology_entries(self, ontology_path: Path) -> list[dict]:
        """Load ontology entries from JSON file."""
        if not ontology_path.exists():
            raise FileNotFoundError(
                f"Topic ontology file not found at {ontology_path}. "
                "Ensure the zomma_kg package is installed correctly with data files."
            )
        try:
            with open(ontology_path) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in ontology file {ontology_path}: {e}"
            ) from e

    def _compute_ontology_hash(self, ontology_path: Path) -> str:
        """Compute SHA256 hash of ontology file."""
        content = ontology_path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    def _should_reload_ontology(self, ontology_path: Path) -> bool:
        """Check if ontology needs reloading based on hash comparison."""
        hash_file = self._get_hash_file_path()

        if not hash_file.exists():
            return True

        try:
            stored = json.loads(hash_file.read_text())
            stored_hash = stored.get("hash", "")
            current_hash = self._compute_ontology_hash(ontology_path)
            return stored_hash != current_hash
        except (json.JSONDecodeError, KeyError):
            return True

    def _save_ontology_hash(self, hash_value: str) -> None:
        """Save ontology hash to file."""
        hash_file = self._get_hash_file_path()
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "hash": hash_value,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        }
        hash_file.write_text(json.dumps(data))

    def _generate_embedding_texts(
        self, entries: list[dict]
    ) -> tuple[list[str], list[str]]:
        """
        Generate embedding texts for ontology entries.

        For each entry, creates:
        - "label: definition"
        - "synonym: definition" for each synonym

        Returns:
            texts: List of embedding texts
            uuids: List of corresponding UUIDs (from uri)
        """
        texts: list[str] = []
        uuids: list[str] = []

        for entry in entries:
            label = entry["label"]
            definition = entry.get("definition", "")
            uri = entry["uri"]
            # Extract UUID from URI (last segment)
            uuid = uri.split("/")[-1]
            synonyms = entry.get("synonyms", [])

            # Add label: definition
            texts.append(f"{label}: {definition}")
            uuids.append(uuid)

            # Add synonym: definition for each synonym
            for synonym in synonyms:
                texts.append(f"{synonym}: {definition}")
                uuids.append(uuid)

        return texts, uuids

    async def _ensure_ontology_loaded(self) -> None:
        """Lazy load ontology with hash-based change detection."""
        ontology_path = self._get_ontology_path()

        # Skip if already loaded and hash matches
        if self._ontology_loaded and not self._should_reload_ontology(ontology_path):
            return

        await self._load_ontology(ontology_path)

    async def reload_ontology(self) -> None:
        """Force reload ontology from JSON, ignoring hash."""
        ontology_path = self._get_ontology_path()
        await self._load_ontology(ontology_path)

    async def _load_ontology(self, ontology_path: Path) -> None:
        """Load ontology into LanceDB."""
        logger.info(f"Loading topic ontology from {ontology_path}")

        # Load entries
        entries = self._load_ontology_entries(ontology_path)

        # Generate embedding texts
        texts, uuids = self._generate_embedding_texts(entries)

        # Generate embeddings
        embeddings_list = await self.embeddings.embed(texts)

        # Build topic dicts for LanceDB
        # Create lookup for definitions
        entry_lookup = {e["uri"].split("/")[-1]: e for e in entries}

        topics = []
        for text, uuid in zip(texts, uuids):
            entry = entry_lookup[uuid]
            topics.append({
                "uuid": uuid,
                "name": text.split(":")[0],  # Extract name from "name: def"
                "definition": entry.get("definition", ""),
                "group_id": self.ontology_group_id,
            })

        # Write to LanceDB
        await self.indices.add_topics(topics, embeddings_list)

        # Save hash
        current_hash = self._compute_ontology_hash(ontology_path)
        self._save_ontology_hash(current_hash)

        self._ontology_loaded = True
        logger.info(f"Loaded {len(entries)} topics ({len(texts)} embeddings)")

    def _topic_to_text(self, topic: TopicDefinition) -> str:
        """Convert topic to embedding text: '{topic}: {definition}'."""
        return f"{topic.topic}: {topic.definition}"

    def _build_batch_verification_prompt(
        self,
        topics: list[TopicDefinition],
        candidates_list: list[list[tuple[dict, float]]],
    ) -> str:
        """Build prompt for batch topic verification."""
        lines = [
            "Match each extracted topic to its best candidate from the ontology.",
            "Return the candidate number (1-indexed) or null if no good match.",
            "",
            "TOPICS TO VERIFY:",
            "",
        ]

        for i, (topic, candidates) in enumerate(zip(topics, candidates_list), 1):
            lines.append(f"{i}. Extracted: \"{topic.topic}\"")
            lines.append(f"   Definition: {topic.definition}")
            lines.append("   Candidates:")

            if not candidates:
                lines.append("   (no candidates found)")
            else:
                for j, (cand, score) in enumerate(candidates, 1):
                    pct = int(score * 100)
                    flag = " [LIKELY MATCH]" if score >= self.high_similarity_flag else ""
                    lines.append(
                        f"   {j}. {cand['name']} ({pct}% similar){flag}"
                    )
                    lines.append(f"      Definition: {cand['definition']}")

            lines.append("")

        lines.extend([
            "RULES:",
            "- Match if the extracted topic and candidate refer to the same concept",
            "- Different names are OK if meaning is same (e.g., 'M&A' = 'Mergers')",
            "- Return null if no candidate is a good semantic match",
            "",
            "Return a decision for each topic.",
        ])

        return "\n".join(lines)

    async def _verify_batch(
        self,
        topics: list[TopicDefinition],
        candidates_list: list[list[tuple[dict, float]]],
    ) -> list[TopicMatchDecision]:
        """Verify a batch of topics using LLM."""
        prompt = self._build_batch_verification_prompt(topics, candidates_list)

        system = (
            "You are a topic resolution expert. Match extracted topics to their "
            "canonical forms in a curated ontology. Be precise - only match if "
            "the concepts are semantically equivalent."
        )

        try:
            response = await self.llm.generate_structured(
                prompt, BatchTopicMatchResponse, system=system
            )
            return response.decisions
        except Exception as e:
            logger.warning(f"Batch LLM verification failed: {e}")
            # Return no-match decisions for all topics
            return [
                TopicMatchDecision(
                    topic=t.topic,
                    selected_number=None,
                    reasoning=f"LLM verification error: {e}",
                )
                for t in topics
            ]

    async def resolve(
        self,
        topics: list[TopicDefinition],
        embeddings: list[list[float]] | None = None,
    ) -> TopicResolutionResult:
        """
        Resolve topics against the ontology.

        Args:
            topics: Topics from extraction (with definitions)
            embeddings: Pre-computed embeddings, or None to generate

        Returns:
            TopicResolutionResult with resolved_topics, uuid_remap, new_topics
        """
        if not topics:
            return TopicResolutionResult()

        # Ensure ontology is loaded
        await self._ensure_ontology_loaded()

        # Deduplicate topics by normalized name before verification to avoid
        # duplicate decisions for case/whitespace variants.
        deduped_topics: list[TopicDefinition] = []
        originals_by_key: dict[str, list[str]] = {}
        for topic in topics:
            key = self._normalize_topic_name(topic.topic)
            if key not in originals_by_key:
                deduped_topics.append(topic)
                originals_by_key[key] = [topic.topic]
            else:
                originals_by_key[key].append(topic.topic)

        # Generate embeddings if not provided
        if embeddings is None:
            texts = [self._topic_to_text(t) for t in deduped_topics]
            embeddings = await self.embeddings.embed(texts)
        elif len(embeddings) != len(deduped_topics):
            raise ValueError(
                "Pre-computed embeddings count must match deduplicated topic count."
            )

        # Search for candidates for each topic
        candidates_list: list[list[tuple[dict, float]]] = []
        for embedding in embeddings:
            candidates = await self.indices.search_topics(
                embedding,
                limit=self.candidate_limit,
                threshold=self.similarity_threshold,
            )
            # Filter to only ontology group
            candidates = [
                (c, s) for c, s in candidates
                if c.get("group_id", self.ontology_group_id) == self.ontology_group_id
            ]
            candidates_list.append(candidates)

        # Batch verification with concurrency
        all_decisions = await self._verify_all_batches(deduped_topics, candidates_list)

        # Build result
        deduped_result = self._build_result(deduped_topics, candidates_list, all_decisions)

        # Expand deduplicated results back to all original topic spellings.
        uuid_remap: dict[str, str] = {}
        new_topics: list[str] = []
        for key, original_names in originals_by_key.items():
            deduped_topic_name = original_names[0]
            matched_uuid = deduped_result.uuid_remap.get(deduped_topic_name)
            if matched_uuid is not None:
                for original_name in original_names:
                    uuid_remap[original_name] = matched_uuid
            elif self.collect_unmatched:
                new_topics.extend(original_names)

        return TopicResolutionResult(
            resolved_topics=deduped_result.resolved_topics,
            uuid_remap=uuid_remap,
            new_topics=new_topics,
        )

    async def _verify_all_batches(
        self,
        topics: list[TopicDefinition],
        candidates_list: list[list[tuple[dict, float]]],
    ) -> list[TopicMatchDecision]:
        """Verify all topics in batches with concurrency control."""
        # Split into batches
        batches: list[tuple[list[TopicDefinition], list[list[tuple[dict, float]]]]] = []
        for i in range(0, len(topics), self.batch_size):
            batch_topics = topics[i : i + self.batch_size]
            batch_candidates = candidates_list[i : i + self.batch_size]
            batches.append((batch_topics, batch_candidates))

        # Set up concurrency
        if self.concurrency == -1:
            # Unlimited parallelism
            tasks = [self._verify_batch(bt, bc) for bt, bc in batches]
            batch_results = await asyncio.gather(*tasks)
        else:
            # Bounded concurrency
            semaphore = asyncio.Semaphore(self.concurrency)

            async def bounded_verify(
                bt: list[TopicDefinition], bc: list[list[tuple[dict, float]]]
            ) -> list[TopicMatchDecision]:
                async with semaphore:
                    return await self._verify_batch(bt, bc)

            tasks = [bounded_verify(bt, bc) for bt, bc in batches]
            batch_results = await asyncio.gather(*tasks)

        # Flatten results
        all_decisions: list[TopicMatchDecision] = []
        for decisions in batch_results:
            all_decisions.extend(decisions)

        return all_decisions

    def _build_result(
        self,
        topics: list[TopicDefinition],
        candidates_list: list[list[tuple[dict, float]]],
        decisions: list[TopicMatchDecision],
    ) -> TopicResolutionResult:
        """Build TopicResolutionResult from decisions."""
        resolved_topics: list[TopicResolution] = []
        uuid_remap: dict[str, str] = {}
        new_topics: list[str] = []

        # Validate decision count matches topic count
        if len(decisions) != len(topics):
            logger.warning(
                f"Decision count mismatch: got {len(decisions)} decisions "
                f"for {len(topics)} topics. Building lookup by topic name."
            )

        # Build decision lookup by topic name for robust matching
        decision_by_topic: dict[str, TopicMatchDecision] = {}
        for decision in decisions:
            # Normalize topic name for matching (case-insensitive)
            key = self._normalize_topic_name(decision.topic)
            if key in decision_by_topic:
                logger.warning(f"Duplicate decision for topic '{decision.topic}'")
            decision_by_topic[key] = decision

        for topic, candidates in zip(topics, candidates_list):
            # Look up decision by topic name
            key = self._normalize_topic_name(topic.topic)
            decision = decision_by_topic.get(key)

            if decision is None:
                logger.warning(
                    f"No decision found for topic '{topic.topic}'. "
                    "Treating as unmatched."
                )
                if self.collect_unmatched:
                    new_topics.append(topic.topic)
                continue

            if decision.selected_number is not None and candidates:
                # Match found
                idx = decision.selected_number - 1  # Convert to 0-indexed
                if 0 <= idx < len(candidates):
                    matched = candidates[idx][0]
                    resolution = TopicResolution(
                        uuid=matched["uuid"],
                        canonical_label=matched["name"],
                        is_new=False,
                        definition=matched.get("definition", ""),
                    )
                    resolved_topics.append(resolution)
                    uuid_remap[topic.topic] = matched["uuid"]
                    continue
                else:
                    logger.warning(
                        f"Invalid candidate index {decision.selected_number} "
                        f"for topic '{topic.topic}' (max: {len(candidates)}). "
                        "Treating as unmatched."
                    )

            # No match
            if self.collect_unmatched:
                new_topics.append(topic.topic)

        return TopicResolutionResult(
            resolved_topics=resolved_topics,
            uuid_remap=uuid_remap,
            new_topics=new_topics,
        )
