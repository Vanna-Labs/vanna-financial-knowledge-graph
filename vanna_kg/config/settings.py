"""
KGConfig - Configuration Management

Sensible defaults with full override capability.

Example:
    >>> # Use defaults (reads from environment)
    >>> kg = KnowledgeGraph("./kb")

    >>> # Explicit configuration
    >>> config = KGConfig(
    ...     llm_provider="anthropic",
    ...     llm_model="claude-sonnet-4-5",
    ... )
    >>> kg = KnowledgeGraph("./kb", config=config)

    >>> # From config file
    >>> config = KGConfig.from_file("./my_config.toml")

Environment Variables:
    VANNA_LLM_PROVIDER - LLM provider name
    VANNA_LLM_MODEL - Model for extraction/synthesis
    VANNA_EMBEDDING_PROVIDER - Embedding provider name
    VANNA_EXTRACTION_CONCURRENCY - Max concurrent extraction LLM calls
    VANNA_REGISTRY_CONCURRENCY - Max concurrent entity registry resolutions
    OPENAI_API_KEY - OpenAI API key (standard name)
    ANTHROPIC_API_KEY - Anthropic API key
    GOOGLE_API_KEY - Google API key
    VOYAGE_API_KEY - Voyage API key

See: docs/architecture/PYTHON_PACKAGE_DESIGN.md Section 6
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

# TOML support: tomllib is built-in for Python 3.11+, use tomli for 3.10
try:
    import tomllib

    def _load_toml(path: Path) -> dict[str, Any]:
        with open(path, "rb") as f:
            return cast(dict[str, Any], tomllib.load(f))

    _HAS_TOML = True
except ImportError:
    try:
        import tomli

        def _load_toml(path: Path) -> dict[str, Any]:
            with open(path, "rb") as f:
                return cast(dict[str, Any], tomli.load(f))

        _HAS_TOML = True
    except ImportError:
        def _load_toml(path: Path) -> dict[str, Any]:
            raise ImportError(
                "TOML parsing requires 'tomli' on Python 3.10. "
                "Install with: pip install tomli"
            )

        _HAS_TOML = False


class KGConfig:
    """Configuration for VannaKG."""

    # === LLM Configuration ===

    llm_provider: str = "openai"
    """LLM provider: "openai", "anthropic", "google", "local" """

    llm_model: str = "gpt-5.1"
    """Model for extraction and synthesis"""

    llm_model_fast: str = "gpt-5-mini"
    """Model for quick operations (verification, critique)"""

    llm_model_cheap: str = "gpt-5-mini"
    """Model for bulk operations (summary merging)"""

    # === Embedding Configuration ===

    embedding_provider: str = "openai"
    """Embedding provider: "openai", "voyage", "local" """

    embedding_model: str = "text-embedding-3-large"
    """Embedding model name"""

    embedding_dimensions: int = 3072
    """Embedding vector dimensions (provider-dependent)"""

    # === API Keys ===

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    voyage_api_key: str | None = None

    # === Processing Configuration ===

    extraction_concurrency: int = 10
    """Max concurrent LLM extraction calls"""

    registry_concurrency: int = 10
    """Max concurrent entity registry resolutions (LLM calls for cross-doc matching)"""

    embedding_concurrency: int = 5
    """Max concurrent embedding batches"""

    embedding_batch_size: int = 100
    """Texts per embedding API call"""

    dedup_similarity_threshold: float = 0.70
    """Cosine similarity threshold for entity deduplication"""

    # === Query Configuration ===

    query_entity_threshold: float = 0.3
    """Minimum similarity for entity resolution in queries"""

    query_fact_threshold: float = 0.3
    """Minimum similarity for fact retrieval"""

    query_topic_threshold: float = 0.35
    """Minimum similarity for topic resolution in queries"""

    query_high_relevance_threshold: float = 0.45
    """Threshold for classifying chunks as high-relevance"""

    query_max_subqueries: int = 5
    """Maximum sub-queries for question decomposition"""

    query_max_entity_candidates: int = 30
    """Maximum entity candidates for resolution vector search"""

    query_max_topic_candidates: int = 20
    """Maximum topic candidates for resolution vector search"""

    query_enable_expansion: bool = True
    """Enable 1-hop neighbor expansion in queries"""

    query_enable_global_search: bool = False
    """Enable global chunk search in addition to entity-based retrieval (not yet implemented)"""

    query_max_high_relevance_chunks: int = 30
    """Maximum high-relevance chunks in context"""

    query_max_facts: int = 40
    """Maximum facts in context"""

    query_max_topic_chunks: int = 15
    """Maximum topic-related chunks in context"""

    query_max_low_relevance_chunks: int = 20
    """Maximum low-relevance supporting chunks in context"""

    query_global_search_limit: int = 50
    """Maximum chunks from global vector search"""

    query_research_concurrency: int = 5
    """Maximum concurrent sub-query research operations"""

    query_topic_llm_verification: bool = True
    """Whether to use LLM to verify topic matches (False = use top-k directly)"""

    # === Cost Telemetry Configuration ===

    cost_debug_warn_threshold_usd: float | None = None
    """Optional warning threshold for per-request estimated cost in cost_debug mode"""

    # === Storage Configuration ===

    parquet_compression: str = "zstd"
    """Parquet compression: "zstd", "snappy", "gzip", "none" """

    lancedb_index_type: str = "IVF_PQ"
    """Vector index type for LanceDB"""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize configuration.

        Args:
            **kwargs: Override any configuration option
        """
        # Load from environment first
        self._load_from_env()

        # Apply explicit overrides
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration option: {key}")

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        import os

        # API keys (standard names)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.voyage_api_key = os.getenv("VOYAGE_API_KEY")

        # VANNA_* prefixed settings
        if provider := os.getenv("VANNA_LLM_PROVIDER"):
            self.llm_provider = provider
        if model := os.getenv("VANNA_LLM_MODEL"):
            self.llm_model = model
        if provider := os.getenv("VANNA_EMBEDDING_PROVIDER"):
            self.embedding_provider = provider
        if concurrency := os.getenv("VANNA_EXTRACTION_CONCURRENCY"):
            self.extraction_concurrency = int(concurrency)
        if concurrency := os.getenv("VANNA_REGISTRY_CONCURRENCY"):
            self.registry_concurrency = int(concurrency)
        if threshold := os.getenv("VANNA_COST_DEBUG_WARN_THRESHOLD_USD"):
            self.cost_debug_warn_threshold_usd = float(threshold)

    @classmethod
    def from_file(cls, path: str | Path) -> "KGConfig":
        """
        Load configuration from TOML file.

        The TOML file can contain any configuration option as a key.
        Nested sections are flattened with underscores.

        Example TOML:
            [llm]
            provider = "anthropic"
            model = "claude-sonnet-4-5"

            [embedding]
            provider = "openai"
            model = "text-embedding-3-large"

            [api_keys]
            openai = "sk-..."

        Args:
            path: Path to TOML configuration file

        Returns:
            KGConfig instance with values from file

        Raises:
            FileNotFoundError: If config file doesn't exist
            ImportError: If tomli not installed on Python 3.10
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        data = _load_toml(path)

        # Flatten nested sections into config keys
        flat_config: dict[str, Any] = {}

        # Map section names to config key prefixes
        section_mapping = {
            "llm": "llm_",
            "embedding": "embedding_",
            "api_keys": "",  # api_keys.openai -> openai_api_key
            "processing": "",
            "query": "query_",
            "cost_telemetry": "cost_debug_",
            "storage": "",
        }

        for section, prefix in section_mapping.items():
            if section in data:
                for key, value in data[section].items():
                    if section == "api_keys":
                        # api_keys.openai -> openai_api_key
                        flat_config[f"{key}_api_key"] = value
                    else:
                        flat_config[f"{prefix}{key}"] = value

        # Also support flat top-level keys
        for key, value in data.items():
            if key not in section_mapping and not isinstance(value, dict):
                flat_config[key] = value

        return cls(**flat_config)

    @classmethod
    def from_env(cls) -> "KGConfig":
        """Load configuration from environment variables only."""
        return cls()

    def to_file(self, path: str | Path) -> None:
        """
        Save configuration to TOML file.

        Organizes configuration into logical sections for readability.
        API keys are excluded by default for security.

        Args:
            path: Path to write TOML configuration file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Organize into sections
        sections: dict[str, dict[str, str | int | float | bool | None]] = {
            "llm": {
                "provider": self.llm_provider,
                "model": self.llm_model,
                "model_fast": self.llm_model_fast,
                "model_cheap": self.llm_model_cheap,
            },
            "embedding": {
                "provider": self.embedding_provider,
                "model": self.embedding_model,
                "dimensions": self.embedding_dimensions,
            },
            "processing": {
                "extraction_concurrency": self.extraction_concurrency,
                "registry_concurrency": self.registry_concurrency,
                "embedding_concurrency": self.embedding_concurrency,
                "embedding_batch_size": self.embedding_batch_size,
                "dedup_similarity_threshold": self.dedup_similarity_threshold,
            },
            "query": {
                "entity_threshold": self.query_entity_threshold,
                "fact_threshold": self.query_fact_threshold,
                "topic_threshold": self.query_topic_threshold,
                "high_relevance_threshold": self.query_high_relevance_threshold,
                "max_subqueries": self.query_max_subqueries,
                "max_entity_candidates": self.query_max_entity_candidates,
                "max_topic_candidates": self.query_max_topic_candidates,
                "enable_expansion": self.query_enable_expansion,
                "enable_global_search": self.query_enable_global_search,
                "max_high_relevance_chunks": self.query_max_high_relevance_chunks,
                "max_facts": self.query_max_facts,
                "max_topic_chunks": self.query_max_topic_chunks,
                "max_low_relevance_chunks": self.query_max_low_relevance_chunks,
                "global_search_limit": self.query_global_search_limit,
                "research_concurrency": self.query_research_concurrency,
                "topic_llm_verification": self.query_topic_llm_verification,
            },
            "cost_telemetry": {
                "warn_threshold_usd": self.cost_debug_warn_threshold_usd,
            },
            "storage": {
                "parquet_compression": self.parquet_compression,
                "lancedb_index_type": self.lancedb_index_type,
            },
        }

        # Build TOML string manually (avoids extra dependency)
        lines = ["# VannaKG Configuration", ""]

        for section_name, section_values in sections.items():
            lines.append(f"[{section_name}]")
            for key, value in section_values.items():
                if isinstance(value, str):
                    lines.append(f'{key} = "{value}"')
                elif isinstance(value, bool):
                    lines.append(f"{key} = {str(value).lower()}")
                elif isinstance(value, (int, float)):
                    lines.append(f"{key} = {value}")
            lines.append("")

        # Note about API keys
        lines.extend([
            "# API keys should be set via environment variables:",
            "# OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, VOYAGE_API_KEY",
            "",
        ])

        path.write_text("\n".join(lines))

    def with_overrides(self, **kwargs: Any) -> "KGConfig":
        """Return new config with specified overrides."""
        new_config = KGConfig.__new__(KGConfig)
        for key in dir(self):
            if not key.startswith("_") and not callable(getattr(self, key)):
                setattr(new_config, key, getattr(self, key))
        for key, value in kwargs.items():
            setattr(new_config, key, value)
        return new_config
