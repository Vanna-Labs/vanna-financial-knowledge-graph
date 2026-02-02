"""
Provider Configurations

Default model configurations for each LLM/embedding provider.

When a provider is selected, appropriate model defaults are applied:
    >>> config = KGConfig(llm_provider="anthropic")
    >>> # Automatically sets:
    >>> #   llm_model = "claude-sonnet-4-5"
    >>> #   llm_model_fast = "claude-haiku-3-5"
"""

# Provider default models
PROVIDER_DEFAULTS = {
    "openai": {
        "llm_model": "gpt-4o",
        "llm_model_fast": "gpt-4o-mini",
        "llm_model_cheap": "gpt-4o-mini",
    },
    "anthropic": {
        "llm_model": "claude-sonnet-4-5",
        "llm_model_fast": "claude-haiku-3-5",
        "llm_model_cheap": "claude-haiku-3-5",
    },
    "google": {
        "llm_model": "gemini-2.5-flash",
        "llm_model_fast": "gemini-2.5-flash-lite",
        "llm_model_cheap": "gemini-2.5-flash-lite",
    },
    "local": {
        "llm_model": "llama3.1:8b",
        "llm_model_fast": "llama3.1:8b",
        "llm_model_cheap": "llama3.1:8b",
    },
}

# Embedding provider defaults
EMBEDDING_DEFAULTS = {
    "openai": {
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": 3072,
    },
    "voyage": {
        "embedding_model": "voyage-finance-2",
        "embedding_dimensions": 1024,
    },
    "local": {
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "embedding_dimensions": 1024,
    },
}
