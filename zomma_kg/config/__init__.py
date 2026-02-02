"""
Configuration System

Manages configuration for ZommaKG with a layered approach.

Configuration Priority (highest to lowest):
    1. Programmatic (passed to KGConfig())
    2. Environment variables (ZOMMA_* prefix)
    3. Config file (~/.zomma/config.toml or ZOMMA_CONFIG_FILE)
    4. Built-in defaults

Modules:
    settings: KGConfig class
    providers: Provider-specific configuration
    defaults: Default values and constants

See: docs/architecture/PYTHON_PACKAGE_DESIGN.md Section 6 (Configuration System)
"""

from zomma_kg.config.settings import KGConfig

__all__ = ["KGConfig"]
