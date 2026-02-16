"""
Configuration System

Manages configuration for VannaKG with a layered approach.

Configuration Priority (highest to lowest):
    1. Programmatic (passed to KGConfig())
    2. Environment variables (VANNA_* prefix)
    3. Config file (~/.vanna/config.toml or VANNA_CONFIG_FILE)
    4. Built-in defaults

Modules:
    settings: KGConfig class
    providers: Provider-specific configuration
    defaults: Default values and constants

See: docs/architecture/PYTHON_PACKAGE_DESIGN.md Section 6 (Configuration System)
"""

from vanna_kg.config.settings import KGConfig

__all__ = ["KGConfig"]
