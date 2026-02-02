"""
Filesystem Navigation

Virtual filesystem interface for knowledge graph exploration.

Modules:
    commands: Command implementations (ls, cd, cat, grep, find, etc.)
    path_resolver: Virtual path resolution
    formatters: Output formatting

Virtual Filesystem Structure:
    /kg/
    ├── entities/
    │   ├── @search
    │   └── {entity_name}/
    │       ├── summary.txt
    │       ├── facts/
    │       ├── connections/
    │       └── chunks/
    ├── topics/
    │   ├── @search
    │   └── {topic_name}/
    │       ├── definition.txt
    │       └── chunks/
    ├── chunks/
    │   ├── @search
    │   └── chunk_*.txt
    └── documents/
        └── {doc_name}/

Commands:
    pwd, cd, ls, cat, head, tail, grep, find, wc

See: docs/architecture/FILESYSTEM_NAVIGATION_ARCHITECTURE.md
"""

__all__ = []
