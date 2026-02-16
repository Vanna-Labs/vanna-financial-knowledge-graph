"""
VannaKG MCP Server

Exposes knowledge graph query operations via a single MCP tool (kg_execute)
that accepts command strings following the skill syntax.

Tool:
    - kg_execute: Execute a VannaKG command (find, search, cat, info, ls, stats)

Workflow:
    1. find --entity {"name":"Apple","definition":"US tech company"}
                                                       # Resolve names
    2. search -entity Apple_Inc --mode around           # Search around selected nodes
    3. search -entity Apple_Inc -topic Inflation --mode between
                                                        # Search only inside selected subset
    4. cat 1                                            # Expand result details
    5. info "Apple Inc"                                 # Entity/topic summary

Usage:
    # Run the MCP server
    python -m vanna_kg.mcp --kb ./my_kb

    # Or in Claude Desktop config:
    {
        "mcpServers": {
            "vanna-kg": {
                "command": "python",
                "args": ["-m", "vanna_kg.mcp", "--kb", "./my_kb"]
            }
        }
    }

See: vanna_kg/skills/kg-query/SKILL.md for full command documentation
"""

from vanna_kg.mcp.server import create_server, run_server

__all__ = ["create_server", "run_server"]
