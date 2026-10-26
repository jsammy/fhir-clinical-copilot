"""Starting the read-only MCP server and calling it safely. Shared by both entry points:
the pre-visit workflow (previsit_cli) and the Q&A agent (ask_cli).

The tool process gets a minimal environment: OS basics plus its own settings. AWS credentials go to
it only when notes search embeds queries with Titan (NOTES_EMBEDDER=bedrock), because then it must
call Bedrock itself. On AWS that would be a role allowed to invoke the embedding model and nothing else.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.client.stdio import get_default_environment

from src.agent.guards import TOOL_ALLOW_LIST

ROOT = Path(__file__).resolve().parents[2]
_EMBEDDER_ENV = ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION", "AWS_PROFILE", "TITAN_EMBED_MODEL_ID")


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """KEY=value lines from the gitignored .env, for settings like AWS_BEARER_TOKEN_BEDROCK that must
    never be committed or typed into a shared terminal. Variables already set win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def tool_env(fhir_base_url: str) -> dict[str, str]:
    env = {**get_default_environment(), "FHIR_BASE_URL": fhir_base_url, "FASTMCP_LOG_LEVEL": "WARNING",
           "NOTES_EMBEDDER": os.environ.get("NOTES_EMBEDDER", "hash")}
    if env["NOTES_EMBEDDER"] == "bedrock":
        env |= {k: os.environ[k] for k in _EMBEDDER_ENV if k in os.environ}
    return env


def mcp_client(fhir_base_url: str) -> MultiServerMCPClient:
    return MultiServerMCPClient({
        "fhir": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "src.agent.mcp_server"],
            "cwd": str(ROOT),
            "env": tool_env(fhir_base_url),
        }
    })


@asynccontextmanager
async def fhir_tools(fhir_base_url: str) -> AsyncIterator[tuple[Any, list[Any]]]:
    """Yields (call_tool, tool definitions). Refuses to start unless the server's tools exactly
    match the allow-list, and checks the allow-list again on every call."""
    async with mcp_client(fhir_base_url).session("fhir") as session:
        tools = (await session.list_tools()).tools
        advertised = {t.name for t in tools}
        if advertised != TOOL_ALLOW_LIST:
            raise PermissionError(f"server tools {sorted(advertised ^ TOOL_ALLOW_LIST)} differ from the allow-list")

        async def call_tool(name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
            if name not in TOOL_ALLOW_LIST:
                raise PermissionError(f"{name} is not an allowed tool")
            result = await session.call_tool(name, tool_args)
            if result.isError:
                raise RuntimeError(f"{name} failed: {' '.join(getattr(c, 'text', '') for c in result.content)}")
            return result.structuredContent or {}

        yield call_tool, tools
