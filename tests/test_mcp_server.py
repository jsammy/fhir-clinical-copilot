"""The MCP surface, checked in-memory: no HAPI needed, because every check here fails or passes
before a request would leave the process."""

from fastmcp import Client

from src.agent.guards import TOOL_ALLOW_LIST
from src.agent.mcp_server import mcp


async def test_tools_match_allow_list_exactly() -> None:
    async with Client(mcp) as client:
        assert {t.name for t in await client.list_tools()} == TOOL_ALLOW_LIST


async def test_every_tool_declares_read_only() -> None:
    async with Client(mcp) as client:
        for tool in await client.list_tools():
            assert tool.annotations.readOnlyHint is True, tool.name
            assert tool.annotations.destructiveHint is False, tool.name


async def test_every_tool_has_structured_output() -> None:
    async with Client(mcp) as client:
        for tool in await client.list_tools():
            assert tool.outputSchema, tool.name


async def test_crafted_patient_id_is_refused() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool("get_conditions", {"patient_id": "1&_count=100000"}, raise_on_error=False)
        assert result.is_error


async def test_count_is_bounded() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool("get_observations", {"patient_id": "1", "count": 5000}, raise_on_error=False)
        assert result.is_error


async def test_category_is_an_enum() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_observations", {"patient_id": "1", "category": "laboratory&_revinclude=*"}, raise_on_error=False)
        assert result.is_error
