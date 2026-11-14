"""Druzhok: the CleoHub MCP server's tools stay direct under tool search; every other MCP server defers."""

import pytest

from tools.registry import registry
from tools.tool_search import (
    ToolSearchConfig, assemble_tool_defs, is_deferrable_tool_name, out_of_scope_reason, scope_block_reason,
)


def _td(name: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": f"{name} does one thing.", "parameters": {"type": "object", "properties": {}}}}


@pytest.fixture
def mcp_tools():
    tools = {"mcp_cleohub_plan_status": "mcp-cleohub", "mcp_github_list_issues": "mcp-github"}
    for name, toolset in tools.items():
        registry.register(name=name, handler=lambda args, **kw: "{}", schema=_td(name)["function"], toolset=toolset)
    yield list(tools)
    for name in tools:
        registry.deregister(name)


def test_cleohub_tools_stay_direct_while_other_mcp_tools_defer(mcp_tools):
    cleohub, github = mcp_tools
    assert not is_deferrable_tool_name(cleohub)
    assert is_deferrable_tool_name(github)

    assembled = assemble_tool_defs(
        [_td("terminal"), _td(cleohub), _td(github)], context_length=200_000,
        config=ToolSearchConfig.from_raw({"enabled": "on"}))
    names = {td["function"]["name"] for td in assembled.tool_defs}

    assert assembled.activated
    assert cleohub in names
    assert github not in names


def test_an_explicit_defer_list_still_defers_cleohub(mcp_tools):
    cleohub, _github = mcp_tools
    assert is_deferrable_tool_name(cleohub, frozenset({cleohub}))


def test_cleohub_out_of_scope_is_not_called_a_desktop_tool(mcp_tools):
    cleohub, _github = mcp_tools
    assert out_of_scope_reason(cleohub) is None
    assert "desktop" not in scope_block_reason(cleohub)


def test_sites_stays_direct_and_is_not_called_a_desktop_tool():
    from tools.registry import discover_builtin_tools

    discover_builtin_tools()
    assert not is_deferrable_tool_name("site_publish")
    assert out_of_scope_reason("site_publish") is None
