"""Druzhok: limits for Telegram group chats, where every member is an untrusted stranger.

A group runs only if its id is in ``DRUZHOK_GROUP_CHATS``. Its turns get ``group_toolsets`` (else
``["web"]``), no owner memory, no context files, no slash commands, no ``@`` references and no local
files out. DMs are untouched: every helper here answers False / passes through for them.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from gateway.config import Platform
from gateway.platforms._shared import platform_gate_env

logger = logging.getLogger(__name__)

GROUP_CHATS_ENV = "DRUZHOK_GROUP_CHATS"
FALLBACK_TOOLSETS = ("web",)
# Not even a config value opens these: ``memory`` loads MEMORY.md and USER.md whatever skip_memory says.
NEVER_TOOLSETS = frozenset({"memory", "session_search", "sites"})
# ``SessionEntry.metadata`` key set on sessions created under these limits. An entry without it was
# created before them: its stored prompt, pinned tools and transcript may hold owner data.
SESSION_MARKER = "group_limits"
_DM_CHAT_TYPES = frozenset({"dm", "private"})  # "private" is Telegram's own word for a DM
MEDIA_NOTE = "[The user attached media. Only its text, if it is shown here, is available in a group chat.]"


def is_limited_group_chat(platform: Any, chat_type: Any) -> bool:
    """Anything on Telegram that is not a DM. An empty chat type counts as a group: fail closed."""
    return platform == Platform.TELEGRAM and str(chat_type or "").strip().lower() not in _DM_CHAT_TYPES


def is_limited_group(source: Any) -> bool:
    if source is None:
        return False
    return is_limited_group_chat(getattr(source, "platform", None), getattr(source, "chat_type", "dm"))


def approved_group_chats() -> frozenset[str]:
    """Parse the comma-separated signed integers. Anything else is ignored, so no wildcard exists."""
    approved = set()
    for part in platform_gate_env(GROUP_CHATS_ENV).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            approved.add(str(int(part)))
        except ValueError:
            logger.warning("Ignoring non-numeric entry in %s", GROUP_CHATS_ENV)
    return frozenset(approved)


def group_chat_approved(chat_id: Any) -> bool:
    return str(chat_id or "").strip() in approved_group_chats()


def group_agent_kwargs(source: Any) -> dict[str, bool]:
    """``AIAgent`` kwargs for a run built outside the turn runner (background tasks)."""
    if not is_limited_group(source):
        return {}
    return {"skip_memory": True, "skip_context_files": True, "load_soul_identity": True, "skip_background_review": True}


def resolve_group_toolsets(user_config: dict, platform_key: str, override: Any) -> list[str]:
    """The toolsets of a group turn: only names the override lists, whatever else the platform
    resolver would add back (recovered natives, plugin toolsets, MCP servers). Never the default
    composite: a missing, empty, unusable or failing override gives ``FALLBACK_TOOLSETS``."""
    from hermes_cli.tools_config import _get_platform_tools

    names = [str(name).strip() for name in override if str(name).strip()] if isinstance(override, list) else []
    names = [name for name in names if name not in NEVER_TOOLSETS] or list(FALLBACK_TOOLSETS)
    config = {**user_config, "platform_toolsets": {platform_key: names}}
    try:
        kept = _get_platform_tools(config, platform_key, include_default_mcp_servers=False) & set(names)
    except Exception:
        logger.warning("Group toolset resolution failed; using %s", FALLBACK_TOOLSETS, exc_info=True)
        kept = set()
    return sorted(kept) or list(FALLBACK_TOOLSETS)


def drop_group_files(source: Any, media_files: list, local_files: list) -> tuple[list, list]:
    """Drop every local file from a group reply, logged by count."""
    if not is_limited_group(source) or not (media_files or local_files):
        return media_files, local_files
    logger.warning(
        "Dropped %d local file(s) from a reply to group chat %s", len(media_files) + len(local_files),
        getattr(source, "chat_id", None),
    )
    return [], []


def replay_refused(platform_value: Any, chat_id: Any) -> bool:
    """A stored reply bound for a group that is not approved now. It may predate the limits."""
    if str(platform_value or "") != Platform.TELEGRAM.value or str(chat_id or "").strip().isdigit():
        return False
    return not group_chat_approved(chat_id)


def refuse_unlimited_runtime(source: Any, runtime: dict) -> None:
    """A runtime with tools outside ``enabled_toolsets`` cannot serve a group: the Codex app-server runs
    its own, and an external-process (ACP) provider gets file read and write from its client."""
    if not is_limited_group(source):
        return
    from hermes_cli.runtime_provider_backends import _is_external_process_provider

    if runtime.get("api_mode") == "codex_app_server" or _is_external_process_provider(runtime.get("provider")):
        raise RuntimeError("This runtime cannot be limited for a group chat")


def refuse_proxied_group(source: Any) -> None:
    """Proxy mode runs the turn on a remote agent with full access; the limits cannot reach it."""
    if is_limited_group(source):
        raise RuntimeError("A proxied gateway cannot serve a group chat")


def refuses_group_command(event: Any) -> Optional[str]:
    """The slash command a group member sent, which is dropped; None when there is none. Commands
    reach owner sessions, config, skills and shell quick commands, so no group gets any of them."""
    if getattr(event, "internal", False) or not is_limited_group(getattr(event, "source", None)):
        return None
    return event.get_command()
