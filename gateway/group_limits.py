"""Druzhok: limits for Telegram group chats, where every member is an untrusted stranger.

A group runs only if its id is in ``DRUZHOK_GROUP_CHATS``. Its turns get ``group_toolsets`` (else
``["web"]``), no owner memory, no context files, no slash commands and no ``@`` references. The only
local files that go out are the pictures ``image_generate`` made in that same turn (the turn's outbox),
each approved once for that chat while its delivery runs; ``group_media`` lists the inbound kinds a
turn may see. DMs are untouched: every helper here answers False / passes through for them.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Optional

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
GROUP_MEDIA_KINDS = frozenset({"image"})
GROUP_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})
REFERENCE_REFUSAL = "в группе можно только нарисовать по описанию"
GROUP_IMAGE_UNSEEN = "[A group member sent an image, but it could not be seen this time.]"
_OUTBOX_TTL_SECONDS = 600.0
_VALIDATED_TTL_SECONDS = 600.0
_state_lock = threading.Lock()
_outboxes: dict[str, tuple[float, frozenset[str]]] = {}  # session key -> (expiry, real paths)
_validated: dict[tuple[str, str], float] = {}  # (chat id, real path) kept by drop_group_files -> expiry
_approvals: dict[tuple[str, str], int] = {}  # (chat id, real path) -> deliveries sending it now


def is_limited_group_chat(platform: Any, chat_type: Any) -> bool:
    """Anything on Telegram that is not a DM. An empty chat type counts as a group: fail closed.
    ``platform`` is a ``Platform`` or its stored value (an admission row's encoded source)."""
    return platform in (Platform.TELEGRAM, Platform.TELEGRAM.value) and str(chat_type or "").strip().lower() not in _DM_CHAT_TYPES


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


def group_media(value: Any) -> frozenset[str]:
    """``platforms.telegram.extra.group_media``: the inbound media kinds a group turn may see. Missing,
    null, empty or not a list is none (fail closed); unknown items are ignored."""
    if not isinstance(value, list):
        return frozenset()
    return frozenset(item.strip().lower() for item in value if isinstance(item, str)) & GROUP_MEDIA_KINDS


def has_reference(value: Any) -> bool:
    """Whether an ``image_url`` / ``reference_image_urls`` argument names any source, in any form."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(has_reference(item) for item in value)
    return value is not None and value is not False


def group_sees_images(runner: Any, source: Any) -> bool:
    """Whether a group turn on ``source`` gets its images described (``"image"`` in ``group_media``)."""
    return "image" in group_media(runner._adapter_extra_for_source(source).get("group_media"))


def in_limited_group_turn() -> bool:
    """Whether the running tool call belongs to a limited group turn, read from the turn's session vars.
    In a gateway process a call with no platform bound counts as one: it cannot prove it is not."""
    from gateway.session_context import get_session_env, session_context_engaged

    platform = get_session_env("HERMES_SESSION_PLATFORM", "").strip().lower()
    if not platform:
        return session_context_engaged()
    return is_limited_group_chat(platform, get_session_env("HERMES_SESSION_CHAT_TYPE", ""))


def _real(path: Any) -> str:
    return os.path.realpath(str(path))


def turn_outbox(messages: list, start: Optional[int]) -> frozenset[str]:
    """Real paths of the local images that successful ``image_generate`` calls made in ``messages[start:]``,
    the rows this run appended. No trusted start (``None``, or past the list) gives nothing."""
    from gateway.media_repair import tool_name_by_call_id

    if not isinstance(messages, list) or not isinstance(start, int) or not 0 <= start <= len(messages):
        return frozenset()
    rows = messages[start:]
    names = tool_name_by_call_id(rows)
    paths = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("role") not in ("tool", "function"):
            continue
        if names.get(str(row.get("tool_call_id") or row.get("call_id") or "")) != "image_generate":
            continue
        paths.update(_real(path) for path in _generated_paths(str(row.get("content") or ""))
                     if os.path.isabs(path) and os.path.isfile(path))
    return frozenset(paths)


def _generated_paths(content: str) -> list[str]:
    """The picture a successful JSON ``image_generate`` result names. A text result's ``MEDIA:`` tag is
    not read: provider text can echo the prompt, which a group member wrote."""
    try:
        payload = json.loads(content)
    except ValueError:
        return []
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return []
    return next(([path] for field in ("host_image", "image") if isinstance(path := payload.get(field), str)), [])


def set_turn_outbox(session_key: Any, paths: Iterable[str]) -> None:
    """Record a finished group turn's outbox for its delivery; replaces whatever an earlier turn left."""
    if not session_key:
        return
    with _state_lock:
        _outboxes[str(session_key)] = (time.monotonic() + _OUTBOX_TTL_SECONDS, frozenset(paths))


def take_turn_outbox(session_key: Any) -> frozenset[str]:
    """The outbox of the session's last finished turn, removed so no later delivery can use it."""
    with _state_lock:
        expiry, paths = _outboxes.pop(str(session_key or ""), (0.0, frozenset()))
    return paths if expiry > time.monotonic() else frozenset()


def record_turn_outbox(source: Any, session_key: Any, result: dict, history_len: int, *, compressed: bool) -> None:
    """After a group run: its outbox, from the rows it appended. The loop's proven current-turn row marks
    them; without it only an uncompressed run's history length does, and a compressed one gets none."""
    if not is_limited_group(source):
        return
    idx = result.get("current_turn_user_idx")
    start = idx + 1 if isinstance(idx, int) and not isinstance(idx, bool) else None if compressed else history_len
    set_turn_outbox(session_key, turn_outbox(result.get("messages"), start))


def _approval_key(chat_id: Any, path: Any) -> tuple[str, str]:
    return str(chat_id or "").strip(), _real(path)


def take_group_file_approval(chat_id: Any, path: Any) -> bool:
    """Use up an approval when a send passes the adapter's gate with it. That send's own retries and
    fallbacks run on its scope (``telegram_group_files``), not on the approval."""
    key = _approval_key(chat_id, path)
    with _state_lock:
        return _approvals.pop(key, 0) > 0


@contextmanager
def group_delivery(source: Any, media_files: list, local_files: list) -> Iterator[None]:
    """Approve a group reply's pictures for the adapter's gate while its attachments go out, after any
    text send before them, and withdraw what is left when the delivery ends. Only paths
    ``drop_group_files`` kept for this chat are approved; each is approved once. While the delivery
    runs, another send naming the same picture for the same chat could use the approval first: it is
    this turn's own picture to its own group, and the delivery's send is then refused."""
    if not is_limited_group(source):
        yield
        return
    chat_id = getattr(source, "chat_id", None)
    now = time.monotonic()
    approved = []
    with _state_lock:
        for path in [p for p, _voice in media_files] + list(local_files):
            key = _approval_key(chat_id, path)
            if _validated.pop(key, 0.0) > now:
                _approvals[key] = 1
                approved.append(key)
    try:
        yield
    finally:
        with _state_lock:
            for key in approved:
                _approvals.pop(key, None)


def _keep_for_group(path: Any, outbox: frozenset[str]) -> bool:
    return (os.path.splitext(str(path))[1].lower() in GROUP_IMAGE_EXTS and _real(path) in outbox
            and os.path.isfile(path))


def drop_group_files(source: Any, media_files: list, local_files: list,
                     session_key: Any = None) -> tuple[list, list]:
    """Keep only the session's last turn's own generated pictures (its outbox, used up here) in a group
    reply, marked for ``group_delivery`` to approve; every other local file is dropped, logged by count."""
    if not is_limited_group(source) or not (media_files or local_files):
        return media_files, local_files
    outbox = take_turn_outbox(session_key)
    kept_media = [(path, voice) for path, voice in media_files if not voice and _keep_for_group(path, outbox)]
    kept_local = [path for path in local_files if _keep_for_group(path, outbox)]
    chat_id = getattr(source, "chat_id", None)
    dropped = len(media_files) + len(local_files) - len(kept_media) - len(kept_local)
    if dropped:
        logger.warning("Dropped %d local file(s) from a reply to group chat %s", dropped, chat_id)
    now = time.monotonic()
    with _state_lock:
        for key in [key for key, until in _validated.items() if until <= now]:
            del _validated[key]
        for path in [p for p, _voice in kept_media] + kept_local:
            _validated[_approval_key(chat_id, path)] = now + _VALIDATED_TTL_SECONDS
    return kept_media, kept_local


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
