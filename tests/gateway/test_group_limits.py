"""Druzhok group limits, from the stranger's side: an approved Telegram group's members get nothing of
the owner's. Each leak test has a DM control, so a pass cannot mean the planted data was never visible."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_context, build_session_key

GROUP_ID = "-1001234"
OWNER_ID = "111"
STRANGER_ID = "222"
SECRET = "PLANTED-OWNER-SECRET"


def _group(chat_id=GROUP_ID, user_id=STRANGER_ID, chat_type="group"):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type=chat_type, user_id=user_id)


def _dm(user_id=OWNER_ID):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=user_id, chat_type="dm", user_id=user_id)


def _runner(adapter=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)})
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter is not None else {}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    return runner


def _adapter(extra=None):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra=extra or {}))
    adapter._bot = AsyncMock()
    adapter._bot.send_document = AsyncMock(return_value=SimpleNamespace(message_id=1))
    return adapter


# ── who may talk ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", [None, "", " , ", "*", "abc"])
def test_no_usable_list_refuses_every_group_even_the_owner(monkeypatch, raw):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", OWNER_ID)
    monkeypatch.setenv("TELEGRAM_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", GROUP_ID)
    if raw is None:
        monkeypatch.delenv("DRUZHOK_GROUP_CHATS", raising=False)
    else:
        monkeypatch.setenv("DRUZHOK_GROUP_CHATS", raw)
    runner = _runner()
    for chat_type in ("group", "forum", "channel"):
        assert not runner._is_user_authorized(_group(user_id=OWNER_ID, chat_type=chat_type))
        assert not runner._is_user_authorized(_group(user_id=None, chat_type=chat_type))


def test_unlisted_group_is_refused_and_a_listed_one_admits_any_member(monkeypatch):
    monkeypatch.setenv("DRUZHOK_GROUP_CHATS", f" {GROUP_ID} , -5")
    runner = _runner()
    assert runner._is_user_authorized(_group())
    assert runner._is_user_authorized(_group(chat_id="-5", chat_type="forum"))
    assert runner._is_user_authorized(_group(user_id=None))  # anonymous admin
    assert not runner._is_user_authorized(_group(chat_id="-999"))
    assert not runner._is_user_authorized(_group(chat_id="5"))  # the sign is part of the id


def test_a_listed_id_never_admits_a_dm(monkeypatch):
    monkeypatch.setenv("DRUZHOK_GROUP_CHATS", STRANGER_ID)
    assert not _runner()._is_user_authorized(_dm(STRANGER_ID))


def test_dm_authorization_is_unchanged(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", OWNER_ID)
    monkeypatch.setenv("TELEGRAM_ALLOW_ALL_USERS", "true")
    runner = _runner()
    assert runner._is_user_authorized(_dm(OWNER_ID))
    assert not runner._is_user_authorized(_dm(STRANGER_ID))


def test_a_group_member_cannot_press_a_gated_button(monkeypatch):
    monkeypatch.setenv("DRUZHOK_GROUP_CHATS", GROUP_ID)
    adapter = _adapter()
    adapter.set_authorization_check(lambda *a, **k: True)
    assert not adapter._is_callback_user_authorized(STRANGER_ID, chat_id=GROUP_ID, chat_type="group")
    assert not adapter._is_callback_user_authorized(STRANGER_ID, chat_id=GROUP_ID, chat_type="supergroup")
    assert adapter._is_callback_user_authorized(OWNER_ID, chat_id=OWNER_ID, chat_type="private")


# ── tools ────────────────────────────────────────────────────────────────────────────────────────


def _resolve(adapter, source, config=None):
    return _runner(adapter)._resolve_enabled_toolsets_for_source(config or {}, source, "telegram")


@pytest.mark.parametrize("extra", [{}, {"group_toolsets": []}, {"group_toolsets": None}, {"group_toolsets": "terminal"}])
def test_missing_or_unusable_override_gives_web(extra):
    assert _resolve(_adapter(extra), _group()) == ["web"]


def test_adapter_failure_or_no_adapter_gives_web():
    broken = _adapter()
    broken.toolsets_for_source = MagicMock(side_effect=RuntimeError("boom"))
    assert _resolve(broken, _group()) == ["web"]
    assert _resolve(None, _group()) == ["web"]


def test_resolver_failure_gives_web():
    with patch("hermes_cli.tools_config._get_platform_tools", side_effect=RuntimeError("boom")):
        assert _resolve(_adapter({"group_toolsets": ["web"]}), _group()) == ["web"]


def test_nothing_is_added_back_to_a_group_list(monkeypatch):
    """Saved list, recovered natives, plugin toolsets and MCP servers all stay out, ``sites`` first."""
    monkeypatch.setenv("DRUZHOK_SITES_ENABLED", "1")
    config = {
        "platform_toolsets": {"telegram": ["terminal", "file", "sites", "memory", "session_search"]},
        "mcp_servers": {"owner_mcp": {"command": "x"}},
    }
    with patch("hermes_cli.tools_config._get_plugin_toolset_keys", return_value={"owner_plugin"}):
        assert _resolve(_adapter({"group_toolsets": ["web"]}), _group(), config) == ["web"]


def test_config_cannot_open_owner_data_toolsets():
    extra = {"group_toolsets": ["web", "memory", "session_search", "sites"]}
    assert _resolve(_adapter(extra), _group()) == ["web"]
    assert _resolve(_adapter({"group_toolsets": ["memory"]}), _group()) == ["web"]


def test_sites_is_absent_from_group_schemas_and_on_in_a_dm(monkeypatch):
    from model_tools import get_tool_definitions

    monkeypatch.setenv("DRUZHOK_SITES_ENABLED", "1")
    assert "sites" in _resolve(_adapter({"group_toolsets": ["web"]}), _dm())
    group = _resolve(_adapter({"group_toolsets": ["web"]}), _group())
    names = {d["function"]["name"] for d in get_tool_definitions(enabled_toolsets=group, quiet_mode=True)}
    assert not names - {"web_search", "web_extract"}


def test_background_runs_use_the_same_resolver():
    enabled, _disabled = _runner(_adapter())._resolve_turn_toolsets({}, _group(), "telegram")
    assert enabled == ["web"]


# ── what the prompt holds ────────────────────────────────────────────────────────────────────────


@pytest.fixture
def planted_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    (home / "memories").mkdir(parents=True)
    workspace.mkdir()
    (home / "SOUL.md").write_text("You are Sharik, a friendly bot.")
    (home / "memories" / "MEMORY.md").write_text(f"{SECRET}-MEMORY")
    (home / "memories" / "USER.md").write_text(f"{SECRET}-USER")
    (workspace / "AGENTS.md").write_text(f"{SECRET}-AGENTS")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    monkeypatch.chdir(workspace)
    return home


def _real_agent(**kwargs):
    from run_agent import AIAgent

    return AIAgent(
        model="fixture", provider="custom", base_url="http://127.0.0.1:9/v1", api_key="fixture", quiet_mode=True,
        platform="telegram", load_soul_identity=True, **kwargs,
    )


def test_group_prompt_keeps_the_name_and_holds_no_owner_data(planted_home):
    group = _real_agent(enabled_toolsets=["web"], skip_memory=True, skip_context_files=True)
    prompt = group._build_system_prompt(None)
    assert "Sharik" in prompt
    assert SECRET not in prompt
    assert group._memory_store is None and group._memory_manager is None
    assert not group.valid_tool_names - {"web_search", "web_extract"}
    assert not {"session_search", "memory", "skill_view", "terminal"} & set(group.valid_tool_names)

    dm = _real_agent(enabled_toolsets=["web", "memory"])
    dm_prompt = dm._build_system_prompt(None)
    assert f"{SECRET}-MEMORY" in dm_prompt and f"{SECRET}-USER" in dm_prompt and f"{SECRET}-AGENTS" in dm_prompt


def test_stored_prompt_and_pinned_tools_of_an_old_session_are_not_restored(planted_home):
    from agent.conversation_loop import _restore_or_build_system_prompt

    agent = _real_agent(enabled_toolsets=["web"], skip_memory=True, skip_context_files=True)
    agent.group_limited = True
    agent._session_db = MagicMock()
    agent._session_db.get_session.return_value = {
        "system_prompt": f"{SECRET}-OLD-PROMPT", "tool_names": json.dumps(["terminal", "read_file"]),
    }
    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])
    assert SECRET not in agent._cached_system_prompt
    assert not {"terminal", "read_file"} & {t["function"]["name"] for t in agent.tools or []}


def test_group_prompt_does_not_name_the_owners_home_channel():
    from gateway.session import build_session_context_prompt

    config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(
        enabled=True, token="t", home_channel=HomeChannel(Platform.TELEGRAM, OWNER_ID, "Owner Private Chat"))})
    assert "Owner Private Chat" in build_session_context_prompt(build_session_context(_dm(), config))
    prompt = build_session_context_prompt(build_session_context(_group(), config))
    assert "Owner Private Chat" not in prompt and OWNER_ID not in prompt


# ── agent cache ──────────────────────────────────────────────────────────────────────────────────


def _turn_runner(session_key, source, cache):
    import threading

    from gateway.run_turn_runner import TurnRunner

    built = MagicMock(name="fresh-agent")
    ctx = SimpleNamespace(
        source=source, session_key=session_key, session_id="sid", user_config={}, enabled_toolsets=["web"],
        disabled_toolsets=None, AIAgent=MagicMock(return_value=built), channel_prompt=None,
        interim_assistant_messages_enabled=False,
    )
    runner = MagicMock()
    runner._agent_cache = cache
    runner._agent_cache_lock = threading.Lock()
    from gateway.run_agent_cache import GatewayAgentCacheMixin

    runner._agent_config_signature = GatewayAgentCacheMixin._agent_config_signature
    runner._extract_cache_busting_config.return_value = {}
    turn = TurnRunner(runner, ctx)
    turn._cached_sid_is_dead = MagicMock(return_value=(None, False))
    turn._current_message_count = MagicMock(return_value=0)
    return turn, ctx, built


def _route():
    return {"model": "m", "runtime": {"api_key": "k"}, "request_overrides": None}


def test_a_dm_agent_is_never_reused_for_a_group():
    from gateway.run_agent_cache import GatewayAgentCacheMixin as Cache

    dm_key = build_session_key(_dm())
    group_key = build_session_key(_group())
    assert dm_key != group_key
    assert build_session_key(_group(), group_sessions_per_user=False) != dm_key

    # Worst case: same key, same toolsets, same user. Only the limits differ, and the signature must.
    source = _group()
    dm_sig = Cache._agent_config_signature(
        "m", {"api_key": "k"}, ["web"], "", cache_keys={}, user_id=source.user_id, skip_context_files=True,
    )
    dm_agent = MagicMock(name="dm-agent")
    turn, _ctx, built = _turn_runner(group_key, source, {group_key: (dm_agent, dm_sig, 0, "sid")})
    agent, reused = turn._resolve_turn_agent(_route(), "telegram", "", 10, None, {})
    assert agent is built and agent is not dm_agent and not reused


def test_group_agent_is_built_limited_and_a_dm_agent_is_not():
    turn, ctx, built = _turn_runner(build_session_key(_group()), _group(), {})
    turn._resolve_turn_agent(_route(), "telegram", "", 10, None, {})
    kwargs = ctx.AIAgent.call_args.kwargs
    assert kwargs["skip_memory"] and kwargs["skip_context_files"] and kwargs["skip_background_review"]
    assert kwargs["load_soul_identity"] and built.group_limited is True

    turn, ctx, _built = _turn_runner(build_session_key(_dm()), _dm(), {})
    turn._runner._skip_context_files = None
    with patch.object(turn, "_skip_context_files", return_value=False):
        turn._resolve_turn_agent(_route(), "telegram", "", 10, None, {})
    kwargs = ctx.AIAgent.call_args.kwargs
    assert not kwargs["skip_memory"] and not kwargs["skip_context_files"] and not kwargs["skip_background_review"]


# ── nothing goes to owner memory afterwards ──────────────────────────────────────────────────────


def _agent_as_the_gateway_builds_it(source):
    """A real AIAgent from the kwargs the turn runner passes, so the gateway wiring is under test."""
    from run_agent import AIAgent

    turn, ctx, _built = _turn_runner(build_session_key(source), source, {})
    with patch.object(turn, "_skip_context_files", return_value=False):
        turn._resolve_turn_agent(_route(), "telegram", "", 10, None, {})
    kwargs = {**ctx.AIAgent.call_args.kwargs, "enabled_toolsets": ["web", "memory"] if source.chat_type == "dm" else ["web"]}
    kwargs.update(model="fixture", provider="custom", base_url="http://127.0.0.1:9/v1", api_key="fixture", session_db=None)
    for key in ("api_key_ref", "fallback_model"):
        kwargs.pop(key, None)
    return AIAgent(**kwargs)


def test_a_group_turn_has_nothing_that_writes_owner_memory_and_a_dm_does(planted_home):
    group = _agent_as_the_gateway_builds_it(_group())
    assert group._memory_manager is None and group._memory_store is None and group.skip_background_review
    dm = _agent_as_the_gateway_builds_it(_dm())
    assert dm._memory_store is not None and not dm.skip_background_review


@pytest.mark.parametrize("source,expect_skip", [(_group(), True), (_dm(), False)])
def test_hygiene_checkpoint_never_loads_the_memory_provider_for_a_group(source, expect_skip):
    runner = _runner()
    runner._session_db = MagicMock()
    runner._session_db.get_session.return_value = None
    entry = SimpleNamespace(session_id="sid", origin=source)
    with patch("run_agent.AIAgent") as agent_cls, patch("gateway.run._seed_hygiene_system_prompt"), patch(
        "hermes_cli.config.load_config", return_value={"compression": {"checkpoint_required": True}},
    ):
        asyncio.run(runner._hmwa_hygiene_build_agent("m", {"api_key": "k"}, entry))
    assert agent_cls.call_args.kwargs["skip_memory"] is expect_skip


def test_prefill_and_checkpoint_stay_out_of_a_group_agent():
    turn, ctx, built = _turn_runner(build_session_key(_group()), _group(), {})
    turn._runner._prefill_messages = [{"role": "user", "content": SECRET}]
    turn._resolve_turn_agent(_route(), "telegram", "", 10, None, {})
    assert ctx.AIAgent.call_args.kwargs["prefill_messages"] is None
    assert ctx.AIAgent.call_args.kwargs["fallback_model"] is None
    assert built.compression_checkpoint_required is False

    turn, ctx, _built = _turn_runner(build_session_key(_dm()), _dm(), {})
    turn._runner._prefill_messages = [{"role": "user", "content": "kept"}]
    with patch.object(turn, "_skip_context_files", return_value=False):
        turn._resolve_turn_agent(_route(), "telegram", "", 10, None, {})
    assert ctx.AIAgent.call_args.kwargs["prefill_messages"]
    assert ctx.AIAgent.call_args.kwargs["fallback_model"] is not None


@pytest.mark.parametrize("runtime", [
    {"api_key": "k", "api_mode": "codex_app_server"},
    {"api_key": "k", "api_mode": "chat_completions", "provider": "fixture-acp"},
])
def test_a_runtime_with_its_own_tools_cannot_serve_a_group(runtime, monkeypatch):
    # Core ships no external-process provider since copilot-acp became a plugin; a plugin may register one.
    import hermes_cli.runtime_provider_backends as backends
    monkeypatch.setattr(backends, "_is_external_process_provider", lambda name: name == "fixture-acp")
    route = {"model": "m", "runtime": runtime, "request_overrides": None}
    turn, ctx, _built = _turn_runner(build_session_key(_group()), _group(), {})
    with pytest.raises(RuntimeError):
        turn._resolve_turn_agent(route, "telegram", "", 10, None, {})
    ctx.AIAgent.assert_not_called()

    turn, ctx, _built = _turn_runner(build_session_key(_dm()), _dm(), {})
    with patch.object(turn, "_skip_context_files", return_value=False):
        turn._resolve_turn_agent(route, "telegram", "", 10, None, {})
    ctx.AIAgent.assert_called_once()


@pytest.mark.parametrize("source,refused", [(_group(), True), (_dm(), False)])
def test_proxy_mode_never_runs_a_group_turn_on_the_remote_agent(source, refused):
    runner = _runner()
    runner._get_proxy_url = MagicMock(return_value="https://remote.example")
    runner._run_agent_via_proxy = AsyncMock(return_value={"final_response": "remote"})
    call = runner._run_agent_inner(
        message=SECRET, context_prompt="", history=[], source=source, session_id="sid", session_key="k",
    )
    if refused:
        with pytest.raises(RuntimeError):
            asyncio.run(call)
        runner._run_agent_via_proxy.assert_not_awaited()
    else:
        assert asyncio.run(call)["final_response"] == "remote"


def test_a_stored_reply_never_reaches_an_unapproved_group(monkeypatch):
    monkeypatch.setenv("DRUZHOK_GROUP_CHATS", "-5")
    adapter = _adapter()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))
    runner = _runner(adapter)
    runner._arm_flood_timers_for_waiting_rows = AsyncMock()

    def row(chat_id, obligation_id):
        return {"obligation_id": obligation_id, "platform": "telegram", "chat_id": chat_id, "content": SECRET, "attempts": 0}

    with patch("gateway.delivery_ledger.mark_delivered") as delivered:
        asyncio.run(runner._redeliver_claimed_obligations([row(GROUP_ID, "a"), row(OWNER_ID, "b"), row("-5", "c")]))
    assert sorted(call.kwargs["chat_id"] for call in adapter.send.await_args_list) == ["-5", OWNER_ID]
    assert {call.args[0] for call in delivered.call_args_list} == {"a", "b", "c"}


def test_the_model_is_not_told_a_group_session_was_suspended():
    from gateway.run import _AUTO_RESET_CONTEXT_NOTES

    assert "suspended" not in _AUTO_RESET_CONTEXT_NOTES["group_limits"]


def test_background_agent_kwargs():
    from gateway.group_limits import group_agent_kwargs

    assert group_agent_kwargs(_dm()) == {}
    assert group_agent_kwargs(_group()) == {
        "skip_memory": True, "skip_context_files": True, "load_soul_identity": True, "skip_background_review": True,
    }


@pytest.mark.parametrize("source,limited", [(_group(), True), (_dm(), False)])
def test_background_run_builds_its_agent_under_the_limits(source, limited):
    adapter = _adapter()
    adapter.send = AsyncMock()
    runner = _runner(adapter)
    runner._thread_metadata_for_source = MagicMock(return_value={})
    runner._resolve_session_agent_runtime = MagicMock(return_value=("m", {"api_key": "k"}))
    runner._resolve_session_reasoning_config = MagicMock(return_value=None)
    runner._resolve_session_service_tier = MagicMock(return_value=None)
    runner._resolve_turn_agent_config = MagicMock(return_value={"model": "m", "runtime": {"api_key": "k"}})
    runner._refresh_fallback_model = MagicMock(return_value=[{"provider": "copilot-acp"}])
    runner._cleanup_agent_resources = MagicMock()
    runner._provider_routing = {}
    runner._session_db = None

    async def in_line(fn, *args):
        return fn(*args)

    runner._run_in_executor_with_context = in_line
    with patch("run_agent.AIAgent") as agent_cls, patch("gateway.run._load_gateway_config", return_value={}):
        agent_cls.return_value.run_conversation.return_value = {"final_response": "ok", "messages": []}
        asyncio.run(runner._run_background_task_inner("hi", source, "task-1"))
    kwargs = agent_cls.call_args.kwargs
    assert kwargs.get("skip_memory", False) is limited and kwargs.get("skip_context_files", False) is limited
    assert (kwargs["enabled_toolsets"] == ["web"]) is limited
    assert (kwargs["fallback_model"] is None) is limited


# ── no files out ─────────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def owner_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    memories = tmp_path / "memories"
    memories.mkdir()
    path = memories / "MEMORY.md"
    path.write_text(SECRET)
    return path


def _event(text, source, **kwargs):
    return MessageEvent(text=text, message_type=kwargs.pop("message_type", MessageType.TEXT), source=source, **kwargs)


def test_media_tag_and_bare_path_are_dropped_for_a_group(owner_file):
    adapter = _adapter()
    reply = f"Here.\nMEDIA:{owner_file}\nAlso {owner_file}"
    extracted = asyncio.run(adapter._extract_response_content(reply, _event("x", _group()), "k", is_ephemeral_response=False))
    assert not extracted.media_files and not extracted.local_files

    control = asyncio.run(adapter._extract_response_content(f"MEDIA:{owner_file}", _event("x", _dm()), "k", is_ephemeral_response=False))
    assert [path for path, _voice in control.media_files] == [str(owner_file.resolve())]


@pytest.mark.parametrize("chat_id", [GROUP_ID, "-5", "@public_channel"])
def test_every_local_file_sender_refuses_a_group(owner_file, chat_id):
    adapter = _adapter()
    path = str(owner_file)

    async def attempts():
        return [
            await adapter.send_document(chat_id=chat_id, file_path=path),
            await adapter.send_image_file(chat_id=chat_id, image_path=path),
            await adapter.send_video(chat_id=chat_id, video_path=path),
            await adapter.send_voice(chat_id=chat_id, audio_path=path),
            await adapter.send_multiple_images(chat_id=chat_id, images=[(f"file://{path}", "")]),
        ]

    assert not any(result.success for result in asyncio.run(attempts()))
    uploaders = ("send_document", "send_photo", "send_video", "send_voice", "send_audio", "send_media_group")
    assert not [name for name in uploaders if getattr(adapter._bot, name).await_count]


def test_a_dm_still_gets_its_file(owner_file):
    adapter = _adapter()
    result = asyncio.run(adapter.send_document(chat_id=OWNER_ID, file_path=str(owner_file)))
    assert result.success and adapter._bot.send_document.await_count == 1


def test_post_stream_delivery_cannot_send_a_group_file(owner_file):
    adapter = _adapter()
    runner = _runner(adapter)
    runner._thread_metadata_for_source = MagicMock(return_value={})
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    asyncio.run(runner._deliver_media_from_response(f"MEDIA:{owner_file}", _event("x", _group()), adapter))
    assert adapter._bot.send_document.await_count == 0


def test_auto_tts_is_off_for_a_group():
    adapter = _adapter()
    adapter._should_auto_tts_for_chat = MagicMock(return_value=True)
    adapter._streaming_tts_turn_completed = MagicMock(return_value=False)
    kwargs = dict(session_key="k", interrupt_event=asyncio.Event(), text_content="hi", media_files=[])
    assert not adapter._wants_auto_tts(_event("x", _group(), message_type=MessageType.VOICE), **kwargs)
    assert adapter._wants_auto_tts(_event("x", _dm(), message_type=MessageType.VOICE), **kwargs)


# ── nothing local is read before the turn ────────────────────────────────────────────────────────


def _prepare(runner, event):
    return asyncio.run(runner._prepare_inbound_message_text(event=event, source=event.source, history=[], session_key="k"))


def _inbound_runner():
    runner = _runner(_adapter())
    runner._peek_session_state = MagicMock(return_value=None)
    runner._enrich_inbound_images = AsyncMock(return_value="ANALYSED")
    runner._inbound_model_context_length = AsyncMock(return_value=100000)
    return runner


@pytest.mark.parametrize("text", ["@file:AGENTS.md", "@folder:../cache/documents", "@file:../cache/documents/a.pdf", "@diff", "@git:1"])
def test_at_references_stay_literal_in_a_group(tmp_path, monkeypatch, text):
    (tmp_path / "AGENTS.md").write_text(SECRET)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    with patch("agent.context_references.preprocess_context_references_async", new=AsyncMock()) as expand:
        assert _prepare(_inbound_runner(), _event(text, _group())) == text
    expand.assert_not_awaited()


def test_at_file_expands_in_a_dm_control(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text(SECRET)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    assert SECRET in _prepare(_inbound_runner(), _event("@file:AGENTS.md", _dm()))


def test_a_quoted_at_reference_stays_literal():
    text = _prepare(_inbound_runner(), _event("look", _group(), reply_to_message_id="9", reply_to_text="@file:AGENTS.md"))
    assert '"@file:AGENTS.md"' in text


def test_group_media_is_not_analysed_and_no_local_path_is_shown():
    path = "/data/tenants/bot/cache/documents/1_2_report.pdf"
    runner = _inbound_runner()
    doc = _event("see", _group(), message_type=MessageType.DOCUMENT, media_urls=[path], media_types=["application/pdf"])
    assert "/data/tenants" not in _prepare(runner, doc)

    photo = _event("see", _group(), message_type=MessageType.PHOTO, media_urls=["/x/1_2_p.jpg"], media_types=["image/jpeg"])
    assert "/x/" not in _prepare(runner, photo)
    runner._enrich_inbound_images.assert_not_awaited()


# ── commands and skills ──────────────────────────────────────────────────────────────────────────


def _admission_runner(monkeypatch):
    monkeypatch.setenv("DRUZHOK_GROUP_CHATS", GROUP_ID)
    runner = _runner(_adapter())
    runner._scale_to_zero_note_real_inbound = MagicMock()
    runner._hm_pre_gateway_dispatch_hook = AsyncMock(side_effect=lambda event, source: event)
    # Admitted on purpose: these tests are about the command gate, not the allowlist.
    runner._is_user_authorized_for_source = MagicMock(return_value=True)
    runner._admit_bot_message_for_source = MagicMock(return_value=True)
    return runner


@pytest.mark.parametrize("text", [
    "/help", "/status", "/new", "/background read MEMORY.md", "/export", "/resume x", "/model", "/usage",
    "/skills", "/any-skill", "/owner_quick_command", "/help@some_bot", "/sethome", "/approve",
])
def test_no_slash_command_reaches_the_gateway_from_a_group(monkeypatch, text):
    runner = _admission_runner(monkeypatch)
    assert asyncio.run(runner._hm_admit_event(_event(text, _group()))) is None


def test_plain_text_and_paths_are_still_answered_in_a_group(monkeypatch):
    runner = _admission_runner(monkeypatch)
    for text in ("hello", "what is /etc/hosts", "/etc/hosts is what"):
        assert asyncio.run(runner._hm_admit_event(_event(text, _group()))) is not None


def test_a_command_is_dropped_before_any_busy_path():
    adapter = _adapter()
    adapter._message_handler = AsyncMock()
    adapter._start_session_processing = MagicMock(return_value=True)
    adapter._handle_message_while_active = AsyncMock()
    asyncio.run(adapter.handle_message(_event("/stop", _group())))
    adapter._handle_message_while_active.assert_not_awaited()
    adapter._start_session_processing.assert_not_called()


# ── sessions from before the limits ──────────────────────────────────────────────────────────────


def _store(path):
    store = SessionStore(sessions_dir=path, config=GatewayConfig())
    store._db = None
    return store


def _old_entry(store, source):
    key = store._generate_session_key(source)
    now = datetime.now()
    entry = SessionEntry(
        session_key=key, session_id="old-owner-session", created_at=now, updated_at=now, origin=source,
        platform=source.platform, chat_type=source.chat_type,
    )
    with store._lock:
        store._ensure_loaded_locked()
        store._entries[key] = entry
    return key


def test_an_old_group_session_is_replaced_once_and_the_new_one_survives_a_restart(tmp_path):
    store = _store(tmp_path)
    _old_entry(store, _group())
    fresh = store.get_or_create_session(_group())
    assert fresh.session_id != "old-owner-session"
    assert store.get_or_create_session(_group()).session_id == fresh.session_id
    store._save()

    reloaded = _store(tmp_path)
    assert reloaded.get_or_create_session(_group()).session_id == fresh.session_id


def test_a_reset_group_session_is_not_mistaken_for_an_old_one(tmp_path):
    store = _store(tmp_path)
    first = store.get_or_create_session(_group())
    reset = store.reset_session(first.session_key)
    assert store.get_or_create_session(_group()).session_id == reset.session_id


def test_a_compression_heal_does_not_carry_an_old_group_session_past_the_limits(tmp_path):
    store = _store(tmp_path)
    key = _old_entry(store, _group())
    tip = patch.object(store, "_compression_tip_for_session_id", return_value="compressed-child")
    with tip:
        fresh = store.get_or_create_session(_group())
    assert fresh.session_key == key
    assert fresh.session_id not in ("old-owner-session", "compressed-child")


def test_an_old_dm_session_is_kept(tmp_path):
    store = _store(tmp_path)
    _old_entry(store, _dm())
    assert store.get_or_create_session(_dm()).session_id == "old-owner-session"


def test_a_durable_row_is_never_adopted_for_a_group(tmp_path):
    store = _store(tmp_path)
    recoverable = MagicMock(return_value=MagicMock(session_id="old-owner-session"))
    with patch.object(store, "_query_recoverable_session", recoverable):
        assert store.get_or_create_session(_group()).session_id != "old-owner-session"
    recoverable.assert_not_called()


def test_telegram_config_extra_reaches_the_adapter():
    extra = {"group_toolsets": ["web"]}
    assert _adapter(extra).toolsets_for_source(_group()) == ["web"]
    assert _adapter(extra).toolsets_for_source(_dm()) is None
