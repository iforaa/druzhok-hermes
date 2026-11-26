"""Druzhok: pictures in an approved Telegram group. A picture ``image_generate`` made in the turn reaches
the group through the real Telegram send path; nothing else local does. Every refusal has a control
that would send, so a pass cannot mean the send path was simply broken."""

import asyncio
import contextvars
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import group_limits
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.group_limits import MEDIA_NOTE, REFERENCE_REFUSAL
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource

GROUP_ID = "-1001234"
OTHER_GROUP_ID = "-1009999"
OWNER_ID = "111"
STRANGER_ID = "222"
SESSION_KEY = "agent:main:telegram:group:-1001234:222"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _group(chat_id=GROUP_ID):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="group", user_id=STRANGER_ID)


def _dm():
    return SessionSource(platform=Platform.TELEGRAM, chat_id=OWNER_ID, chat_type="dm", user_id=OWNER_ID)


def _event(source, text="x", **kwargs):
    return MessageEvent(text=text, message_type=kwargs.pop("message_type", MessageType.TEXT), source=source, **kwargs)


def _adapter(extra=None):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra=extra or {}))
    adapter._bot = AsyncMock()
    for name in ("send_document", "send_photo", "send_video", "send_voice", "send_audio", "send_message"):
        setattr(adapter._bot, name, AsyncMock(return_value=SimpleNamespace(message_id=1)))
    adapter._bot.send_media_group = AsyncMock(return_value=[SimpleNamespace(message_id=1)])
    return adapter


def _runner(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)})
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.pairing_store = MagicMock()
    runner._thread_metadata_for_source = MagicMock(return_value={})
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    return runner


def _uploads(adapter):
    """Every file the fake bot was handed, by Telegram method."""
    names = ("send_document", "send_photo", "send_video", "send_voice", "send_audio", "send_media_group")
    return {name: getattr(adapter._bot, name).await_count for name in names if getattr(adapter._bot, name).await_count}


def _image_turn(*paths, success=True):
    """The rows one run appends: the user row, an ``image_generate`` call per path and its result."""
    rows = [{"role": "user", "content": "draw a cat"}]
    for i, path in enumerate(paths):
        call_id = f"call_{i}"
        rows.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "image_generate", "arguments": "{}"}}]})
        rows.append({"role": "tool", "tool_call_id": call_id,
                     "content": json.dumps({"success": success, "image": str(path)})})
    return rows


def _finish_turn(source, *paths, history=(), session_key=SESSION_KEY):
    messages = list(history) + _image_turn(*paths)
    group_limits.record_turn_outbox(source, session_key, {"messages": messages}, len(history), compressed=False)


@pytest.fixture(autouse=True)
def _clean_state():
    for state in (group_limits._outboxes, group_limits._validated, group_limits._approvals):
        state.clear()
    yield
    for state in (group_limits._outboxes, group_limits._validated, group_limits._approvals):
        state.clear()


@pytest.fixture
def picture(tmp_path):
    path = tmp_path / "cache" / "images" / "generated_cat.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(PNG)
    return path


@pytest.fixture
def owner_picture(tmp_path):
    path = tmp_path / "home" / "owner_photo.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(PNG)
    return path


async def _non_streamed(adapter, response, source=None, session_key=SESSION_KEY, between=None):
    """The non-streamed turn's delivery: ``_extract_response_content``, then (after the text send,
    ``between``) ``_deliver_attachments``."""
    event = _event(source or _group())
    extracted = await adapter._extract_response_content(response, event, session_key, is_ephemeral_response=False)
    if between is not None:
        between()
    await adapter._deliver_attachments(event, extracted, {}, anything_sent=True, record_delivery=lambda _result: None)
    return extracted


def _streamed(adapter, response, source=None, session_key=SESSION_KEY):
    runner = _runner(adapter)
    asyncio.run(runner._deliver_media_from_response(response, _event(source or _group()), adapter, session_key=session_key))


# ── a picture made this turn arrives ─────────────────────────────────────────────────────────────


def test_non_streamed_turn_sends_its_picture_to_the_group(picture):
    adapter = _adapter()
    _finish_turn(_group(), picture)
    asyncio.run(_non_streamed(adapter, f"Here it is\nMEDIA:{picture}"))
    assert adapter._bot.send_media_group.await_count == 1


def test_streamed_turn_sends_its_picture_to_the_group(picture):
    adapter = _adapter()
    _finish_turn(_group(), picture)
    _streamed(adapter, f"Here it is\nMEDIA:{picture}")
    assert adapter._bot.send_media_group.await_count == 1


def test_queued_lane_sends_its_picture_to_the_group(picture):
    adapter = _adapter()
    runner = _runner(adapter)
    _finish_turn(_group(), picture)
    delivered = asyncio.run(runner._deliver_queued_first_response(
        f"Here it is\nMEDIA:{picture}", source=_group(), adapter=adapter, metadata={}, text_already_delivered=True,
        session_key=SESSION_KEY))
    assert delivered and adapter._bot.send_media_group.await_count == 1


def test_picture_falls_back_to_a_single_photo_when_the_album_is_refused(picture):
    adapter = _adapter()
    adapter._bot.send_media_group = AsyncMock(side_effect=RuntimeError("Bad Request: wrong file"))
    _finish_turn(_group(), picture)
    _streamed(adapter, f"MEDIA:{picture}")
    assert adapter._bot.send_photo.await_count == 1


# ── nothing else local arrives ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("deliver", ["streamed", "non_streamed"])
def test_a_typed_path_to_a_real_file_is_refused(picture, owner_picture, deliver):
    adapter = _adapter()
    _finish_turn(_group(), picture)
    response = f"MEDIA:{owner_picture}\n{owner_picture}"
    if deliver == "streamed":
        _streamed(adapter, response)
    else:
        asyncio.run(_non_streamed(adapter, response))
    assert not _uploads(adapter)


def test_a_relative_path_to_a_real_file_is_refused(picture, owner_picture, monkeypatch):
    monkeypatch.chdir(owner_picture.parent)
    adapter = _adapter()
    _finish_turn(_group(), picture)
    asyncio.run(_non_streamed(adapter, f"MEDIA:./{owner_picture.name}\n./{owner_picture.name}"))
    assert not _uploads(adapter)


def test_an_older_file_and_a_previous_turns_picture_are_refused(tmp_path, picture):
    adapter = _adapter()
    older = tmp_path / "cache" / "images" / "older.png"
    older.write_bytes(PNG)
    _finish_turn(_group(), older)
    _streamed(adapter, f"MEDIA:{older}")
    assert adapter._bot.send_media_group.await_count == 1  # the control: it went out with its own turn

    _finish_turn(_group(), picture, history=_image_turn(older))
    _streamed(adapter, f"MEDIA:{older}\nMEDIA:{picture}")
    assert adapter._bot.send_media_group.await_count == 2
    sent = adapter._bot.send_media_group.await_args.kwargs["media"]
    assert len(sent) == 1  # this turn's picture only

    _finish_turn(_group())  # a turn that drew nothing
    _streamed(adapter, f"MEDIA:{picture}")
    assert adapter._bot.send_media_group.await_count == 2


def test_an_outbox_is_used_once(picture):
    adapter = _adapter()
    _finish_turn(_group(), picture)
    _streamed(adapter, f"MEDIA:{picture}")
    _streamed(adapter, f"MEDIA:{picture}")
    assert adapter._bot.send_media_group.await_count == 1


def test_a_non_image_from_image_generate_is_refused(tmp_path):
    adapter = _adapter()
    document = tmp_path / "result.pdf"
    document.write_bytes(b"%PDF-1.4")
    _finish_turn(_group(), document)
    _streamed(adapter, f"MEDIA:{document}")
    assert not _uploads(adapter)


def test_a_failed_image_generate_result_gives_no_outbox(picture):
    messages = _image_turn(picture, success=False)
    group_limits.record_turn_outbox(_group(), SESSION_KEY, {"messages": messages}, 0, compressed=False)
    adapter = _adapter()
    _streamed(adapter, f"MEDIA:{picture}")
    assert not _uploads(adapter)


def test_a_compressed_run_without_a_proven_turn_row_gives_no_outbox(picture):
    group_limits.record_turn_outbox(_group(), SESSION_KEY, {"messages": _image_turn(picture)}, 0, compressed=True)
    assert group_limits.take_turn_outbox(SESSION_KEY) == frozenset()
    proven = {"messages": _image_turn(picture), "current_turn_user_idx": 0}
    group_limits.record_turn_outbox(_group(), SESSION_KEY, proven, 0, compressed=True)
    assert group_limits.take_turn_outbox(SESSION_KEY) == {str(picture.resolve())}


def test_another_sessions_outbox_is_not_used(picture):
    adapter = _adapter()
    _finish_turn(_group(), picture, session_key="agent:main:telegram:group:-1001234:333")
    _streamed(adapter, f"MEDIA:{picture}")
    assert not _uploads(adapter)


@pytest.mark.parametrize("chat_id", [GROUP_ID, OTHER_GROUP_ID, "@public_channel"])
def test_a_send_that_skipped_drop_group_files_is_refused(picture, chat_id):
    """Background tasks, topics and notifications call the adapter directly: outside a reply's
    attachment delivery nothing is approved, even the picture that reply is about to send."""
    adapter = _adapter()
    _finish_turn(_group(), picture)
    group_limits.drop_group_files(_group(), [(str(picture), False)], [], SESSION_KEY)  # kept for GROUP_ID

    async def attempts():
        return [
            await adapter.send_image_file(chat_id=chat_id, image_path=str(picture)),
            await adapter.send_document(chat_id=chat_id, file_path=str(picture)),
            await adapter.send_multiple_images(chat_id=chat_id, images=[(f"file://{picture}", "")]),
        ]

    assert not any(result.success for result in asyncio.run(attempts()))
    assert not _uploads(adapter)


def test_an_approval_sends_once_and_only_to_its_chat(picture):
    adapter = _adapter()

    async def sends():
        with _approve(picture):
            other = await adapter.send_image_file(chat_id=OTHER_GROUP_ID, image_path=str(picture))
            return [other] + [await adapter.send_image_file(chat_id=GROUP_ID, image_path=str(picture)) for _ in range(2)]

    other, first, second = asyncio.run(sends())
    assert first.success and not second.success and not other.success
    assert adapter._bot.send_photo.await_count == 1


def test_an_approval_ends_with_its_delivery(picture):
    adapter = _adapter()
    with _approve(picture):
        pass
    assert not asyncio.run(adapter.send_image_file(chat_id=GROUP_ID, image_path=str(picture))).success
    assert not _uploads(adapter)


def test_a_slow_text_send_before_the_picture_does_not_lose_it(picture, monkeypatch):
    """A 60 s RetryAfter on the text: the picture is approved only when its own delivery starts."""
    adapter = _adapter()
    _finish_turn(_group(), picture)
    start = group_limits.time.monotonic()
    later = SimpleNamespace(monotonic=lambda: start + 300)
    asyncio.run(_non_streamed(adapter, f"Here it is\nMEDIA:{picture}",
                              between=lambda: monkeypatch.setattr(group_limits, "time", later)))
    assert adapter._bot.send_media_group.await_count == 1


def test_streamed_completion_delivers_the_runner_recorded_outbox(picture):
    """``_hmwa_deliver_turn_response`` for a streamed group turn (``already_sent``) reads the outbox the
    runner recorded under the turn's session key."""
    adapter = _adapter()
    runner = _runner(adapter)
    runner._should_send_voice_reply = MagicMock(return_value=False)
    runner._event_thread_metadata = MagicMock(return_value={})
    _finish_turn(_group(), picture)
    response = f"Here it is\nMEDIA:{picture}"
    entry = SimpleNamespace(session_id="s1")
    returned = asyncio.run(runner._hmwa_deliver_turn_response(
        _event(_group()), _group(), entry, SESSION_KEY, None, {"already_sent": True}, [], response, None, False))
    assert returned is None and adapter._bot.send_media_group.await_count == 1


@pytest.mark.parametrize("name", ["clip.mp4", "voice.ogg"])
def test_video_and_voice_stay_refused_even_from_this_turn(tmp_path, name):
    adapter = _adapter()
    media = tmp_path / name
    media.write_bytes(b"\x00" * 16)
    _finish_turn(_group(), media)
    _streamed(adapter, f"[[audio_as_voice]]\nMEDIA:{media}")
    assert not _uploads(adapter)


def test_a_dm_still_gets_any_file(owner_picture):
    adapter = _adapter()
    _streamed(adapter, f"MEDIA:{owner_picture}", source=_dm(), session_key="agent:main:telegram:dm:111")
    assert adapter._bot.send_media_group.await_count == 1


# ── image_generate draws from text only in a group ───────────────────────────────────────────────


def _image_generate(chat_type, args):
    from gateway.session_context import set_session_vars
    from tools import image_generation_tool as tool

    def call():
        set_session_vars(platform="telegram", chat_id=GROUP_ID if chat_type != "dm" else OWNER_ID, chat_type=chat_type)
        with patch.object(tool, "_dispatch_to_plugin_provider",
                          return_value=json.dumps({"success": True, "image": "https://example.test/x.png"})) as dispatch:
            return json.loads(tool._handle_image_generate({"prompt": "a cat", **args})), dispatch

    return contextvars.copy_context().run(call)


REFERENCES = [
    {"image_url": "/home/owner/photo.png"}, {"image_url": "notes/secret.png"}, {"image_url": "../workspace/x.jpg"},
    {"image_url": "~/photo.png"}, {"image_url": "file:///home/owner/photo.png"},
    {"image_url": "data:image/png;base64,AAAA"}, {"image_url": "https://example.test/a.png"},
    {"image_url": "HTTP://EXAMPLE.TEST/A.PNG"}, {"reference_image_urls": ["/home/owner/photo.png"]},
    {"reference_image_urls": ["", "https://example.test/a.png"]}, {"reference_image_urls": "/home/owner/photo.png"},
    {"reference_image_urls": [{"url": "/home/owner/photo.png"}]},
]


@pytest.mark.parametrize("args", REFERENCES)
@pytest.mark.parametrize("chat_type", ["group", "supergroup", "forum", ""])
def test_image_generate_refuses_every_reference_in_a_group(args, chat_type):
    result, dispatch = _image_generate(chat_type, args)
    assert REFERENCE_REFUSAL in result["error"]
    dispatch.assert_not_called()


def test_image_generate_draws_from_text_in_a_group():
    result, dispatch = _image_generate("group", {"image_url": "  ", "reference_image_urls": []})
    assert result["success"] and dispatch.call_count == 1


@pytest.mark.parametrize("args", [{"image_url": "https://example.test/a.png"}, {"reference_image_urls": ["https://example.test/a.png"]}])
def test_image_generate_keeps_references_in_a_dm(args):
    result, dispatch = _image_generate("dm", args)
    assert result["success"] and dispatch.call_count == 1


# ── seeing a group member's photo ────────────────────────────────────────────────────────────────


PHOTO_PATH = "/data/tenants/bot/cache/images/1_2_photo.jpg"


def _inbound_runner(extra, mode="native"):
    runner = _runner(_adapter(extra))
    runner._peek_session_state = MagicMock(return_value=None)
    runner._inbound_model_context_length = AsyncMock(return_value=100000)
    runner._decide_image_input_mode = MagicMock(return_value=mode)
    runner._resolve_session_agent_runtime = MagicMock(return_value=("m", {}))
    state = SimpleNamespace(persistent=SimpleNamespace(native_image_paths=[]))
    runner._session_state = MagicMock(return_value=state)
    return runner, state


def _photo(source):
    return _event(source, "what is this?", message_type=MessageType.PHOTO, media_urls=[PHOTO_PATH], media_types=["image/jpeg"])


def _prepare(runner, event):
    return asyncio.run(runner._prepare_inbound_message_text(event=event, source=event.source, history=[], session_key="k"))


def test_native_group_photo_is_attached_without_a_path(tmp_path):
    runner, state = _inbound_runner({"group_media": ["image"]})
    text = _prepare(runner, _photo(_group()))
    assert state.persistent.native_image_paths == [PHOTO_PATH]
    assert PHOTO_PATH not in text and "vision_analyze" not in text and MEDIA_NOTE not in text

    from agent.image_routing import build_native_content_parts
    photo = tmp_path / "1_2_photo.png"
    photo.write_bytes(PNG)
    parts, _ = build_native_content_parts(text, [str(photo)], hide_paths=True)
    assert any(part["type"] == "image_url" for part in parts)
    assert str(photo) not in json.dumps(parts[0]) and "Image attached" not in parts[0]["text"]


def test_native_turn_message_hides_the_path_only_in_a_group(tmp_path):
    from gateway.run_turn_runner import TurnRunner

    photo = tmp_path / "1_2_photo.png"
    photo.write_bytes(PNG)
    texts = {}
    for name, source in (("group", _group()), ("dm", _dm())):
        turn = object.__new__(TurnRunner)
        turn._ctx = SimpleNamespace(session_key="k", message="look", source=source)
        turn._runner = SimpleNamespace(_consume_pending_native_image_paths=lambda _key: [str(photo)])
        texts[name] = turn._native_image_run_message()[0]["text"]
    assert str(photo) not in texts["group"]
    assert f"[Image attached at: {photo}]" in texts["dm"]


@pytest.mark.parametrize("analysis", [{"success": True, "analysis": "A cat on a sofa."}, {"success": False}, RuntimeError("boom")])
def test_text_mode_group_description_has_no_path_and_no_tool_hint(analysis):
    runner, _state = _inbound_runner({"group_media": ["image"]}, mode="text")
    side = {"side_effect": analysis} if isinstance(analysis, Exception) else {"return_value": json.dumps(analysis)}
    with patch("tools.vision_tools.vision_analyze_tool", new=AsyncMock(**side)):
        text = _prepare(runner, _photo(_group()))
    assert "group member sent an image" in text
    assert PHOTO_PATH not in text and "vision_analyze" not in text and "image_url" not in text
    if isinstance(analysis, dict) and analysis["success"]:
        assert "A cat on a sofa." in text


def test_text_mode_dm_description_keeps_its_path_and_hint():
    runner, _state = _inbound_runner({}, mode="text")
    with patch("tools.vision_tools.vision_analyze_tool",
               new=AsyncMock(return_value=json.dumps({"success": True, "analysis": "A cat."}))):
        text = _prepare(runner, _photo(_dm()))
    assert PHOTO_PATH in text and "vision_analyze" in text


@pytest.mark.parametrize("extra", [{}, {"group_media": None}, {"group_media": []}, {"group_media": "image"},
                                   {"group_media": ["video", 7]}, {"group_media": {"image": True}}])
def test_without_image_in_group_media_a_photo_gets_the_media_note(extra):
    runner, _state = _inbound_runner(extra)
    runner._enrich_inbound_images = AsyncMock(return_value="ANALYSED")
    text = _prepare(runner, _photo(_group()))
    assert text.startswith(MEDIA_NOTE) and PHOTO_PATH not in text
    runner._enrich_inbound_images.assert_not_awaited()


def test_other_media_keep_the_note_when_images_are_seen():
    runner, _state = _inbound_runner({"group_media": ["image"]})
    doc = _event(_group(), "see", message_type=MessageType.DOCUMENT, media_urls=["/x/report.pdf"],
                 media_types=["application/pdf"])
    text = _prepare(runner, doc)
    assert text.startswith(MEDIA_NOTE) and "/x/" not in text


@pytest.mark.parametrize("value, expected", [
    (None, set()), ("image", set()), ([], set()), ({"image": 1}, set()), (["image"], {"image"}),
    ([" Image ", "video", 3, None], {"image"}),
])
def test_group_media_parsing_fails_closed(value, expected):
    assert group_limits.group_media(value) == expected


# ── an approved picture's own retries and fallbacks ──────────────────────────────────────────────


class _Flood(Exception):
    retry_after = 0


def _approve(picture):
    """A reply's delivery of ``picture``: kept by ``drop_group_files``, approved by ``group_delivery``."""
    _finish_turn(_group(), picture)
    kept, _ = group_limits.drop_group_files(_group(), [(str(picture), False)], [], SESSION_KEY)
    return group_limits.group_delivery(_group(), kept, [])


def test_a_flood_retry_of_an_approved_picture_is_sent(picture):
    adapter = _adapter()
    adapter._bot.send_photo = AsyncMock(side_effect=[_Flood("Flood control exceeded. Retry in 0 seconds"),
                                                     SimpleNamespace(message_id=2)])
    with _approve(picture):
        result = asyncio.run(adapter.send_image_file(chat_id=GROUP_ID, image_path=str(picture)))
    assert result.success and adapter._bot.send_photo.await_count == 2


def test_the_document_fallback_of_an_approved_picture_is_sent(picture):
    adapter = _adapter()
    adapter._bot.send_photo = AsyncMock(side_effect=RuntimeError("Photo_invalid_dimensions"))
    with _approve(picture):
        result = asyncio.run(adapter.send_image_file(chat_id=GROUP_ID, image_path=str(picture)))
    assert result.success and adapter._bot.send_document.await_count == 1


def test_an_approved_send_opens_nothing_for_the_next_send(picture, owner_picture):
    adapter = _adapter()

    async def one_task():
        first = await adapter.send_image_file(chat_id=GROUP_ID, image_path=str(picture))
        second = await adapter.send_document(chat_id=GROUP_ID, file_path=str(owner_picture))
        third = await adapter.send_multiple_images(chat_id=GROUP_ID, images=[(f"file://{owner_picture}", "")])
        return first, second, third

    with _approve(picture):
        first, second, third = asyncio.run(one_task())
    assert first.success and not second.success and not third.success
    assert adapter._bot.send_document.await_count == 0 and adapter._bot.send_media_group.await_count == 0


def test_a_text_media_tag_from_image_generate_is_not_an_outbox(picture):
    rows = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "image_generate", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": f"could not draw: MEDIA:{picture}"}]
    assert group_limits.turn_outbox(rows, 0) == frozenset()


def test_an_unreadable_native_group_photo_gets_a_note(tmp_path):
    from gateway.run_turn_runner import TurnRunner

    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(session_key="k", message="look", source=_group())
    turn._runner = SimpleNamespace(_consume_pending_native_image_paths=lambda _key: [str(tmp_path / "gone.png")])
    message = turn._native_image_run_message()
    assert message == f"{group_limits.GROUP_IMAGE_UNSEEN}\n\nlook" and "gone.png" not in message


# ── image_generate sees the group turn it runs in ────────────────────────────────────────────────


def _gateway_turn_tool_call(source, args):
    """``image_generate`` the way a gateway turn runs it: session vars bound by ``_set_session_env``,
    the turn on the gateway's executor (``copy_context``), the tool on the agent's tool worker thread
    (``propagate_context_to_thread``), dispatched through ``model_tools.handle_function_call``."""
    from concurrent.futures import ThreadPoolExecutor

    from gateway.session import SessionContext
    from model_tools import handle_function_call
    from tools import image_generation_tool as tool
    from tools.thread_context import propagate_context_to_thread

    runner = _runner(_adapter())
    context = SessionContext(source=source, connected_platforms=[Platform.TELEGRAM], home_channels={},
                             session_key=SESSION_KEY, session_id="s1")

    def turn():
        runner._set_session_env(context)
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(propagate_context_to_thread(
                lambda: handle_function_call("image_generate", {"prompt": "a cat", **args}))).result()

    async def gateway():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            return await loop.run_in_executor(pool, contextvars.copy_context().run, turn)

    with patch.object(tool, "_dispatch_to_plugin_provider",
                      return_value=json.dumps({"success": True, "image": "https://example.test/x.png"})) as dispatch:
        return json.loads(contextvars.copy_context().run(asyncio.run, gateway())), dispatch


def test_a_group_turns_tool_call_sees_the_group():
    result, dispatch = _gateway_turn_tool_call(_group(), {"image_url": "/home/owner/photo.png"})
    assert REFERENCE_REFUSAL in result["error"]
    dispatch.assert_not_called()


def test_a_dm_turns_tool_call_keeps_references():
    result, dispatch = _gateway_turn_tool_call(_dm(), {"image_url": "https://example.test/a.png"})
    assert result["success"] and dispatch.call_count == 1


def test_a_group_turn_with_no_chat_type_still_refuses():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=GROUP_ID, chat_type=None, user_id=STRANGER_ID)
    result, dispatch = _gateway_turn_tool_call(source, {"image_url": "/home/owner/photo.png"})
    assert REFERENCE_REFUSAL in result["error"]
    dispatch.assert_not_called()


def _scoped_call(monkeypatch, **binding):
    """``image_generate`` under a session binding other than a gateway message turn, in a process where
    the gateway has engaged session context."""
    from gateway import session_context
    from tools import image_generation_tool as tool

    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.setattr(session_context, "_session_context_engaged", True)

    def call():
        session_context.reset_session_vars()
        session_context.set_session_vars(**binding)
        with patch.object(tool, "_dispatch_to_plugin_provider",
                          return_value=json.dumps({"success": True, "image": "https://example.test/x.png"})) as dispatch:
            return json.loads(tool._handle_image_generate({"prompt": "a cat", "image_url": "/home/owner/photo.png"})), dispatch

    return contextvars.copy_context().run(call)


@pytest.mark.parametrize("binding", [
    {"platform": "", "session_key": "cron:job", "cron_session": "1"},  # cron/scheduler.py _CronRunScope
    {"source": "tui", "session_key": "tui:1", "session_id": "s1", "cron_session": ""},  # tui_gateway
    {"session_key": "acp:1", "session_id": "s1", "cron_session": ""},  # acp_adapter
])
def test_cron_tui_and_acp_calls_keep_their_references(monkeypatch, binding):
    result, dispatch = _scoped_call(monkeypatch, **binding)
    assert result["success"] and dispatch.call_count == 1


def test_a_bound_group_turn_event_refuses_even_without_a_platform(monkeypatch):
    result, dispatch = _scoped_call(monkeypatch, platform="", turn_event=_event(_group()))
    assert REFERENCE_REFUSAL in result["error"]
    dispatch.assert_not_called()
