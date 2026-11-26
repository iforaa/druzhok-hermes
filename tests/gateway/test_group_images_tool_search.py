"""Druzhok: pictures in an approved Telegram group when ``image_generate`` is deferred behind tool search
(the default: it is in ``tools.tool_search.defer``). The model then calls the ``tool_call`` bridge, so the
assistant row names ``tool_call`` and only the result row, built by the executor, names the tool it ran.
Live regression: every such picture was dropped from the group."""

import asyncio
import contextvars
import json
from unittest.mock import patch

import pytest

from agent.tool_dispatch_helpers import make_tool_result_message
from gateway import group_limits
from gateway.group_limits import REFERENCE_REFUSAL
from tests.gateway.test_group_images import (
    GROUP_ID, PNG, SESSION_KEY, _adapter, _dm, _group, _non_streamed, _streamed, _uploads,
)
from tools.tool_search_catalog import TOOL_CALL_NAME


@pytest.fixture(autouse=True)
def _clean_state():
    states = (group_limits._outboxes, group_limits._validated, group_limits._approvals)
    for state in states:
        state.clear()
    yield
    for state in states:
        state.clear()


@pytest.fixture
def picture(tmp_path):
    path = tmp_path / "cache" / "generated" / "images" / "openai_cat.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(PNG)
    return path


@pytest.fixture
def owner_picture(tmp_path):
    path = tmp_path / "home" / "owner_photo.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(PNG)
    return path


def _bridge_call(call_id, *entries):
    """An assistant row calling the ``tool_call`` bridge with ``entries`` (``(name, arguments)``)."""
    calls = [{"name": name, "arguments": arguments} for name, arguments in entries]
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": TOOL_CALL_NAME, "arguments": json.dumps({"calls": calls})}}]}


def _image_result(path):
    return json.dumps({"success": True, "image": str(path), "model": "gpt-image-1", "provider": "openai"})


def _bridge_turn(*paths):
    """The rows one run appends when each picture is drawn through its own ``tool_call``: the result row
    is what the executor builds for the unwrapped tool (``make_tool_result_message``)."""
    rows = [{"role": "user", "content": "нарисуй рыжего кота в космосе"}]
    for i, path in enumerate(paths):
        call_id = f"call_{i}"
        rows.append(_bridge_call(call_id, ("image_generate", {"prompt": "a ginger cat in space"})))
        rows.append(make_tool_result_message("image_generate", _image_result(path), call_id))
    return rows


def _record(rows):
    group_limits.record_turn_outbox(_group(), SESSION_KEY, {"messages": rows}, 0, compressed=False)


# ── a picture drawn through the bridge arrives ───────────────────────────────────────────────────


@pytest.mark.parametrize("deliver", ["streamed", "non_streamed"])
def test_a_bridged_picture_reaches_the_group(picture, deliver):
    adapter = _adapter()
    _record(_bridge_turn(picture))
    response = f"Готово\nMEDIA:{picture}"
    if deliver == "streamed":
        _streamed(adapter, response)
    else:
        asyncio.run(_non_streamed(adapter, response))
    assert adapter._bot.send_media_group.await_count == 1


def test_pictures_from_several_bridge_calls_in_one_turn_all_arrive(tmp_path, picture):
    second = picture.parent / "openai_second.png"
    second.write_bytes(picture.read_bytes())
    adapter = _adapter()
    _record(_bridge_turn(picture, second))
    _streamed(adapter, f"MEDIA:{picture}\nMEDIA:{second}")
    assert len(adapter._bot.send_media_group.await_args.kwargs["media"]) == 2


# ── nothing else counts as a bridged picture ─────────────────────────────────────────────────────


def test_a_bridge_call_with_several_entries_runs_nothing_and_gives_no_outbox(image_backend, picture):
    """``tool_call`` runs one local tool per call: a batch naming ``image_generate`` is refused before
    any tool runs, and its result row is the bridge's own error."""
    from model_tools import handle_function_call

    entries = [("image_generate", {"prompt": "a"}), ("image_generate", {"prompt": "b"})]
    args = json.loads(_bridge_call("c", *entries)["tool_calls"][0]["function"]["arguments"])
    with patch("tools.image_generation_tool._dispatch_to_plugin_provider") as dispatch:
        error = handle_function_call(TOOL_CALL_NAME, args, enabled_toolsets=["web", "image_gen"])
    dispatch.assert_not_called()
    assert "error" in json.loads(error)
    rows = [_bridge_call("c", *entries), make_tool_result_message(TOOL_CALL_NAME, error, "c")]
    assert group_limits.turn_outbox(rows, 0) == frozenset()


@pytest.mark.parametrize("ran", [TOOL_CALL_NAME, "web_extract", "", None])
def test_a_bridge_row_that_did_not_run_image_generate_gives_no_outbox(picture, ran):
    """The bridge's arguments name ``image_generate``, but the executor recorded another tool (or none):
    the row's content is not trusted, even when it looks like a picture result."""
    row = make_tool_result_message("image_generate", _image_result(picture), "c")
    row["tool_name"] = ran
    rows = [_bridge_call("c", ("image_generate", {"prompt": "x"})), row]
    assert group_limits.turn_outbox(rows, 0) == frozenset()


def test_a_row_labelled_image_generate_under_another_call_gives_no_outbox(picture):
    rows = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "web_extract", "arguments": "{}"}}]},
        make_tool_result_message("image_generate", _image_result(picture), "c")]
    assert group_limits.turn_outbox(rows, 0) == frozenset()


def test_a_typed_owner_path_beside_a_bridged_picture_is_refused(picture, owner_picture):
    adapter = _adapter()
    _record(_bridge_turn(picture))
    _streamed(adapter, f"MEDIA:{owner_picture}")
    assert not _uploads(adapter)


# ── the reference refusal holds through the bridge ───────────────────────────────────────────────


@pytest.fixture
def image_backend(monkeypatch):
    """An image backend is configured, as on a bot with image generation on."""
    import tools.image_generation_tool  # registers image_generate
    from tools.registry import invalidate_check_fn_cache, registry

    monkeypatch.setattr(registry.get_entry("image_generate"), "check_fn", lambda: True)
    invalidate_check_fn_cache()
    yield
    invalidate_check_fn_cache()


def _bridged_image_generate(chat_type, arguments):
    """``image_generate`` reached through ``model_tools.handle_function_call("tool_call", ...)`` in a turn
    bound to ``chat_type``, with the bot's group toolsets."""
    from gateway.session_context import set_session_vars
    from model_tools import handle_function_call
    from tools import image_generation_tool as tool

    def call():
        set_session_vars(platform="telegram", chat_id=GROUP_ID if chat_type != "dm" else "111", chat_type=chat_type)
        with patch.object(tool, "_dispatch_to_plugin_provider",
                          return_value=json.dumps({"success": True, "image": "https://example.test/x.png"})) as dispatch:
            result = handle_function_call(
                TOOL_CALL_NAME, {"calls": [{"name": "image_generate", "arguments": {"prompt": "a cat", **arguments}}]},
                enabled_toolsets=["web", "image_gen"])
            return json.loads(result), dispatch

    return contextvars.copy_context().run(call)


@pytest.mark.parametrize("arguments", [
    {"image_url": "/home/owner/photo.png"}, {"image_url": "notes/secret.png"},
    {"image_url": "https://example.test/a.png"}, {"reference_image_urls": ["/home/owner/photo.png"]},
])
def test_image_generate_through_the_bridge_refuses_references_in_a_group(image_backend, arguments):
    result, dispatch = _bridged_image_generate("group", arguments)
    assert REFERENCE_REFUSAL in result["error"]
    dispatch.assert_not_called()


def test_image_generate_through_the_bridge_draws_from_text_in_a_group(image_backend):
    result, dispatch = _bridged_image_generate("group", {})
    assert result["success"] and dispatch.call_count == 1


def test_image_generate_through_the_bridge_keeps_references_in_a_dm(image_backend):
    assert _dm().chat_type == "dm"
    result, dispatch = _bridged_image_generate("dm", {"image_url": "https://example.test/a.png"})
    assert result["success"] and dispatch.call_count == 1
