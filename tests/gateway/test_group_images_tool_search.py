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
    GROUP_ID, SESSION_KEY, _adapter, _group, _non_streamed, _streamed, clear_group_state, write_png,
)
from tools.tool_search_catalog import TOOL_CALL_NAME


@pytest.fixture(autouse=True)
def _clean_state():
    clear_group_state()
    yield
    clear_group_state()


@pytest.fixture
def picture(tmp_path):
    return write_png(tmp_path / "cache" / "generated" / "images" / "openai_cat.png")


def _bridge_arguments(*prompts):
    return json.dumps({"calls": [{"name": "image_generate", "arguments": {"prompt": p}} for p in prompts]})


def _assistant(*calls):
    """An assistant row carrying ``(id, name, arguments)`` tool calls."""
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        for call_id, name, arguments in calls]}


def _image_result(call_id, path):
    """The result row the executor builds for the tool it ran (``make_tool_result_message``)."""
    content = json.dumps({"success": True, "image": str(path), "model": "gpt-image-1", "provider": "openai"})
    return make_tool_result_message("image_generate", content, call_id)


def _record(rows):
    group_limits.record_turn_outbox(_group(), SESSION_KEY, {"messages": rows}, 0, compressed=False)


@pytest.mark.parametrize("deliver", ["streamed", "non_streamed"])
def test_a_bridged_picture_reaches_the_group(picture, deliver):
    adapter = _adapter()
    _record([{"role": "user", "content": "нарисуй рыжего кота в космосе"},
             _assistant(("call_0", TOOL_CALL_NAME, _bridge_arguments("a ginger cat in space"))),
             _image_result("call_0", picture)])
    response = f"Готово\nMEDIA:{picture}"
    if deliver == "streamed":
        _streamed(adapter, response)
    else:
        asyncio.run(_non_streamed(adapter, response))
    assert adapter._bot.send_media_group.await_count == 1


def test_every_picture_of_a_split_bridge_batch_reaches_the_group(picture):
    """One ``tool_call`` naming ``image_generate`` twice is split into one bridge call per entry before it
    runs (``expand_local_tool_batches``); each child's result row pairs with its own new call id."""
    from agent.tool_call_batches import expand_local_tool_batches
    from agent.transports.types import ToolCall

    second = write_png(picture.parent / "openai_second.png")
    children = expand_local_tool_batches([ToolCall(id="call_0", name=TOOL_CALL_NAME, arguments=_bridge_arguments("a", "b"))])
    assert len(children) == 2 and len({child.id for child in children}) == 2
    _record([_assistant(*((child.id, child.name, child.arguments) for child in children)),
             _image_result(children[0].id, picture), _image_result(children[1].id, second)])
    adapter = _adapter()
    _streamed(adapter, f"MEDIA:{picture}\nMEDIA:{second}")
    assert len(adapter._bot.send_media_group.await_args.kwargs["media"]) == 2


def test_only_the_tool_the_executor_ran_counts(picture):
    """The bridge's arguments name ``image_generate``, but the executor recorded another tool: the row's
    content is not trusted, even when it looks like a picture result."""
    row = _image_result("call_0", picture)
    row["tool_name"] = "web_extract"
    rows = [_assistant(("call_0", TOOL_CALL_NAME, _bridge_arguments("x"))), row]
    assert group_limits.turn_outbox(rows, 0) == frozenset()


def test_the_bridge_keeps_the_reference_refusal_in_a_group(monkeypatch):
    """The refusal lives in the ``image_generate`` handler, which the bridge dispatches to."""
    import tools.image_generation_tool as tool
    from gateway.session_context import set_session_vars
    from model_tools import handle_function_call
    from tools.registry import invalidate_check_fn_cache, registry

    monkeypatch.setattr(registry.get_entry("image_generate"), "check_fn", lambda: True)  # a backend is set
    invalidate_check_fn_cache()
    arguments = {"calls": [{"name": "image_generate",
                            "arguments": {"prompt": "a cat", "image_url": "/home/owner/photo.png"}}]}

    def call():
        set_session_vars(platform="telegram", chat_id=GROUP_ID, chat_type="group")
        with patch.object(tool, "_dispatch_to_plugin_provider") as dispatch:
            result = handle_function_call(TOOL_CALL_NAME, arguments, enabled_toolsets=["web", "image_gen"])
            return json.loads(result), dispatch

    try:
        result, dispatch = contextvars.copy_context().run(call)
    finally:
        invalidate_check_fn_cache()
    assert REFERENCE_REFUSAL in result["error"]
    dispatch.assert_not_called()
