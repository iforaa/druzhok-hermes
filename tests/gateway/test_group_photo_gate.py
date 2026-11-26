"""Ship-gate regressions: album chat boundaries and the watchdog/asyncio timeout window."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway import group_image_refs as refs
from gateway.group_limits import REFERENCE_REFUSAL
from gateway.platforms.event import MessageType
from gateway.session_context import set_session_vars
from gateway.session_state import SessionState
from gateway.turn_context import TurnContext
from tests.gateway.test_group_images import GROUP_ID, OTHER_GROUP_ID, _adapter, _event, _group, _inbound_runner, _prepare
from tests.gateway.test_group_photo_refs import bound, dispatch, png, prepare

pytest_plugins = ("tests.gateway.test_group_photo_refs",)


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["photo", "document"])
async def test_overlapping_album_ids_keep_each_chats_attachments_and_labels(tmp_path, backend, monkeypatch, lane):
    from plugins.platforms.telegram import telegram_media_group as albums

    adapter = _adapter({"group_media": ["image"], "group_toolsets": ["image_gen"]})
    monkeypatch.setattr(albums, "media_group_wait_seconds", lambda adapter: 0)
    delivered, ready = {}, {chat: asyncio.Event() for chat in (GROUP_ID, OTHER_GROUP_ID)}
    real_handle = adapter.handle_message
    async def handle(event):
        await real_handle(event)  # real native-origin stamp, before gateway preparation
        delivered[event.source.chat_id] = event
        ready[event.source.chat_id].set()
    adapter.handle_message = handle
    messages, events, originals = [], [], {}
    for chat, color in [(GROUP_ID, "red"), (OTHER_GROUP_ID, "green")]:
        path = tmp_path / f"{chat}.png"
        path.write_bytes(png(color))
        originals[chat] = path.read_bytes()
        messages.append(SimpleNamespace(chat=SimpleNamespace(id=chat), media_group_id="shared-album"))
        events.append(_event(_group(chat), "redraw", message_type=MessageType.PHOTO,
            media_urls=[str(path)], media_types=["image/png"]))
    async def queue(msg, event):
        if lane == "photo":
            await adapter._route_photo_event(msg, event)
        else:
            # Document albums enter this same queue after the document handler's download.
            await adapter._queue_media_group_event(msg.media_group_id, event)
    try:
        with albums.tracking_album_item(adapter, messages[0]):
            await queue(messages[0], events[0])
            with albums.tracking_album_item(adapter, messages[1]):
                await queue(messages[1], events[1])
                assert len(adapter._media_group_events) == len(adapter._media_group_tasks) == 2
            # A pending download in chat A cannot hold chat B's album flush.
            await asyncio.wait_for(ready[OTHER_GROUP_ID].wait(), 3)
            assert not ready[GROUP_ID].is_set()
        await asyncio.wait_for(ready[GROUP_ID].wait(), 3)
        await asyncio.gather(*list(adapter._media_group_tasks.values()))
    finally:
        tasks = list(adapter._media_group_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert set(delivered) == {GROUP_ID, OTHER_GROUP_ID}
    scopes = {}
    for chat, event in delivered.items():
        assert event.media_urls == [str(tmp_path / f"{chat}.png")]
        assert event._native_media_origin == ("telegram", chat)
        runner, _ = _inbound_runner({"group_media": ["image"], "group_toolsets": ["image_gen"]})
        await asyncio.to_thread(_prepare, runner, event)
        scopes[chat] = event._group_photo_scope
    for chat, scope in scopes.items():
        label = next(iter(scope.photos))
        other = OTHER_GROUP_ID if chat == GROUP_ID else GROUP_ID
        foreign = next(iter(scopes[other].photos))
        assert bound(scope, {"image_url": label}, chat_id=chat)["success"]
        assert backend[-1][1] == [originals[chat]]
        assert REFERENCE_REFUSAL in bound(scope, {"image_url": foreign}, chat_id=chat)["error"]
    assert len(backend) == 2


@pytest.mark.asyncio
async def test_watchdog_revokes_before_asyncio_waiter_cleanup(tmp_path, backend, monkeypatch):
    import gateway.run as gateway_run

    runner, event, _ = await asyncio.to_thread(prepare, tmp_path, "native")
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    state = SessionState()
    runner._session_state = lambda key: state
    runner._peek_session_state = lambda key: state
    runner._run_in_executor_with_context = lambda fn: asyncio.to_thread(fn)
    monkeypatch.setattr(gateway_run, "_float_env", lambda name, default: 1 if name == "HERMES_AGENT_TIMEOUT" else 0)
    monkeypatch.setattr(gateway_run, "_reap_gateway_turn_processes", lambda *a, **kw: 0)
    entered, release_worker, fire_watchdog, marked, finish_watchdog, watchdog_done = [threading.Event() for _ in range(6)]
    real_watchdog = gateway_run._watch_gateway_turn_inactivity
    def watchdog(**kw):
        try:
            assert fire_watchdog.wait(10)
            kw["poll_interval"] = 0.01
            real_watchdog(**kw)
        finally:
            watchdog_done.set()
    def after_mark(task_id):
        assert scope.revoked
        assert worker.timeout_fired.is_set()
        marked.set()
        assert finish_watchdog.wait(10)
    monkeypatch.setattr(gateway_run, "_watch_gateway_turn_inactivity", watchdog)
    monkeypatch.setattr(gateway_run, "_dump_wedged_turn_stacks", after_mark)
    turn = TurnContext(source=_group(), session_key="k", session_id="gate", _run_still_current=lambda: True)
    turn.agent_holder[0] = SimpleNamespace(get_activity_summary=lambda: {"seconds_since_activity": 100})
    monkeypatch.setattr(gateway_run, "request_hard_interrupt", lambda *a, **kw: None)
    def run_sync():
        entered.set()
        assert release_worker.wait(10)
        return dispatch({"image_url": label})
    set_session_vars(platform="telegram", chat_id=GROUP_ID, chat_type="group", turn_event=event)
    with refs.run_photo_scope(runner, turn):
        worker = runner._run_agent_start_turn_worker(turn, run_sync)
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            fire_watchdog.set()
            assert await asyncio.to_thread(marked.wait, 10)
            # No asyncio timeout waiter or turn cleanup has run; the worker still owns its copied context.
            release_worker.set()
            result = await asyncio.wait_for(worker.executor_task, 10)
            assert REFERENCE_REFUSAL in result["error"]
            assert not backend
        finally:
            release_worker.set()
            fire_watchdog.set()
            finish_watchdog.set()
            await asyncio.gather(worker.executor_task, return_exceptions=True)
            assert await asyncio.to_thread(watchdog_done.wait, 10)
