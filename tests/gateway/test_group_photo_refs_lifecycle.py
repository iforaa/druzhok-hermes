"""Photo authority expires across real turn orchestration and generation changes."""

import asyncio
import contextvars
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import group_image_refs as refs
from gateway.group_limits import REFERENCE_REFUSAL
from gateway.run import GatewayRunner
from gateway.session_context import set_session_vars
from gateway.session_state import SessionState
from gateway.turn_context import TurnContext
from tests.gateway.test_group_photo_refs import bound, dispatch, prepare

pytest_plugins = ("tests.gateway.test_group_photo_refs",)
from tests.gateway.test_group_images import GROUP_ID, _group


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["done", "exception", "cancel", "timeout"])
async def test_turn_runner_revokes_before_drain_and_on_every_exit(tmp_path, backend, monkeypatch, ending):
    runner, event, message = prepare_sync(tmp_path)
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    state = SessionState()
    runner._session_state = lambda key: state
    runner._peek_session_state = lambda key: state
    set_session_vars(platform="telegram", chat_id=GROUP_ID, chat_type="group", turn_event=event)
    runner._get_proxy_url = lambda: None
    display = SimpleNamespace(needs_progress_queue=False, log_mode_enabled=False, _native_slack_task_cards=False)
    runner._run_agent_display_settings = lambda source: display
    ctx = TurnContext(source=_group(), session_key="k", message=message, _run_still_current=lambda: True,
                      mute_notification_reply=True)
    runner._run_agent_build_turn_context = lambda *a, **kw: (ctx, SimpleNamespace(run_sync=lambda: None), None)
    runner._run_agent_bind_turn_wiring = lambda *a: {}
    for name in ("_run_agent_stream_consumer_task", "_run_agent_track_agent", "_run_agent_monitor_for_interrupt",
                 "_run_agent_finalize_streaming_tts", "_run_agent_mark_streamed_delivery"):
        setattr(runner, name, AsyncMock())
    runner._run_agent_evict_on_fallback = lambda *a: None
    runner._run_agent_schedule_bubble_cleanup = lambda *a: None
    copied, workers = [], []
    entered, release = threading.Event(), threading.Event()
    def late_worker():
        entered.set()
        assert release.wait(10)
        return dispatch({"image_url": label})
    def start(ctx, run_sync):
        copied.append(contextvars.copy_context())
        target = late_worker if ending in {"cancel", "timeout"} else lambda: dispatch({"image_url": label})
        task = asyncio.create_task(asyncio.to_thread(target))
        workers.append(task)
        return SimpleNamespace(executor_task=task)
    runner._run_agent_start_turn_worker = start
    async def await_worker(worker, *a):
        if ending in {"cancel", "timeout"}:
            assert await asyncio.to_thread(entered.wait, 10)
            if ending == "cancel":
                raise asyncio.CancelledError
            raise TimeoutError("worker timed out")
        result = await worker.executor_task
        assert result["success"], result
        ctx.result_holder[0] = {"messages": [], "completed": True}
        if ending == "exception":
            raise RuntimeError("worker failed")
        return result
    runner._run_agent_await_turn_worker = await_worker
    async def drain(*a):
        assert scope.revoked
        assert refs._current.get() is None
        assert REFERENCE_REFUSAL in copied[0].run(dispatch, {"image_url": label})["error"]
        return None, None
    runner._run_agent_drain_pending = drain
    async def cleanup(ctx, **tasks):
        for task in tasks.values():
            if task is not None:
                task.cancel()
        await asyncio.gather(*(task for task in tasks.values() if task is not None), return_exceptions=True)
    runner._run_agent_cleanup_turn_tasks = cleanup
    errors = {"exception": RuntimeError, "cancel": asyncio.CancelledError, "timeout": TimeoutError}
    call = runner._run_agent_inner(message=message, source=_group(), context_prompt="", history=[], session_key="k", session_id="test")
    if ending in errors:
        with pytest.raises(errors[ending]):
            await call
    else:
        await call
    if ending in {"cancel", "timeout"}:
        release.set()
        late_result = await asyncio.wait_for(workers[0], 10)
        assert REFERENCE_REFUSAL in late_result["error"]
    assert scope.revoked and not scope.photos
    assert refs._current.get() is None
    assert state.turn.group_photo_scope is None
    assert REFERENCE_REFUSAL in copied[0].run(dispatch, {"image_url": label})["error"]
    assert len(backend) == (0 if ending in {"cancel", "timeout"} else 1)


def prepare_sync(tmp_path):
    # prepare() owns asyncio.run; run its existing synchronous helper off the live test loop.
    from tests.gateway.test_group_photo_refs import png
    from tests.gateway.test_group_images import _event, _inbound_runner
    path = tmp_path / "photo.png"
    path.write_bytes(png())
    runner, _ = _inbound_runner({"group_media": ["image"], "group_toolsets": ["image_gen"]})
    event = _event(_group(), "redraw", media_urls=[str(path)], media_types=["image/png"])
    event._group_photo_scope = refs.snapshot_photos(event.source, [str(path)])
    return runner, event, "redraw"


def test_generation_loss_revokes_old_executor_before_successor(tmp_path, backend):
    runner, first, _ = prepare(tmp_path, "native")
    old_scope = first._group_photo_scope
    label = next(iter(old_scope.photos))
    assert bound(old_scope, {"image_url": label})["success"]
    state = SessionState()
    runner._session_state = lambda key: state
    runner._peek_session_state = lambda key: state
    generation = runner._begin_session_run_generation("k")
    old_turn = TurnContext(source=_group(), session_key="k", _run_still_current=runner._run_still_current_fn("k", generation))
    entered, release = threading.Event(), threading.Event()
    result = []
    def old_worker():
        entered.set()
        assert release.wait(10)
        result.append(dispatch({"image_url": label}))
    def run():
        set_session_vars(platform="telegram", chat_id=GROUP_ID, chat_type="group", turn_event=first)
        with refs.run_photo_scope(runner, old_turn):
            copied = contextvars.copy_context()
            thread = threading.Thread(target=copied.run, args=(old_worker,))
            thread.start()
            try:
                assert entered.wait(10)
                runner._begin_session_run_generation("k")
                assert old_scope.revoked
                _, second, _ = prepare(tmp_path, "native")
                new_label = next(iter(second._group_photo_scope.photos))
                assert bound(second._group_photo_scope, {"image_url": new_label})["success"]
                release.set()
                thread.join(10)
                assert not thread.is_alive()
            finally:
                release.set()
                thread.join(10)
    contextvars.copy_context().run(run)
    assert REFERENCE_REFUSAL in result[0]["error"]
    assert len(backend) == 2


def test_event_authority_consumed_once_and_eventless_followup_gets_none(tmp_path, backend):
    runner, event, _ = prepare(tmp_path, "native")
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    state = SessionState()
    runner._session_state = lambda key: state
    runner._peek_session_state = lambda key: state
    turn = TurnContext(source=_group(), session_key="k", _run_still_current=lambda: True)
    def run():
        set_session_vars(platform="telegram", chat_id=GROUP_ID, chat_type="group", turn_event=event)
        with refs.run_photo_scope(runner, turn):
            assert dispatch({"image_url": label})["success"]
        with refs.run_photo_scope(runner, turn):
            assert REFERENCE_REFUSAL in dispatch({"image_url": label})["error"]
        assert refs._current.get() is None
    contextvars.copy_context().run(run)
    assert len(backend) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["native", "text"])
async def test_busy_text_shares_labels_but_busy_photo_queues_its_own(tmp_path, backend, mode):
    from unittest.mock import Mock
    from tests.gateway.test_group_photo_refs import png
    from tests.gateway.test_group_images import _event

    runner, event, _ = await asyncio.to_thread(prepare, tmp_path, mode)
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    state = SessionState()
    runner._session_state = lambda key: state
    runner._peek_session_state = lambda key: state
    agent = SimpleNamespace(steer=Mock(return_value=True), _active_children=[])
    state.turn.agent, state.turn.event = agent, event
    turn = TurnContext(source=_group(), session_key="k", _run_still_current=lambda: True)
    set_session_vars(platform="telegram", chat_id=GROUP_ID, chat_type="group", turn_event=event)
    with refs.run_photo_scope(runner, turn):
        assert dispatch({"image_url": label})["success"]
        steered = await runner._resolve_busy_steer_or_redirect(_event(_group(), "now in blue"), "k", "steer", agent)
        assert steered.steered
        assert dispatch({"image_url": label})["success"]
        path = tmp_path / "busy.png"
        path.write_bytes(png("green"))
        busy = _event(_group(), "redraw this", media_urls=[str(path)], media_types=["image/png"])
        busy._native_media_origin = ("telegram", str(busy.source.chat_id))
        outcome = await runner._resolve_busy_steer_or_redirect(busy, "k", "steer", agent)
        assert not outcome.steered and outcome.effective_mode == "queue"
        runner._queue_or_replace_pending_event("k", busy)
        assert list(scope.photos) == [label]
        assert REFERENCE_REFUSAL in dispatch({"image_url": str(path)})["error"]
    adapter = runner._delivery_adapter_for(busy.source)
    queued = adapter._pending_messages.pop("k")
    runner._session_key_for_source = lambda source: "k"
    runner._pinned_channel_inputs = lambda key, prompt, source, **kw: (prompt, source)
    runner._persist_prompt_pins = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._intake_adapter_for = lambda source: None
    async def followup(**kw):
        next_scope = queued._group_photo_scope
        next_label = next(iter(next_scope.photos))
        next_turn = TurnContext(source=kw["source"], session_key=kw["session_key"], _run_still_current=lambda: True)
        with refs.run_photo_scope(runner, next_turn):
            assert (await asyncio.to_thread(dispatch, {"image_url": next_label}))["success"]
            assert REFERENCE_REFUSAL in (await asyncio.to_thread(dispatch, {"image_url": label}))["error"]
        return {"messages": [], "completed": True}
    runner._run_agent = followup
    from unittest.mock import patch
    with patch("tools.vision_tools.vision_analyze_tool", AsyncMock(return_value='{"success": true, "analysis": "green"}')):
        await runner._run_agent_queued_followup(
            turn, adapter, queued.text, queued, {}, {"messages": [], "interrupted": True}, None)
    assert len(backend) == 3
