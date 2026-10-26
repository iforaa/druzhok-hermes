"""display.end_user_mode: gateway-built replies without diagnostics for bots hosted for end users."""
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_config_loaders import GatewayConfigLoadersMixin
from tests.gateway.test_session_model_reset import _make_event, _make_runner, _make_source


def _set_mode(monkeypatch, on: bool):
    monkeypatch.setattr(GatewayConfigLoadersMixin, "_load_end_user_mode", staticmethod(lambda: on))


class TestLoader:
    def test_default_off(self, monkeypatch, tmp_path):
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_end_user_mode() is False

    def test_reads_config_yaml(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text("display:\n  end_user_mode: true\n")
        import gateway.run as gw
        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_end_user_mode() is True


@pytest.mark.asyncio
@pytest.mark.parametrize("on", [False, True])
async def test_new_hides_session_info_and_tip(monkeypatch, on):
    _set_mode(monkeypatch, on)
    runner = _make_runner()
    runner._format_session_info = lambda: "◆ Model: `test-model`"
    monkeypatch.setattr("hermes_cli.tips.get_random_tip", lambda: "run hermes -c")

    reply = await runner._handle_reset_command(_make_event("/new"))

    assert ("test-model" in reply) is not on
    assert ("hermes -c" in reply) is not on


@pytest.mark.asyncio
@pytest.mark.parametrize("on", [False, True])
async def test_start_greets_only_in_end_user_mode(monkeypatch, on):
    _set_mode(monkeypatch, on)
    runner = _make_runner()
    source = _make_source()

    handled, reply = await runner._hm_cmd_start(_make_event("/start"), source, "k")
    busy_reply = await runner._busy_start_command(_make_event("/start"), "k", source)

    assert handled is True
    assert bool(reply) is on
    assert bool(busy_reply) is on
    if on:
        assert reply == busy_reply


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/status", "/context"])
@pytest.mark.parametrize("on", [False, True])
async def test_busy_status_and_context_pass_the_gate_in_end_user_mode(monkeypatch, on, command):
    _set_mode(monkeypatch, on)
    runner = _make_runner()
    runner._handle_status_command = AsyncMock(return_value="status-internals")
    runner._handle_context_command = AsyncMock(return_value="status-internals")
    runner._check_slash_access = lambda _source, _name: "denied"
    event = MessageEvent(text=command, source=_make_source(), message_id="m2")

    handled, reply = await runner._hm_busy_slash_or_photo(event, _make_source(), "k")

    assert handled is True
    assert reply == ("denied" if on else "status-internals")


@pytest.mark.parametrize("on", [False, True])
def test_background_notice_drops_command_and_tail(monkeypatch, on):
    from gateway.run import _format_concise_process_notification

    _set_mode(monkeypatch, on)
    ok = _format_concise_process_notification("s1", "python secret_job.py --token x", 0, "")
    failed = _format_concise_process_notification("s1", "python secret_job.py", 2, "Traceback: boom")

    assert ("secret_job" in ok) is not on
    assert ("secret_job" in failed) is not on
    assert ("Traceback" in failed) is not on
    assert (" 2" in failed) is not on
