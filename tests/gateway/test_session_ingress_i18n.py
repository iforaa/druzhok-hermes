"""Druzhok: session-ingress notices come from the locale catalog, and end users get a lost-turn
notice that names neither Hermes nor /discard (the tenant tier refuses it)."""
from types import SimpleNamespace

import pytest

from agent import i18n
from gateway.config import Platform
from gateway.run_config_loaders import GatewayConfigLoadersMixin
from gateway.session_ingress import pause_notice, queue_full_reply


@pytest.fixture
def lang(monkeypatch):
    def use(code):
        monkeypatch.setenv("HERMES_LANGUAGE", code)
        i18n.reset_language_cache()
    yield use
    i18n.reset_language_cache()


def _notice(reason, platform=Platform.TELEGRAM):
    live = SimpleNamespace(pause_notified=False, source=SimpleNamespace(platform=platform))
    return pause_notice(SimpleNamespace(sessions={"s": live}), SimpleNamespace(session_id="s"), reason)


def _end_user(monkeypatch, on):
    monkeypatch.setattr(GatewayConfigLoadersMixin, "_load_end_user_mode", staticmethod(lambda: on))


@pytest.mark.parametrize("code", ["en", "ru"])
def test_end_user_lost_turn_points_at_new(monkeypatch, lang, code):
    lang(code)
    _end_user(monkeypatch, True)
    notice = _notice("unknown_execution")
    assert "/new" in notice
    assert "/discard" not in notice and "Hermes" not in notice


def test_lost_turn_keeps_discard_without_end_user_mode(monkeypatch, lang):
    lang("en")
    _end_user(monkeypatch, False)
    assert "Send /discard" in _notice("unknown_execution")
    assert "/hermes discard" in _notice("unknown_execution", platform=Platform.SLACK)


@pytest.mark.parametrize("reason", ["unknown_execution", "runtime_draining", "session_busy", "not_authorized"])
def test_russian_tenants_read_russian(monkeypatch, lang, reason):
    lang("ru")
    _end_user(monkeypatch, False)
    english = _english(monkeypatch, lang, reason)
    lang("ru")
    notice = _notice(reason)
    assert notice != english and not notice.startswith("gateway.")
    assert "Разговор" in notice or "сообщение" in notice


def _english(monkeypatch, lang, reason):
    lang("en")
    return _notice(reason)


def test_queue_full_reply_is_localized(lang):
    lang("en")
    english = queue_full_reply()
    lang("ru")
    assert queue_full_reply() != english
    assert "сообщений" in queue_full_reply()
