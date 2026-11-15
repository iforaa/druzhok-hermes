"""Druzhok (display.end_user_mode): a failed cron run tells the chat one short localized line, with no
raw error, job id, `hermes` command or path. Off: upstream's operator notice is unchanged."""
import pytest

from agent import i18n
from cron import scheduler
from cron.scheduler import _compose_run_delivery

JOB = {"name": "Попить воды", "id": "10bcc04c5f22"}
GATEWAY_ERROR = "GatewayClientError: Gateway connection failed; no local fallback"


@pytest.fixture
def lang(monkeypatch):
    def use(code):
        monkeypatch.setenv("HERMES_LANGUAGE", code)
        i18n.reset_language_cache()
    yield use
    i18n.reset_language_cache()


def _mode(monkeypatch, on):
    monkeypatch.setattr(scheduler, "load_config_readonly", lambda: {"display": {"end_user_mode": on}})
    monkeypatch.setattr(scheduler, "load_config", dict)
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    monkeypatch.setattr(scheduler, "_upsert_incident_for_failure", lambda *a, **k: (False, None))


@pytest.mark.parametrize("error", [GATEWAY_ERROR, "[blocked_config] provider credential missing: no key",
                                   "Error code: 429 - rate limited", "idle for 900s (limit 600s)"])
def test_end_user_reads_one_russian_line(monkeypatch, lang, error):
    lang("ru")
    _mode(monkeypatch, True)
    text, *_ = _compose_run_delivery(JOB, success=False, error=error, final_response="", output_file=None)
    assert text == "⚠️ Не получилось выполнить напоминание «Попить воды»."


def test_operator_notice_is_unchanged_without_end_user_mode(monkeypatch, lang):
    lang("en")
    _mode(monkeypatch, False)
    text, *_ = _compose_run_delivery(JOB, success=False, error=GATEWAY_ERROR, final_response="", output_file=None)
    assert "hermes cron runs 10bcc04c5f22" in text


def test_a_successful_run_is_delivered_as_is(monkeypatch, lang):
    lang("ru")
    _mode(monkeypatch, True)
    text, *_ = _compose_run_delivery(JOB, success=True, error=None, final_response="Пора пить воду", output_file=None)
    assert text == "Пора пить воду"
