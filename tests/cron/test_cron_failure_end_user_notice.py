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


def test_a_crashed_run_also_reads_one_line(monkeypatch, lang):
    """A run that raised out of run_job (disk full, a locked DB) is delivered by the crash path."""
    lang("ru")
    _mode(monkeypatch, True)
    sent = []
    monkeypatch.setattr(scheduler, "_deliver_result", lambda job, text, **kw: sent.append(text))
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets", lambda job, **kw: [{"platform": "telegram"}])
    scheduler._deliver_crash_failure(JOB, "OSError: [Errno 28] No space left on device", adapters=None, loop=None)
    assert sent == ["⚠️ Не получилось выполнить напоминание «Попить воды»."]


@pytest.mark.asyncio
@pytest.mark.parametrize("on", [False, True])
async def test_a_restart_interrupt_notice_names_no_cli_for_end_users(monkeypatch, lang, on):
    from unittest.mock import patch

    from gateway.run_config_loaders import GatewayConfigLoadersMixin
    from tests.gateway.test_cron_interrupt_notification import _bind_notifier, _telegram_target
    from tests.gateway.restart_test_helpers import make_restart_runner

    lang("ru")
    monkeypatch.setattr(GatewayConfigLoadersMixin, "_load_end_user_mode", staticmethod(lambda: on))
    runner, adapter = make_restart_runner()
    _bind_notifier(runner)
    with patch("cron.jobs.get_job", return_value=JOB), \
         patch("cron.scheduler._resolve_delivery_targets", return_value=[_telegram_target()]):
        assert await runner._notify_interrupted_cron_jobs([JOB["id"]]) == 1
    body = adapter.sent[0]
    assert ("hermes cron run" in body) is not on
    if on:
        assert body == "⚠️ Не получилось выполнить напоминание «Попить воды»."
