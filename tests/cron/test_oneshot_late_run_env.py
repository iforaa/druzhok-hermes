"""``HERMES_CRON_ONESHOT_LATE_S`` widens how late a due one-shot may still run (druzhok patch).

A bot that sleeps in a host queue can wake long after a reminder's time. With the env set the
run-time retire paths (due scan, run-claim marker, misfire backstop) let it fire late; without it
they keep the upstream 120 s window. Creation keeps refusing a time in the past either way.
"""

from datetime import UTC, datetime, timedelta

import pytest

from cron.env_settings import oneshot_late_run_seconds
from cron.jobs import create_job, get_due_jobs, load_jobs, save_jobs
from cron.scheduler_provider import fire_overdue_jobs

ENV = "HERMES_CRON_ONESHOT_LATE_S"
FIXED_NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC)
TEN_MIN_LATE = FIXED_NOW - timedelta(minutes=10)


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: FIXED_NOW)
    monkeypatch.delenv(ENV, raising=False)
    return tmp_path


def _oneshot(jid, run_at_dt):
    return {
        "id": jid,
        "name": jid,
        "prompt": "x",
        "schedule": {"kind": "once", "run_at": run_at_dt.isoformat()},
        "next_run_at": run_at_dt.isoformat(),
        "last_run_at": None,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 1, "completed": 0},
        "deliver": "local",
    }


class _RecordingProvider:
    def __init__(self):
        self.claimed = []

    def claim_fire(self, job_id):
        self.claimed.append(job_id)


def _run_backstop(monkeypatch, jobs):
    provider = _RecordingProvider()
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: jobs)
    monkeypatch.setattr("cron.jobs.is_job_runnable", lambda j: True)
    monkeypatch.setattr("cron.scheduler_provider._misfire_grace_minutes", lambda: 5.0)
    fire_overdue_jobs(provider, now=FIXED_NOW)
    return provider


class TestDueScan:
    def test_ten_minutes_late_runs_with_env(self, cron_store, monkeypatch):
        monkeypatch.setenv(ENV, "86400")
        save_jobs([_oneshot("late", TEN_MIN_LATE)])

        assert [d["id"] for d in get_due_jobs()] == ["late"]
        (stored,) = load_jobs()
        # A late-run window dispatch is not an outage dispatch.
        assert "outage" not in (stored.get("run_claim") or {})

    def test_ten_minutes_late_retired_without_env(self, cron_store):
        save_jobs([_oneshot("late", TEN_MIN_LATE)])

        assert get_due_jobs() == []
        assert load_jobs() == []

    def test_beyond_widened_window_still_retired(self, cron_store, monkeypatch):
        monkeypatch.setenv(ENV, "300")
        save_jobs([_oneshot("late", TEN_MIN_LATE)])

        assert get_due_jobs() == []
        assert load_jobs() == []


class TestMisfireBackstop:
    def test_ten_minutes_late_fires_with_env(self, monkeypatch):
        monkeypatch.setenv(ENV, "86400")
        provider = _run_backstop(monkeypatch, [_oneshot("late", TEN_MIN_LATE)])
        assert provider.claimed == ["late"]

    def test_ten_minutes_late_skipped_without_env(self, monkeypatch):
        monkeypatch.delenv(ENV, raising=False)
        provider = _run_backstop(monkeypatch, [_oneshot("late", TEN_MIN_LATE)])
        assert provider.claimed == []


class TestCreationStillRefusesPast:
    @pytest.mark.parametrize("env_value", [None, "86400"])
    def test_ten_minutes_in_past_refused(self, cron_store, monkeypatch, env_value):
        if env_value is not None:
            monkeypatch.setenv(ENV, env_value)
        with pytest.raises(ValueError, match="in the past"):
            create_job("remind me", TEN_MIN_LATE.isoformat(), deliver="local")
        assert load_jobs() == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 120), ("", 120), ("abc", 120), ("1.5", 120), ("-5", 120), ("60", 120),
     ("86400", 86400), (" 600 ", 600)],
)
def test_window_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, raw)
    assert oneshot_late_run_seconds() == expected
