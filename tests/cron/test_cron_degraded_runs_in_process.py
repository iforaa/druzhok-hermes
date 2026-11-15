"""Druzhok: a cron fire that cannot get a restart-safe user scope runs inside the gateway, through its
in-process owner, instead of in a degraded worker whose only way to the owner is the gateway's loopback
WebSocket (dropped by druzhok's per-bot egress firewall). A scoped dispatch still uses the worker."""
from tests.cron.test_restart_safe_worker import _stub_external_worker_launch


def _dispatch(monkeypatch, mode):
    from tools.process_registry import GatewayChildDispatch
    monkeypatch.setattr("tools.process_registry.restart_safe_gateway_child_argv",
                        lambda command, **_: GatewayChildDispatch(mode, command))


def test_degraded_dispatch_runs_the_job_in_process(tmp_path, monkeypatch):
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    _dispatch(monkeypatch, "degraded")
    spawned, _payloads, handoff, _get = _stub_external_worker_launch(scheduler, monkeypatch)

    assert scheduler._launch_external_cron_worker({"id": "job-1", "execution_id": "exec-1", "prompt": "p"}) is False
    assert spawned == []
    handoff.assert_not_called()


def test_scoped_dispatch_still_hands_the_job_to_a_worker(tmp_path, monkeypatch):
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    _dispatch(monkeypatch, "scoped")
    spawned, _payloads, _handoff, _get = _stub_external_worker_launch(scheduler, monkeypatch)

    assert scheduler._launch_external_cron_worker({"id": "job-1", "execution_id": "exec-1", "prompt": "p"}) is True
    assert len(spawned) == 1
