"""Druzhok ``sites`` toolset: per-bot gate, saved lists, and the contract with druzhok's /v2/sites."""
import gzip
import http.server
import io
import json
import os
import tarfile
import threading
from types import SimpleNamespace

import pytest

import model_tools
import tools.site_tools  # registers the tools
from gateway import run_turn
from hermes_cli.tools_config import _get_platform_tools, _recover_platform_native_toolsets
from tools.registry import registry
from tools.site_tools import _MAX_BYTES, _MAX_DEPTH, _MAX_DIRS, _MAX_FILES

KEY = "tenant-key-not-for-output"
SITE_TOOLS = {"site_publish", "site_delete"}


def tool_names(toolsets):
    return {t["function"]["name"] for t in model_tools.get_tool_definitions(enabled_toolsets=sorted(toolsets), quiet_mode=False)}


class Druzhok:
    """A real HTTP server standing in for druzhok: records requests, answers what the test set."""

    def __init__(self):
        self.requests = []
        self.status = 201
        self.reply = json.dumps({"url": "https://vasya.oldey.dev/shop/", "files": 3, "bytes": 10, "published_at": "2026-11-01T12:00:00Z"})
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _serve(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                outer.requests.append(SimpleNamespace(method=self.command, path=self.path, headers=self.headers, body=body))
                payload = outer.reply.encode()
                self.send_response(outer.status)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_PUT = do_DELETE = _serve

            def log_message(self, *_args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def druzhok(monkeypatch):
    server = Druzhok()
    monkeypatch.setenv("DRUZHOK_SITES_ENABLED", "1")
    monkeypatch.setenv("FIRECRAWL_API_URL", server.url)
    monkeypatch.setenv("FIRECRAWL_API_KEY", KEY)
    yield server
    server.close()


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    folder = tmp_path / "sites" / "shop"
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<h1>Магазин</h1>", encoding="utf-8")
    return folder


def call(name, **args):
    return json.loads(registry.dispatch(name, args))


def entries(request):
    with tarfile.open(fileobj=io.BytesIO(request.body), mode="r:gz") as tar:
        return {m.name: (m, tar.extractfile(m).read() if m.isreg() else None) for m in tar.getmembers()}


# ---- who gets the tools ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "", "0", "true", "yes"])
def test_no_tools_and_no_schema_unless_the_env_is_exactly_1(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("DRUZHOK_SITES_ENABLED", raising=False)
    else:
        monkeypatch.setenv("DRUZHOK_SITES_ENABLED", value)
    assert not SITE_TOOLS & tool_names(["hermes-telegram"])
    assert not SITE_TOOLS & tool_names(["sites"])


def test_env_on_offers_both_tools_in_a_dm_with_no_saved_list(monkeypatch):
    monkeypatch.setenv("DRUZHOK_SITES_ENABLED", "1")
    enabled = _get_platform_tools({}, "telegram")
    assert "sites" in enabled
    assert SITE_TOOLS <= tool_names(enabled)


def test_other_platforms_defaults_do_not_get_it():
    assert "sites" not in _get_platform_tools({}, "cli")
    assert "sites" not in _get_platform_tools({}, "cron")
    assert "sites" not in _get_platform_tools({}, "discord")


@pytest.mark.parametrize("saved", [["web"], ["web", "file", "terminal"], []])
def test_recovery_does_not_add_it_to_a_saved_list(saved):
    enabled = set(saved)
    _recover_platform_native_toolsets(enabled, "telegram", skip=set())
    assert "sites" not in enabled
    assert "sites" not in _get_platform_tools({"platform_toolsets": {"telegram": saved}}, "telegram")


def resolve_for_group(config, override):
    runner = SimpleNamespace(_delivery_adapter_for=lambda _source: SimpleNamespace(toolsets_for_source=lambda _s: override))
    source = SimpleNamespace(chat_type="group", chat_id="-100", platform="telegram")
    return run_turn.GatewayTurnMixin._resolve_enabled_toolsets_for_source(runner, config, source, "telegram")


@pytest.mark.parametrize(
    ("config", "override"),
    [
        ({"platform_toolsets": {"telegram": ["web", "file"]}}, None),
        ({}, ["web"]),
        ({"platform_toolsets": {"telegram": ["web", "file"]}}, ["web"]),
    ],
)
def test_saved_list_or_override_without_sites_keeps_site_tools_out_of_a_group_turn(monkeypatch, config, override):
    monkeypatch.setenv("DRUZHOK_SITES_ENABLED", "1")
    enabled = resolve_for_group(config, override)
    assert "sites" not in enabled
    assert not SITE_TOOLS & tool_names(enabled)


# ---- site_publish ---------------------------------------------------------------------------------

def test_publish_sends_the_contract_request_and_returns_the_url(druzhok, site):
    (site / "assets").mkdir()
    (site / "assets" / "стиль.css").write_text("/* ё */", encoding="utf-8")
    result = call("site_publish", site="shop")
    assert result["success"] is True
    assert result["url"] == "https://vasya.oldey.dev/shop/"
    assert "skipped_count" not in result

    request = druzhok.requests[0]
    assert (request.method, request.path) == ("PUT", "/v2/sites/shop")
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert request.headers["Content-Type"] == "application/gzip"
    assert request.headers["Content-Length"] == str(len(request.body))
    gzip.decompress(request.body)
    got = entries(request)
    assert set(got) == {"index.html", "assets/стиль.css"}, "files sit at the archive root, with no directory entries"
    assert got["index.html"][1].decode("utf-8") == "<h1>Магазин</h1>"
    assert KEY not in json.dumps(result)


def test_publish_leaves_out_dotfiles_and_says_so(druzhok, site):
    (site / ".DS_Store").write_bytes(b"x")
    (site / ".git").mkdir()
    (site / ".git" / "config").write_text("secret")
    (site / "css").mkdir()
    (site / "css" / ".hidden").write_text("x")
    result = call("site_publish", site="shop")
    assert result["success"] is True
    assert set(entries(druzhok.requests[0])) == {"index.html"}
    assert result["skipped_count"] == 3
    assert ".DS_Store" in result["skipped_dotfiles"]


def test_a_hard_link_refuses_locally_wherever_its_other_name_is(druzhok, site, tmp_path):
    secret = tmp_path / "MEMORY.md"
    secret.write_text("secret")
    os.link(secret, site / "a.txt")
    assert "hard link" in call("site_publish", site="shop")["error"]
    assert druzhok.requests == []


@pytest.mark.parametrize("target", ["file", "dir"])
def test_a_symlink_refuses_locally_and_never_reads_the_target(druzhok, site, tmp_path, target):
    secret = tmp_path / "outside"
    if target == "dir":
        secret.mkdir()
        (secret / "MEMORY.md").write_text("secret")
    else:
        secret.write_text("secret")
    (site / "link").symlink_to(secret)
    result = call("site_publish", site="shop")
    assert "symbolic link" in result["error"]
    assert druzhok.requests == []


def test_the_site_folder_itself_being_a_symlink_refuses(druzhok, site, tmp_path):
    real = tmp_path / "memories"
    real.mkdir()
    (real / "index.html").write_text("secret")
    (site.parent / "leak").symlink_to(real)
    result = call("site_publish", site="leak")
    assert "symbolic link" in result["error"]
    assert druzhok.requests == []


def test_the_sites_parent_being_a_symlink_refuses(druzhok, tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    real = tmp_path / "memories" / "shop"
    real.mkdir(parents=True)
    (real / "index.html").write_text("secret")
    (tmp_path / "sites").symlink_to(tmp_path / "memories")
    assert "symbolic link" in call("site_publish", site="shop")["error"]
    assert druzhok.requests == []


def test_a_fifo_refuses_without_blocking(druzhok, site):
    os.mkfifo(site / "pipe")
    assert "special file" in call("site_publish", site="shop")["error"]
    assert druzhok.requests == []


def test_missing_folder_and_missing_index_refuse_locally(druzhok, site):
    assert "no sites/nope/ folder" in call("site_publish", site="nope")["error"]
    (site / "index.html").unlink()
    (site / "sub").mkdir()
    (site / "sub" / "index.html").write_text("x")
    assert "no index.html" in call("site_publish", site="shop")["error"]
    assert druzhok.requests == []


def test_file_and_size_limits_refuse_locally(druzhok, site):
    for i in range(_MAX_FILES):
        (site / f"{i}.txt").write_text("x")
    assert "over the limit" in call("site_publish", site="shop")["error"]
    for i in range(_MAX_FILES):
        (site / f"{i}.txt").unlink()
    with open(site / "big.bin", "wb") as f:
        f.truncate(_MAX_BYTES + 1)
    assert "over the limit" in call("site_publish", site="shop")["error"]
    assert druzhok.requests == []


def test_exactly_at_the_limits_is_sent(druzhok, site):
    for i in range(_MAX_FILES - 1):
        (site / f"{i}.txt").write_text("x")
    assert call("site_publish", site="shop")["success"] is True
    assert len(entries(druzhok.requests[0])) == _MAX_FILES


def test_exactly_the_byte_limit_is_sent(druzhok, site):
    used = (site / "index.html").stat().st_size
    with open(site / "big.bin", "wb") as f:
        f.truncate(_MAX_BYTES - used)
    assert call("site_publish", site="shop")["success"] is True


def test_too_many_folders_refuses_locally(druzhok, site):
    for i in range(_MAX_DIRS + 1):
        (site / f"d{i}").mkdir()
    assert "folders" in call("site_publish", site="shop")["error"]
    assert druzhok.requests == []


def test_depth_at_the_limit_is_sent_and_past_it_refuses(druzhok, site):
    folder = site
    for _ in range(_MAX_DEPTH - 1):
        folder = folder / "d"
    folder.mkdir(parents=True)
    (folder / "f.txt").write_text("x")
    assert call("site_publish", site="shop")["success"] is True
    (folder / "d").mkdir()
    (folder / "d" / "f.txt").write_text("x")
    assert "too long or too deep" in call("site_publish", site="shop")["error"]
    assert len(druzhok.requests) == 1


def test_a_path_past_the_byte_limit_refuses_locally(druzhok, site):
    folder = site
    for _ in range(3):
        folder = folder / ("d" * 170)
    folder.mkdir(parents=True)
    (folder / "f.txt").write_text("x")
    assert "too long or too deep" in call("site_publish", site="shop")["error"]
    assert druzhok.requests == []


@pytest.mark.parametrize("name", ["bad\\name.html", "bad\nname.html", "x" * 201 + ".html"])
def test_names_the_gateway_would_refuse_are_named_locally(druzhok, site, name):
    (site / name).write_text("x")
    assert name[:3] in call("site_publish", site="shop")["error"]
    assert druzhok.requests == []


@pytest.mark.parametrize("name", ["", "../x", "A", "a/b", "-a", "a" * 51, "shop\n", 5])
def test_bad_site_names_never_touch_disk_or_network(druzhok, site, name):
    assert "site name" in call("site_publish", site=name)["error"]
    assert "site name" in call("site_delete", site=name)["error"]
    assert druzhok.requests == []


@pytest.mark.parametrize(
    ("status", "code", "text"),
    [
        (400, "bad_archive", "refused the files"),
        (400, "bad_name", "refused this name"),
        (403, "not_partner_account", "cannot publish"),
        (403, "slug_reserved", "reserved"),
        (403, "slug_taken", "Another bot"),
        (403, "site_disabled", "disabled by the operator"),
        (403, "sites_off", "turned off"),
        (413, "too_large", "over the limit"),
        (429, "rate_limited", "20 an hour"),
        (401, "invalid_api_key", "did not accept"),
        (401, "key_revoked", "did not accept"),
        (403, "account_suspended", "suspended"),
        (408, "upload_incomplete", "did not finish"),
        (500, "internal_error", "internal error"),
        (503, "sites_unavailable", "unavailable"),
        (502, "gateway_unavailable", "result is unknown"),
    ],
)
def test_every_service_code_becomes_plain_language(druzhok, site, status, code, text):
    druzhok.status = status
    druzhok.reply = json.dumps({"error": code})
    result = call("site_publish", site="shop")
    assert text in result["error"]
    assert result["code"] == code
    assert "url" not in result
    assert KEY not in json.dumps(result)


@pytest.mark.parametrize(("status", "reply"), [(500, "<html>oops</html>"), (418, '{"error":"new_code"}'), (201, "[]"), (201, "{}")])
def test_an_unexpected_answer_is_an_error_with_the_status(druzhok, site, status, reply):
    druzhok.status = status
    druzhok.reply = reply
    assert f"HTTP {status}" in call("site_publish", site="shop")["error"]


def test_no_server_is_a_plain_error_without_the_key(druzhok, site, monkeypatch):
    druzhok.close()
    druzhok.server = SimpleNamespace(shutdown=lambda: None, server_close=lambda: None)
    result = call("site_publish", site="shop")
    assert "could not be reached" in result["error"]
    assert KEY not in json.dumps(result)


def test_trailing_slash_in_the_base_url(druzhok, site, monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_URL", druzhok.url + "/")
    call("site_publish", site="shop")
    assert druzhok.requests[0].path == "/v2/sites/shop"


@pytest.mark.parametrize("missing", ["FIRECRAWL_API_URL", "FIRECRAWL_API_KEY"])
def test_either_setting_missing_refuses_before_any_request(druzhok, site, monkeypatch, missing):
    monkeypatch.delenv(missing)
    assert "not set up" in call("site_publish", site="shop")["error"]
    assert "not set up" in call("site_delete", site="shop")["error"]
    assert druzhok.requests == []


def test_the_reserved_name_refuses_before_reading_files(druzhok, site):
    (site.parent / "media").mkdir()
    (site.parent / "media" / "index.html").write_text("x")
    assert "reserved" in call("site_publish", site="media")["error"]
    assert druzhok.requests == []


# ---- site_delete ----------------------------------------------------------------------------------

def test_delete_uses_the_same_path_and_keeps_the_files(druzhok, site):
    druzhok.status = 200
    druzhok.reply = "{}"
    assert call("site_delete", site="shop") == {"success": True, "site": "shop", "deleted": True}
    request = druzhok.requests[0]
    assert (request.method, request.path, request.body) == ("DELETE", "/v2/sites/shop", b"")
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert (site / "index.html").exists()


def test_delete_of_an_unpublished_site_says_so(druzhok):
    druzhok.status = 404
    druzhok.reply = json.dumps({"error": "not_found"})
    result = call("site_delete", site="shop")
    assert "No published site" in result["error"]
    assert result["code"] == "not_found"
