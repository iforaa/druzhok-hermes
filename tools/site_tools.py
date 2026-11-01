"""Druzhok: publish a website from the workspace through druzhok, with the tenant key.

Tools and not a skill calling curl, because a terminal command should not need the key.
The limits and refused paths mirror ruoc-gateway's site API, so a bot hears why here and
does not spend one of its 20 publishes an hour on an archive the gateway would refuse.
"""
import logging
import os
import re
import stat
import tarfile
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from agent.secret_scope import get_secret
from tools.registry import no_cache_check_fn, registry, tool_error, tool_result
from tools.terminal_scope import terminal_env

logger = logging.getLogger(__name__)

_SITE_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,49}")
_RESERVED_SITE_NAMES = {"media"}  # Its path on every site belongs to the host's media files.
_MAX_FILES = 100
_MAX_BYTES = 50 * 1024 * 1024
_MAX_DIRS = 100
_MAX_DEPTH = 16
_MAX_PATH_BYTES = 512
_MAX_SEGMENT_BYTES = 200
_MAX_SKIPPED_SHOWN = 10
# Connect to druzhok on this machine; it answers after the gateway has unpacked.
_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=180.0, pool=10.0)

_LIMITS_TEXT = "A site can hold 100 files and 50 MB; an account can hold 5 sites and 200 MB."
_ERROR_TEXT = {
    "bad_archive": "The site service refused the files. Check that index.html is at the top of the folder.",
    "bad_name": "The site service refused this name, or this bot's name cannot be used in a web address.",
    "not_partner_account": "This bot's account cannot publish sites.",
    "slug_reserved": "This bot's name is reserved, so it cannot have a site address.",
    "slug_taken": "Another bot already holds the web address for this bot's name.",
    "site_disabled": "This site was disabled by the operator and cannot be published.",
    "sites_off": "Website hosting is turned off in this bot's settings.",
    "too_large": f"The site is over the limit. {_LIMITS_TEXT}",
    "rate_limited": "Too many publishes right now: one at a time and at most 20 an hour. Try again later.",
    "not_found": "No published site has this name.",
    "invalid_api_key": "The site service did not accept this bot's key.",
    "key_revoked": "The site service did not accept this bot's key.",
    "account_suspended": "This bot's account is suspended.",
    "upload_incomplete": "The upload did not finish. Try again.",
    "internal_error": "The site service had an internal error. Try again later.",
    "sites_unavailable": "The site service is unavailable right now. Try again later.",
    "gateway_unavailable": "The site service did not answer, so the result is unknown. Try again shortly.",
}


class _Refused(Exception):
    """A plain-language reason this site cannot be published as it stands."""


@dataclass
class _Scan:
    files: list[tuple[str, Path]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    dirs: int = 0
    total_bytes: int = 0


@no_cache_check_fn
def _sites_enabled() -> bool:
    return get_secret("DRUZHOK_SITES_ENABLED") == "1"


def _site_dir(site: str) -> Path:
    workspace = terminal_env("TERMINAL_CWD") or os.getcwd()
    return Path(os.path.abspath(os.path.expanduser(workspace))) / "sites" / site


def _raise_on_walk_error(err: OSError) -> None:
    raise err


def _check_name(rel: str) -> None:
    """The names the gateway refuses, which would otherwise refuse the whole archive."""
    try:
        encoded = rel.encode("utf-8")
    except UnicodeEncodeError:
        raise _Refused(f"A file name is not valid text: {rel!r}. Rename it.") from None
    segments = rel.split("/")
    if (
        len(encoded) > _MAX_PATH_BYTES
        or len(segments) > _MAX_DEPTH
        or any(len(s.encode("utf-8")) > _MAX_SEGMENT_BYTES for s in segments)
    ):
        raise _Refused(f"This path is too long or too deep: {rel}. Shorten it.")
    if "\\" in rel or any(unicodedata.category(c) == "Cc" for c in rel):
        raise _Refused(f"This file name has a character a site cannot use: {rel!r}. Rename it.")


def _check_plain_file(st: os.stat_result, rel: str) -> None:
    """A link of either kind could carry a file from outside the folder, such as the bot's .env."""
    if not stat.S_ISREG(st.st_mode):
        raise _Refused(f"{rel} is a symbolic link or a special file, which a site cannot hold. Put a real file there.")
    if st.st_nlink > 1:
        raise _Refused(f"{rel} is a hard link, which a site cannot hold. Put a real copy of the file there.")


def _scan_entry(scan: _Scan, full: Path, rel: str) -> bool:
    """Count one directory or file; True when it is a directory."""
    _check_name(rel)
    st = os.lstat(full)
    if stat.S_ISDIR(st.st_mode):
        scan.dirs += 1
        if scan.dirs > _MAX_DIRS:
            raise _Refused(f"The site has more than {_MAX_DIRS} folders.")
        return True
    _check_plain_file(st, rel)
    scan.total_bytes += st.st_size
    if len(scan.files) >= _MAX_FILES or scan.total_bytes > _MAX_BYTES:
        raise _Refused(f"The site is over the limit. {_LIMITS_TEXT}")
    scan.files.append((rel, full))
    return False


def _scan_site(site_dir: Path) -> _Scan:
    """Walk without following a link. Dotfiles are dropped: one .DS_Store refuses a whole archive."""
    scan = _Scan()
    for root, dirnames, filenames in os.walk(site_dir, onerror=_raise_on_walk_error):
        prefix = Path(root).relative_to(site_dir)
        keep = []
        for name in sorted(dirnames) + sorted(filenames):
            rel = (prefix / name).as_posix()
            if name.startswith("."):
                scan.skipped.append(rel)
            elif _scan_entry(scan, Path(root) / name, rel):
                keep.append(name)
        dirnames[:] = keep
    return scan


def _check_site_dir(site: str, site_dir: Path) -> None:
    for path in (site_dir.parent, site_dir):
        if path.is_symlink():
            raise _Refused(f"{path.name} is a symbolic link. Put the site's files in a real sites/{site}/ folder.")
    if not site_dir.is_dir():
        raise _Refused(f"There is no sites/{site}/ folder in the workspace. Write the site's files there first.")


def _add_file(tar: tarfile.TarFile, rel: str, full: Path, budget: int) -> int:
    """Add one file as a plain regular entry."""
    fd = os.open(full, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        _check_plain_file(st, rel)
        if st.st_size > budget:
            raise _Refused(f"The site is over the limit. {_LIMITS_TEXT}")
        info = tarfile.TarInfo(rel)
        info.size = st.st_size
        info.mode = 0o644
        info.mtime = int(st.st_mtime)
        tar.addfile(info, f)
    return st.st_size


def _write_archive(out, scan: _Scan) -> None:
    """A gzip tar with the site's files at its root. Directories are implied by the file paths."""
    budget = _MAX_BYTES
    with tarfile.open(fileobj=out, mode="w:gz", format=tarfile.PAX_FORMAT, compresslevel=6) as tar:
        for rel, full in scan.files:
            budget -= _add_file(tar, rel, full, budget)


def _endpoint(site: str) -> tuple[str, str]:
    base = (get_secret("FIRECRAWL_API_URL") or "").strip().rstrip("/")
    key = (get_secret("FIRECRAWL_API_KEY") or "").strip()
    if not base or not key:
        raise _Refused("Site publishing is not set up on this bot.")
    return f"{base}/v2/sites/{site}", key


def _answer(response: httpx.Response) -> dict:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _service_error(action: str, response: httpx.Response, body: dict) -> str:
    code = body.get("error") if isinstance(body.get("error"), str) else None
    logger.warning("%s answered HTTP %s, code %s", action, response.status_code, code)
    text = _ERROR_TEXT.get(code) or f"{action} failed: the site service answered HTTP {response.status_code}."
    return tool_error(text, code=code) if code else tool_error(text)


def _bad_name(site: str) -> str | None:
    if isinstance(site, str) and _SITE_NAME.fullmatch(site):
        return None
    return tool_error(
        "A site name uses lowercase Latin letters, digits and hyphens, starts with a letter or digit, "
        "and has at most 50 characters."
    )


def _send_publish(url: str, key: str, scan: _Scan) -> str:
    with tempfile.TemporaryFile() as archive:
        _write_archive(archive, scan)
        archive.seek(0)
        response = httpx.put(
            url,
            content=archive,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/gzip"},
            timeout=_TIMEOUT,
            trust_env=False,
        )
    body = _answer(response)
    if response.status_code not in (200, 201) or not isinstance(body.get("url"), str):
        return _service_error("Publishing", response, body)
    result = {k: body.get(k) for k in ("url", "files", "bytes", "published_at")}
    if scan.skipped:
        result["skipped_dotfiles"] = scan.skipped[:_MAX_SKIPPED_SHOWN]
        result["skipped_count"] = len(scan.skipped)
    return tool_result(success=True, **result)


def site_publish(site: str) -> str:
    """Pack ``<workspace>/sites/<site>/`` and publish it; the answer's ``url`` is the only valid link."""
    if (bad := _bad_name(site)) is not None:
        return bad
    if site in _RESERVED_SITE_NAMES:
        return tool_error(f"The name {site} is reserved for the site service. Rename the folder and publish that.")
    try:
        url, key = _endpoint(site)
        site_dir = _site_dir(site)
        _check_site_dir(site, site_dir)
        scan = _scan_site(site_dir)
        if not any(rel == "index.html" for rel, _ in scan.files):
            raise _Refused(f"sites/{site}/ has no index.html at its top level, and a site needs one.")
        return _send_publish(url, key, scan)
    except _Refused as refused:
        return tool_error(str(refused))
    except httpx.HTTPError as exc:
        logger.warning("site_publish could not reach the site service: %s", type(exc).__name__)
        return tool_error("The site service could not be reached, so the result is unknown. Try again shortly.")
    except OSError as exc:
        logger.warning("site_publish could not read the site's files: %s", type(exc).__name__)
        return tool_error(f"The site's files could not be read ({type(exc).__name__}). Check sites/{site}/ and try again.")


def site_delete(site: str) -> str:
    """Take a site offline. Its files stay in the workspace."""
    if (bad := _bad_name(site)) is not None:
        return bad
    try:
        url, key = _endpoint(site)
        response = httpx.delete(url, headers={"Authorization": f"Bearer {key}"}, timeout=_TIMEOUT, trust_env=False)
    except _Refused as refused:
        return tool_error(str(refused))
    except httpx.HTTPError as exc:
        logger.warning("site_delete could not reach the site service: %s", type(exc).__name__)
        return tool_error("The site service could not be reached, so the result is unknown. Try again shortly.")
    if response.status_code != 200:
        return _service_error("Deleting", response, _answer(response))
    return tool_result(success=True, site=site, deleted=True)


_SITE_PARAM = {
    "type": "string",
    "description": "The site's folder name under sites/: lowercase Latin letters, digits and hyphens, at most 50 characters, and not media.",
}

registry.register(
    name="site_publish",
    toolset="sites",
    schema={
        "name": "site_publish",
        "description": (
            "Publish the website in the workspace folder sites/<site>/, replacing any earlier version. The folder "
            "needs index.html at its top level. Dotfiles are left out and symbolic links are refused. "
            "Returns the public url: give the user exactly that url and never build one yourself."
        ),
        "parameters": {"type": "object", "properties": {"site": _SITE_PARAM}, "required": ["site"]},
    },
    handler=lambda args, **_kw: site_publish(args.get("site", "")),
    check_fn=_sites_enabled,
    emoji="🌍",
)

registry.register(
    name="site_delete",
    toolset="sites",
    schema={
        "name": "site_delete",
        "description": "Take a published website offline. Its files stay in the workspace folder sites/<site>/.",
        "parameters": {"type": "object", "properties": {"site": _SITE_PARAM}, "required": ["site"]},
    },
    handler=lambda args, **_kw: site_delete(args.get("site", "")),
    check_fn=_sites_enabled,
    emoji="🌍",
)
