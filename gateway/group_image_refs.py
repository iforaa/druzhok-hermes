"""Run-local authority to edit inbound group photos; labels never authorize a file read."""

from __future__ import annotations

import base64
import secrets
import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from gateway.group_limits import REFERENCE_REFUSAL

MAX_PHOTOS = 4
MAX_PHOTO_BYTES = 20 * 1024 * 1024
MAX_PHOTO_PIXELS = 16_000_000
# Plug.Parsers' default multipart limit is 8,000,000 bytes. This stricter inline
# budget also fits managed FAL's JSON limit, leaving room for prompt and framing.
INLINE_BUDGET = 3_500_000
TOO_LARGE = "фото слишком большое, пришли одно фото поменьше"
TOO_MANY = "слишком много фото для этой модели, выбери меньше фото"
_current: ContextVar[Any] = ContextVar("group_image_references", default=None)


@dataclass
class PhotoScope:
    chat_id: str
    photos: dict[str, tuple[bytes, str]] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    revoked: bool = False
    still_current: Callable[[], bool] = lambda: True
    max_sources: int = 0
    lock: Any = field(default_factory=threading.RLock, repr=False)

    def revoke(self) -> None:
        with self.lock:
            self.revoked = True
            self.photos.clear()

    def hint(self, path: str) -> str:
        label = self.paths.get(str(path))
        if self.revoked or not label:
            return ""
        hint = f'[Фото {label.rsplit(":", 1)[1]}: чтобы перерисовать, передай image_url="{label}"]'
        if self.max_sources:
            hint += f"\n[В одном вызове image_generate можно выбрать до {self.max_sources} фото суммарно в image_url и reference_image_urls.]"
        return hint


def edit_capabilities() -> dict:
    from tools.image_generation_tool import _active_image_capabilities
    from tools.registry import _check_fn_cached, registry

    entry = registry.get_entry("image_generate")
    if entry is None or (entry.check_fn is not None and not _check_fn_cached(entry.check_fn)):
        return {}
    caps = _active_image_capabilities(for_edit=True)
    if "image" not in caps.get("modalities", []) or caps.get("source_image_role") == "style":
        return {}
    return caps if int(caps.get("max_source_images") or caps.get("max_reference_images") or 0) > 0 else {}


def _snapshot_photo(path: str, input_limit: int) -> tuple[bytes, str] | None:
    import io
    from PIL import Image
    from agent.image_routing import _sniff_mime_from_bytes, _transcode_to_png
    from tools.image_generation_tool import _shrink_data_url

    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(input_limit + 1)
        if not raw or len(raw) > input_limit:
            return None
        mime = _sniff_mime_from_bytes(raw)
        if mime not in {"image/jpeg", "image/png"}:
            raw = _transcode_to_png(raw, max_pixels=MAX_PHOTO_PIXELS)
            mime = "image/png"
        if raw is None:
            return None
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > MAX_PHOTO_PIXELS:
                return None
            animated = getattr(image, "is_animated", False)
            image.verify()
        if animated:
            raw = _transcode_to_png(raw, max_pixels=MAX_PHOTO_PIXELS)
            if raw is None:
                return None
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
        if len(raw) > MAX_PHOTO_BYTES:
            url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
            url = _shrink_data_url(url, MAX_PHOTO_BYTES)
            mime = url.partition(";")[0].removeprefix("data:")
            raw = base64.b64decode(url.partition(",")[2])
        return (raw, mime) if len(raw) <= MAX_PHOTO_BYTES else None
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None


def snapshot_photos(source: Any, paths: list[str]) -> PhotoScope:
    from gateway.platforms.base import DEFAULT_INBOUND_MEDIA_MAX_BYTES, get_inbound_media_max_bytes

    scope = PhotoScope(str(source.chat_id))
    nonce = secrets.token_hex(16)
    input_limit = get_inbound_media_max_bytes()
    if input_limit <= 0:
        input_limit = DEFAULT_INBOUND_MEDIA_MAX_BYTES
    for number, path in enumerate(paths[:MAX_PHOTOS], 1):
        if (photo := _snapshot_photo(path, input_limit)) is None:
            continue
        label = f"ref:{nonce}:{number}"
        scope.photos[label] = photo
        scope.paths[str(path)] = label
    return scope


def prepare_event_photos(runner: Any, event: Any, paths: list[str]) -> PhotoScope | None:
    from gateway.run import _load_gateway_config, _platform_config_key

    if event.internal or event._native_media_origin != ("telegram", str(event.source.chat_id)):
        return None
    enabled, disabled = runner._resolve_turn_toolsets(
        _load_gateway_config(), event.source, _platform_config_key(event.source.platform))
    from model_tools import _select_tool_names
    if "image_generate" not in _select_tool_names(enabled, disabled, quiet_mode=True):
        return None
    caps = edit_capabilities()
    if not caps:
        return None
    scope = snapshot_photos(event.source, paths)
    scope.max_sources = int(caps.get("max_source_images") or caps["max_reference_images"])
    return scope


@contextmanager
def photo_scope(scope: PhotoScope | None):
    """Copies of this context share the revocable object, never a mutable env fallback."""
    token = _current.set(scope)
    try:
        yield
    finally:
        _current.reset(token)


def photo_hint(path: str) -> str:
    scope = _current.get()
    return scope.hint(path) if scope is not None else ""


def take_event_scope(source: Any) -> PhotoScope | None:
    from gateway.session_context import bound_turn_event

    event = bound_turn_event()
    scope = getattr(event, "_group_photo_scope", None)
    if scope is not None:
        del event._group_photo_scope
        if scope.chat_id == str(source.chat_id):
            return scope
        scope.revoke()
    return None


def resolve_references(image_url: Any, references: Any, prompt: str) -> tuple[Any, Any, Any]:
    """Validate the entire model-supplied selection before producing any data URLs."""
    from gateway.session_context import get_session_env
    from tools.image_generation_tool import _shrink_data_url

    scope = _current.get()
    if scope is None:
        raise ValueError(REFERENCE_REFUSAL)
    if image_url is not None and not isinstance(image_url, str):
        raise ValueError(REFERENCE_REFUSAL)
    if references is not None and not isinstance(references, list):
        raise ValueError(REFERENCE_REFUSAL)
    selected = ([image_url] if image_url is not None else []) + (references or [])
    with scope.lock:
        if scope.revoked or not scope.still_current():
            scope.revoke()
            raise ValueError(REFERENCE_REFUSAL)
        if scope.chat_id != get_session_env("HERMES_SESSION_CHAT_ID", ""):
            raise ValueError(REFERENCE_REFUSAL)
        if not selected or any(not isinstance(label, str) or label not in scope.photos for label in selected):
            raise ValueError(REFERENCE_REFUSAL)
        labels = list(dict.fromkeys(selected))
        photos = [scope.photos[label] for label in labels]
    caps = edit_capabilities()
    if not caps:
        raise ValueError(REFERENCE_REFUSAL)
    if len(labels) > int(caps.get("max_source_images") or caps["max_reference_images"]):
        raise ValueError(TOO_MANY)
    budget = INLINE_BUDGET - len(json.dumps(prompt, ensure_ascii=True).encode("utf-8")) - 64_000
    if budget <= 0:
        raise ValueError(TOO_LARGE)
    urls = [f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
            for raw, mime in photos]
    if sum(map(len, urls)) > budget:
        share = budget // len(urls)
        urls = [_shrink_data_url(url, share) if len(url) > share else url for url in urls]
    if sum(map(len, urls)) > budget:
        raise ValueError(TOO_LARGE)
    return urls[0], urls[1:] or None, caps.get("edit_route")


@contextmanager
def run_photo_scope(runner: Any, turn: Any):
    """Bind before copying the worker context, revoke before draining a follow-up."""
    scope = take_event_scope(turn.source)
    if scope is not None and not turn._run_still_current():
        scope.revoke()
        scope = None
    if scope is not None:
        scope.still_current = turn._run_still_current
        runner._session_state(turn.session_key).turn.group_photo_scope = scope
    try:
        with photo_scope(scope):
            yield
    finally:
        if scope is not None:
            scope.revoke()
            state = runner._peek_session_state(turn.session_key)
            if state is not None and state.turn.group_photo_scope is scope:
                state.turn.group_photo_scope = None
