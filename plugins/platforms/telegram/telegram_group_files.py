"""Druzhok: the Telegram adapter's last gate for a local file, however a delivery path got here.

Telegram DM ids are positive integers, so any other chat is a group or channel, and it gets no local
file except a picture ``drop_group_files`` approved for that chat (``gateway.group_limits``). The
approval is used up by the first public send that passes the gate. That send's own retries and
fallbacks (inline flood retry, deferred re-upload, photo -> document, album -> single photo) run inside
its scope and pass. Every other send opens a fresh scope and is refused.
"""

import functools
import logging
import os
from contextvars import ContextVar
from typing import Any, Optional
from urllib.parse import unquote

from gateway.group_limits import consume_group_file_approval, group_file_approved, group_media
from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.telegram.adapter")

# The running public media send: ``{"chat": <id>}`` once an approved picture passed the gate. A task
# spawned inside the send (the deferred flood re-upload) copies it, so it belongs to the same send.
_SEND_SCOPE: ContextVar[Optional[dict]] = ContextVar("telegram_group_send_scope", default=None)


def group_send(fn):
    """Mark a public media sender: the outermost call opens the scope its nested calls share."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        if _SEND_SCOPE.get() is not None:
            return await fn(*args, **kwargs)
        token = _SEND_SCOPE.set({})
        try:
            return await fn(*args, **kwargs)
        finally:
            _SEND_SCOPE.reset(token)
    return wrapper


def _is_dm(chat_id: Any) -> bool:
    return str(chat_id).strip().isdigit()


def _cleared(chat_id: Any) -> bool:
    scope = _SEND_SCOPE.get()
    return scope is not None and scope.get("chat") == str(chat_id).strip()


def _use_approval(chat_id: Any, path: str) -> bool:
    """Use up an approval for this send, opening its scope to the send's own retries."""
    if not group_file_approved(chat_id, path):
        return False
    consume_group_file_approval(chat_id, path)
    scope = _SEND_SCOPE.get()
    if scope is not None:
        scope["chat"] = str(chat_id).strip()
    return True


def group_file_refused(adapter: Any, chat_id: Any, path: str) -> Optional[SendResult]:
    if _is_dm(chat_id) or _cleared(chat_id) or _use_approval(chat_id, path):
        return None
    logger.warning("[%s] Refused to send a local file to group chat %s: %s", adapter.name, chat_id, os.path.basename(path))
    return SendResult(success=False, error="Local files are not delivered to group chats")


def group_images(adapter: Any, chat_id: Any, images: list) -> list:
    """``(url, alt)`` images a chat may get: every one in a DM; in a group, URLs and approved pictures."""
    if _is_dm(chat_id) or _cleared(chat_id):
        return images
    kept = [img for img in images if not img[0].startswith("file://") or _use_approval(chat_id, unquote(img[0][7:]))]
    if len(kept) != len(images):
        logger.warning("[%s] Refused %d local image(s) to group chat %s", adapter.name, len(images) - len(kept), chat_id)
    return kept


def adapter_group_media(adapter: Any) -> frozenset[str]:
    """The inbound media kinds a group turn may see; ``group_media`` fails closed on a bad value."""
    return group_media((getattr(adapter.config, "extra", None) or {}).get("group_media"))
