"""Reply quote policy for Telegram end-user chats."""

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from plugins.platforms.telegram.adapter import TelegramAdapter


def suppress_private_reply_quote(
    adapter: "TelegramAdapter", chat_id: Optional[str], metadata: Optional[dict[str, Any]],
) -> bool:
    """Ordinary private chats omit quotes; topic lanes retain configured anchors."""
    if not chat_id or not str(chat_id).isdecimal() or int(chat_id) <= 0:
        return False
    if (adapter._metadata_thread_id(metadata) is not None
            or adapter._metadata_direct_messages_topic_id(metadata) is not None
            or adapter._dm_topic_fallback(metadata)):
        return False
    from gateway.run_config_loaders import GatewayConfigLoadersMixin

    return GatewayConfigLoadersMixin._load_end_user_mode()
