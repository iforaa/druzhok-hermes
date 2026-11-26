"""OpenAI GPT Image 2 and 2.5 Flare/Sunburst quality tiers;
base64 output → image cache. Selection: ``OPENAI_IMAGE_MODEL`` → ``image_gen.openai.model`` →
``image_gen.model`` → :data:`DEFAULT_MODEL`; an id outside the catalog is sent verbatim.
Endpoint: ``image_gen.openai.base_url`` → the named endpoint ``image_gen.openai.provider`` →
``OPENAI_BASE_URL`` → SDK default; key: env named by ``image_gen.openai.key_env`` → the named
endpoint's credential → ``OPENAI_API_KEY``."""

from __future__ import annotations

import io
import hashlib
import threading
from time import monotonic
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.secret_scope import get_secret, get_secret_str
from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, resolve_aspect_ratio, success_response
from plugins.image_gen._common import (
    GPT_IMAGE_2_API_MODEL as API_MODEL, GPT_IMAGE_2_DEFAULT as DEFAULT_MODEL, GPT_IMAGE_2_TIERS,
    StaticImageGenProvider, collect_source_images, error_factory, import_openai, load_image_gen_config,
    materialize_image, openai_importable, prompt_required_error, record_token_usage, resolve_static_model,
    size_for)

logger = logging.getLogger(__name__)

# Keep subscription routing independent: Codex does not verify explicit image model selection.
MODELS = {
    **{key: {**meta, "api_model": API_MODEL} for key, meta in GPT_IMAGE_2_TIERS.items()},
    **{
        model if quality == "auto" else f"{model}-{quality}": {
            "display": f"GPT Image 2.5 {name} ({quality.title()})",
            "speed": speed,
            "strengths": strengths,
            "api_model": model,
            "quality": quality,
        }
        for model, name, speed, strengths in (
            ("gpt-image-2.5-flare", "Flare", "Fast", "Everyday image generation and editing"),
            ("gpt-image-2.5-sunburst", "Sunburst", "Slower", "Precision generation and editing"),
        )
        for quality in ("auto", "low", "medium", "high", "xhigh", "max")
    },
}


def _resolve_model() -> tuple[str, dict[str, Any]]:
    return resolve_static_model(
        MODELS, DEFAULT_MODEL, env_var="OPENAI_IMAGE_MODEL", config_key="openai", passthrough=True)


def _named_endpoint(name: str) -> tuple[str, str]:
    """``(base_url, api_key)`` of the user-declared custom endpoint *name* (``providers:`` /
    ``custom_providers:``), so image generation reuses a chat endpoint's URL and credential without
    duplicating the key into OpenAI variables (#83080). Unknown name → ``("", "")`` with a warning."""
    from hermes_cli.runtime_provider import _get_named_custom_provider

    entry = _get_named_custom_provider(name)
    if not entry:
        logger.warning("image_gen.openai.provider %r matches no custom endpoint in providers:", name)
        return "", ""
    key_env = str(entry.get("key_env") or "").strip()
    api_key = str(entry.get("api_key") or "").strip() or (get_secret(key_env) if key_env else None) or ""
    return str(entry.get("base_url") or "").strip().rstrip("/"), api_key


def _resolve_endpoint() -> tuple[str, str]:
    """``(base_url, api_key)`` — ``image_gen.openai.base_url`` → ``OPENAI_BASE_URL`` → ``""`` (SDK default);
    the env var named by ``image_gen.openai.key_env`` → ``OPENAI_API_KEY``. Only the var NAME lives in
    config.yaml; ``is_available()`` and ``generate()`` share this so they cannot disagree (#65309)."""
    cfg = load_image_gen_config("openai")
    named = str(cfg.get("provider") or "").strip()
    named_base, named_key = _named_endpoint(named) if named else ("", "")
    base_url = (str(cfg.get("base_url") or "").strip().rstrip("/") or named_base
                or get_secret_str("OPENAI_BASE_URL").strip())
    key_env = str(cfg.get("key_env") or "").strip()
    api_key = (get_secret(key_env) if key_env else None) or named_key or get_secret("OPENAI_API_KEY") or ""
    return base_url, api_key


def _build_client(openai: Any, base_url: str, api_key: str) -> Any:
    """``openai.OpenAI`` on Hermes' env-only-proxy httpx client, so a local/custom endpoint never
    routes through a macOS system proxy whose ExceptionsList httpx cannot see (#64888). The project
    header is blanked: an ``OPENAI_PROJECT_ID`` set for chat makes ``/images/generations`` 403 on
    projects with a model allow-list, and the key already carries the project (#60748)."""
    from agent.process_bootstrap import build_keepalive_http_client

    kwargs: dict[str, Any] = {"api_key": api_key, "default_headers": {"OpenAI-Project": ""}}
    # Passing the scoped default prevents the SDK inheriting a different profile's URL.
    kwargs["base_url"] = base_url or "https://api.openai.com/v1"
    http_client = build_keepalive_http_client(base_url)
    if http_client is not None:
        kwargs["http_client"] = http_client
    return openai.OpenAI(**kwargs)


def _load_image_bytes(ref: str) -> tuple[bytes, str]:
    """Load ``(data, filename)`` from a URL, data URI or local path; raises on IO/network error."""
    ref = ref.strip()
    lower = ref.lower()
    if lower.startswith(("http://", "https://")):
        from tools.url_safety import create_ssrf_safe_client, is_safe_url

        if not is_safe_url(ref):
            raise ValueError(f"Image reference URL failed the SSRF safety check: {ref}")
        with create_ssrf_safe_client(timeout=60, follow_redirects=True) as client:
            resp = client.get(ref)
        resp.raise_for_status()
        name = ref.split("?", 1)[0].rsplit("/", 1)[-1] or "image.png"
        return resp.content, name
    if lower.startswith("data:"):
        import base64

        header, _, b64 = ref.partition(",")
        ext = (header.split("image/", 1)[1].split(";", 1)[0] if "image/" in header else "") or "png"
        return base64.b64decode(b64), f"image.{ext}"
    from agent.file_safety import raise_if_read_blocked  # credential-read guard before local bytes

    raise_if_read_blocked(ref)
    with open(ref, "rb") as fh:
        data = fh.read()
    return data, os.path.basename(ref) or "image.png"


def _named_bytes_io(ref: str) -> io.BytesIO:
    """``images.edit()`` expects named file-like objects for correct multipart."""
    data, fname = _load_image_bytes(ref)
    bio = io.BytesIO(data)
    bio.name = fname
    return bio


_EDIT_CAPABILITIES_TTL = 600.0
_EDIT_CAPABILITIES_LOCK = threading.Lock()
_EDIT_CAPABILITIES_KEYS: dict[tuple, Any] = {}
_EDIT_CAPABILITIES_CACHE: dict[tuple[str, bytes, str], tuple[float, dict[str, Any]]] = {}


def _cached_edit_capabilities(base_url: str, api_key: str, model: str) -> dict[str, Any]:
    # Catalogue errors belong to the credential too: a sibling's 401 must not disable this bot.
    key = (base_url, hashlib.sha256(api_key.encode()).digest(), model)
    with _EDIT_CAPABILITIES_LOCK:
        lock = _EDIT_CAPABILITIES_KEYS.setdefault(key, threading.Lock())
    with lock:
        cached = _EDIT_CAPABILITIES_CACHE.get(key)
        if cached is not None and monotonic() < cached[0]:
            return dict(cached[1])
        result = _probe_edit_capabilities(base_url, api_key, model)
        _EDIT_CAPABILITIES_CACHE[key] = (monotonic() + _EDIT_CAPABILITIES_TTL, result)
        return dict(result)


def _probe_edit_capabilities(base_url: str, api_key: str, model: str) -> dict[str, Any]:
    openai, error = import_openai("openai", DEFAULT_ASPECT_RATIO)
    if error:
        return {}
    try:
        with _build_client(openai, base_url, api_key) as client:
            catalogue = client.with_options(timeout=5, max_retries=0).get("/images/models", cast_to=dict[str, Any])
        editors = [row for row in catalogue.get("models", [])
                   if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"].strip()
                   and row.get("edits") is True]
        if not editors:
            return {}
        chosen = next((row for row in editors if row["id"] == model), editors[0])
        # A proxy may choose its bot's own editor. Without effective-route metadata,
        # use the minimum supported count across its catalogue, one when undeclared.
        limit = min(16, *(int(row.get("max_reference_images", 1)) for row in editors))
        if limit < 1:
            return {}
    except Exception:
        logger.debug("Could not verify the gateway image edit catalogue", exc_info=True)
        return {}
    return {"modalities": ["text", "image"], "max_reference_images": limit, "edit_model": chosen["id"]}


class OpenAIImageGenProvider(StaticImageGenProvider):
    """OpenAI ``images.generate`` / ``images.edit`` backend with selectable API models."""

    provider_id = "openai"
    label = "OpenAI"
    models = MODELS
    default_model_id = DEFAULT_MODEL
    price = "varies"
    setup = dict(
        name="OpenAI", badge="paid",
        tag="GPT Image 2 / 2.5 Flare / 2.5 Sunburst — text-to-image & image editing",
        key="OPENAI_API_KEY", prompt="OpenAI API key", url="https://platform.openai.com/api-keys")

    def is_available(self) -> bool:
        return bool(_resolve_endpoint()[1]) and openai_importable()

    def capabilities(self) -> dict[str, Any]:
        # images.edit() accepts up to 16 source images.
        return {"modalities": ["text", "image"], "max_reference_images": 16}

    def edit_capabilities(self, model: Optional[str] = None) -> dict[str, Any]:
        from urllib.parse import urlparse

        base_url, api_key = _resolve_endpoint()
        if not api_key:
            return {}
        _, meta = _resolve_model()
        if not base_url or urlparse(base_url).hostname == "api.openai.com":
            caps = self.capabilities()
        else:
            caps = _cached_edit_capabilities(base_url, api_key, meta["api_model"])
        if caps:
            caps["edit_route"] = {
                "base_url": base_url.rstrip("/") or "https://api.openai.com/v1",
                "credential": hashlib.sha256(api_key.encode()).hexdigest(),
                "model": caps.get("edit_model") or meta["api_model"],
                "quality": meta["quality"] if caps.get("edit_model", meta["api_model"]) == meta["api_model"] else None,
            }
        return caps

    def generate(
        self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return prompt_required_error("openai", aspect)
        base_url, api_key = _resolve_endpoint()
        if not api_key:
            return error_factory("openai", aspect)(
                "OPENAI_API_KEY not set (or the variable named by image_gen.openai.key_env is empty). "
                "Run `hermes tools` → Image Generation → OpenAI to configure, or `hermes setup` "
                "to add the key.",
                "auth_required")

        openai, err = import_openai("openai", aspect)
        if err:
            return err
        tier_id, meta = _resolve_model()
        size = size_for(aspect)
        sources = collect_source_images(image_url, reference_image_urls, limit=16)
        is_edit = bool(sources)
        route = kwargs.get("verified_edit_route") if is_edit else None
        if route and (route["base_url"] != (base_url.rstrip("/") or "https://api.openai.com/v1")
                      or route["credential"] != hashlib.sha256(api_key.encode()).hexdigest()):
            return error_factory("openai", aspect)("Image editing route changed; retry the request.", "route_changed")
        edit_model = route["model"] if route else None
        reported_model = edit_model if edit_model and edit_model != meta["api_model"] else tier_id
        dispatched_model = edit_model or meta["api_model"]
        fail = error_factory("openai", aspect, model=reported_model, prompt=prompt)
        client = _build_client(openai, base_url, api_key)

        # gpt-image-2 returns b64_json unconditionally and REJECTS
        # ``response_format`` as an unknown parameter. Don't send it.
        # A custom (non-catalog) model id carries no quality tier: gateways reject unknown enum values.
        request: dict[str, Any] = dict(model=dispatched_model, prompt=prompt, size=size, n=1)
        quality = route["quality"] if route else meta["quality"]
        if quality is not None:
            request["quality"] = quality
        if is_edit:
            try:
                files = [_named_bytes_io(ref) for ref in sources]
            except Exception as exc:
                return fail(f"Could not load source image for editing: {exc}", "io_error")
            request["image"] = files if len(files) > 1 else files[0]
        verb, call = ("edit", client.images.edit) if is_edit else ("generation", client.images.generate)
        try:
            response = call(**request)
        except Exception as exc:
            logger.debug("OpenAI image %s failed", verb, exc_info=True)
            return fail(f"OpenAI image {'editing' if is_edit else 'generation'} failed: {exc}", "api_error")

        # gpt-image bills per text/image token; the tier id is a Hermes label, the API model prices.
        # Recorded before extraction/save: the tokens are billed whether or not an image came back.
        custom_edit = bool(route and route["base_url"] != "https://api.openai.com/v1")
        actual_model = getattr(response, "model", None) if custom_edit else None
        actual_model = actual_model.strip() if isinstance(actual_model, str) else ""
        usage_model = actual_model or ("unknown" if custom_edit else dispatched_model)
        if actual_model or custom_edit:
            reported_model = usage_model
        record_token_usage(getattr(response, "usage", None), model=usage_model, provider="openai")
        data = getattr(response, "data", None) or []
        if not data:
            return fail("OpenAI returned no image data", "empty_response")
        first = data[0]
        image_ref, err = materialize_image(
            getattr(first, "b64_json", None), getattr(first, "url", None),
            prefix=f"openai_{tier_id}", label="OpenAI", provider="openai",
            model=reported_model, prompt=prompt, aspect=aspect, log=logger)
        if err:
            return err
        extra: dict[str, Any] = {"size": size, "quality": request.get("quality")}
        if custom_edit:
            extra["requested_model"] = dispatched_model
        if getattr(first, "revised_prompt", None):
            extra["revised_prompt"] = first.revised_prompt
        return success_response(
            image=image_ref, model=reported_model, prompt=prompt, aspect_ratio=aspect, provider="openai",
            modality="image" if is_edit else "text", extra=extra)


def register(ctx) -> None:
    """Plugin entry point — wire ``OpenAIImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(OpenAIImageGenProvider())
