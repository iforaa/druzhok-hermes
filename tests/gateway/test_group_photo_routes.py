"""Effective editing route, scoped catalogue caching and endpoint isolation."""

import asyncio
import base64
import contextvars
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from gateway import group_image_refs as refs
from gateway.group_limits import REFERENCE_REFUSAL
from tests.gateway.test_group_images import _group, _event, _inbound_runner, _prepare
from tests.gateway.test_group_photo_refs import bound, png, prepare

from tools.image_generation_tool import _managed_model_plugin as _real_managed_model_plugin

pytest_plugins = ("tests.gateway.test_group_photo_refs",)


def photos(tmp_path, count):
    paths = [tmp_path / f"{i}.png" for i in range(count)]
    for i, path in enumerate(paths):
        path.write_bytes(png((i*50, 100, 100)))
    return paths


@pytest.mark.parametrize("status", [200, 403])
def test_catalogue_caches_success_and_failure_with_scope_and_expiry(tmp_path, backend, monkeypatch, status):
    import plugins.image_gen.openai as provider
    clock = [100.0]
    monkeypatch.setattr(provider, "monotonic", lambda: clock[0])
    monkeypatch.setattr(provider, "_resolve_endpoint", lambda: ("http://localhost/v1", "key-a"))
    backend.catalogue_status = status
    instance = provider.OpenAIImageGenProvider()
    first = instance.edit_capabilities()
    assert bool(first) == (status == 200)
    clock[0] += 599
    assert instance.edit_capabilities() == first and len(backend.probes) == 1
    clock[0] += 1
    assert instance.edit_capabilities() == first and len(backend.probes) == 2
    monkeypatch.setattr(provider, "_resolve_endpoint", lambda: ("http://localhost/v1", "key-b"))
    instance.edit_capabilities()
    assert len(backend.probes) == 3
    monkeypatch.setattr(provider, "_resolve_endpoint", lambda: ("http://other.local/v1", "key-b"))
    instance.edit_capabilities()
    assert backend.probes == [
        ("http://localhost/v1/images/models", "Bearer key-a"),
        ("http://localhost/v1/images/models", "Bearer key-a"),
        ("http://localhost/v1/images/models", "Bearer key-b"),
        ("http://other.local/v1/images/models", "Bearer key-b"),
    ]


def test_group_multipart_uses_verified_api_model_and_safest_catalogue_limit(tmp_path, backend, monkeypatch):
    monkeypatch.setattr("plugins.image_gen.openai._resolve_endpoint", lambda: ("http://localhost/v1", "test"))
    backend.catalogue = {"models": [{"id": "editor", "edits": True, "max_reference_images": 4},
                                    {"id": "other-editor", "edits": True, "max_reference_images": 1}]}
    _, event, _ = prepare(tmp_path, "native", photos(tmp_path, 2))
    scope = event._group_photo_scope
    labels = list(scope.photos)
    result = bound(scope, {"image_url": labels[0]})
    assert result["success"]
    assert result["model"] == "editor" and result.get("quality") is None
    assert backend.models[-1] == "editor" and backend.usage_models[-1] == "editor"
    assert "quality" not in backend.fields[-1]
    assert refs.TOO_MANY in bound(scope, {"reference_image_urls": labels})["error"]
    assert len(backend) == 1 and len(backend.probes) == 1


def test_openai_client_pins_scoped_default_even_when_sdk_env_points_elsewhere(monkeypatch):
    import plugins.image_gen.openai as provider
    import openai
    monkeypatch.setenv("OPENAI_BASE_URL", "https://launch-profile.invalid/v1")
    monkeypatch.setattr("agent.process_bootstrap.build_keepalive_http_client", lambda base_url: None)
    with provider._build_client(openai, "", "secondary-key") as client:
        assert str(client.base_url) == "https://api.openai.com/v1/"
    with provider._build_client(openai, "https://secondary.invalid/v1", "secondary-key") as client:
        assert str(client.base_url) == "https://secondary.invalid/v1/"


@pytest.mark.parametrize("override", [1, True, "image_gen", {"image_gen": True}, []])
def test_malformed_group_toolsets_follow_resolver_fallback(tmp_path, backend, override):
    _, allowed, _ = prepare(tmp_path, "native", extra={"group_media": ["image"], "group_toolsets": [" image_gen "]})
    label = next(iter(allowed._group_photo_scope.photos))
    assert bound(allowed._group_photo_scope, {"image_url": label})["success"]
    _, event, text = prepare(tmp_path, "native", extra={"group_media": ["image"], "group_toolsets": override})
    assert event._group_photo_scope is None and "ref:" not in text
    assert REFERENCE_REFUSAL in bound(None, {"image_url": label})["error"]


def test_disabled_image_tool_suppresses_labels(tmp_path, backend, monkeypatch):
    _, allowed, _ = prepare(tmp_path, "native")
    label = next(iter(allowed._group_photo_scope.photos))
    assert bound(allowed._group_photo_scope, {"image_url": label})["success"]
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"agent": {"disabled_toolsets": ["image_gen"]}})
    _, event, _ = prepare(tmp_path, "native")
    assert event._group_photo_scope is None
    assert REFERENCE_REFUSAL in bound(None, {"image_url": label})["error"]


def test_preprocessing_has_hints_but_no_reference_authority(tmp_path, backend):
    from tools.vision_tools import vision_analyze_tool
    path = photos(tmp_path, 1)[0]
    runner, _ = _inbound_runner({"group_media": ["image"], "group_toolsets": ["image_gen"]}, "text")
    event = _event(_group(), "redraw", media_urls=[str(path)], media_types=["image/png"])
    event._native_media_origin = ("telegram", str(event.source.chat_id))
    async def analyze(**kw):
        assert refs._current.get() is None
        return '{"success": true, "analysis": "red"}'
    with patch("tools.vision_tools.vision_analyze_tool", analyze):
        text = _prepare(runner, event)
    label = next(iter(event._group_photo_scope.photos))
    assert label in text
    assert bound(event._group_photo_scope, {"image_url": label})["success"]
    assert REFERENCE_REFUSAL in bound(None, {"image_url": label})["error"]


def test_corrupt_photo_is_skipped_without_renumbering(tmp_path, backend):
    paths = photos(tmp_path, 2)
    paths[0].write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00"*20)
    _, event, _ = prepare(tmp_path, "native", paths)
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    assert label.endswith(":2")
    assert bound(scope, {"image_url": label})["success"]
    assert REFERENCE_REFUSAL in bound(scope, {"image_url": label.rsplit(":", 1)[0] + ":1"})["error"]


def test_large_admitted_png_normalizes_below_retention_and_request_budgets(tmp_path, backend, monkeypatch):
    path = tmp_path / "large.png"
    # A large ancillary chunk leaves a tiny valid image: verify real ingress and retention budgets
    # without allocating a large decoded raster on this shared host.
    import struct
    import zlib
    data = png()
    payload = b"note\x00" + b"x"*(22*1024*1024)
    kind = b"tEXt"
    chunk = struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    path.write_bytes(data[:-12] + chunk + data[-12:])
    _, event, _ = prepare(tmp_path, "native", [path])
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    assert len(scope.photos[label][0]) < refs.MAX_PHOTO_BYTES
    assert bound(scope, {"image_url": label})["success"]
    assert backend[-1][2] < 8_000_000
    assert REFERENCE_REFUSAL in bound(scope, {"image_url": str(path)})["error"]


def _post_recorder(module, monkeypatch):
    sent = []
    def post(url, *, payload, **kw):
        sent.append(payload)
        if url.endswith("/chat/completions"):
            data = "data:image/png;base64," + base64.b64encode(png("blue")).decode()
            return {"choices": [{"message": {"images": [{"image_url": {"url": data}}]}}]}, None
        return {"data": [{"b64_json": base64.b64encode(png("blue")).decode()}]}, None
    monkeypatch.setattr(module, "post_json", post)
    return sent


def test_xai_total_source_limit_allows_three_and_refuses_four(tmp_path, backend, monkeypatch):
    import plugins.image_gen.xai as provider
    import tools.image_generation_tool as tool
    instance = provider.XAIImageGenProvider()
    sent = _post_recorder(provider, monkeypatch)
    monkeypatch.setattr(provider, "resolve_xai_http_credentials", lambda: {"api_key": "test", "base_url": "http://localhost/v1"})
    monkeypatch.setattr(tool, "_read_configured_image_provider", lambda: "xai")
    monkeypatch.setattr(tool, "_get_plugin_provider", lambda name: instance)
    paths = photos(tmp_path, 4)
    scope = refs.snapshot_photos(_group(), [str(p) for p in paths])
    labels = list(scope.photos)
    result = bound(scope, {"reference_image_urls": labels[:3]})
    assert result["success"], result
    assert [base64.b64decode(x["url"].partition(",")[2]) for x in sent[0]["images"]] == [p.read_bytes() for p in paths[:3]]
    assert refs.TOO_MANY in bound(scope, {"reference_image_urls": labels})["error"]
    assert len(sent) == 1


def test_openrouter_uses_dispatch_model_and_forced_surface_limits(tmp_path, backend, monkeypatch):
    import plugins.image_gen.openrouter as provider
    import tools.image_generation_tool as tool
    instance = provider.OpenRouterCompatImageProvider(
        provider_name="openrouter", display_name="OpenRouter", runtime_name="openrouter", config_key="openrouter",
        model_env_var="OPENROUTER_IMAGE_MODEL", setup_schema=None, supports_image_api=True)
    monkeypatch.setattr(instance, "_credentials", lambda: ("test", "http://localhost/v1"))
    monkeypatch.setattr(tool, "_read_configured_image_provider", lambda: "openrouter")
    monkeypatch.setattr(tool, "_get_plugin_provider", lambda name: instance)
    model = ["microsoft/mai-image-2.5"]
    monkeypatch.setattr(tool, "_read_configured_image_model", lambda: model[0])
    monkeypatch.setenv("OPENROUTER_IMAGE_MODEL", "openai/gpt-image-2")
    sent = _post_recorder(provider, monkeypatch)
    paths = photos(tmp_path, 4)
    scope = refs.snapshot_photos(_group(), [str(p) for p in paths])
    labels = list(scope.photos)
    assert bound(scope, {"image_url": labels[0]})["success"]
    assert sent[-1]["model"] == "microsoft/mai-image-2.5"
    assert len(sent[-1]["input_references"]) == 1
    assert refs.TOO_MANY in bound(scope, {"reference_image_urls": labels[:2]})["error"]
    model[0] = "openai/gpt-image-2"
    monkeypatch.setenv("OPENROUTER_IMAGE_API_SURFACE", "chat")
    assert bound(scope, {"reference_image_urls": labels[:3]})["success"]
    content = sent[-1]["messages"][0]["content"]
    assert len([p for p in content if p["type"] == "image_url"]) == 3
    assert refs.TOO_MANY in bound(scope, {"reference_image_urls": labels})["error"]
    model[0] = "krea/krea-2-medium"
    assert REFERENCE_REFUSAL in bound(scope, {"image_url": labels[0]})["error"]
    assert len(sent) == 2


def test_managed_krea_unavailable_uses_same_fal_fallback_as_dispatch(tmp_path, backend, monkeypatch):
    import tools.image_generation_tool as tool
    from plugins.image_gen import krea
    # Restore the real route resolver that the shared fake-backend fixture overrides.
    monkeypatch.setattr(tool, "_managed_model_plugin", _real_managed_model_plugin)
    monkeypatch.setattr(tool, "_read_configured_image_provider", lambda: None)
    monkeypatch.setattr(tool, "_read_configured_image_model", lambda: "krea-2-medium")
    monkeypatch.setattr(krea, "_resolve_managed_krea_gateway", lambda: None)
    sent = []
    def fal(prompt, aspect_ratio, **sources):
        sent.append(sources)
        return json.dumps({"success": True, "image": str(tmp_path / "generated.png")})
    monkeypatch.setattr(tool, "image_generate_tool", fal)
    monkeypatch.setattr(tool, "check_fal_api_key", lambda: True)
    monkeypatch.setattr(tool, "_load_fal_client", lambda: None)
    _, event, _ = prepare(tmp_path, "native")
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    assert bound(scope, {"image_url": label})["success"]
    assert sent[0]["image_url"].startswith("data:")
    from tests.gateway.test_group_photo_provenance import describe_group
    described = describe_group()
    assert "edit" in described["description"]
    assert "Style reference" not in described["parameters"]["properties"]["image_url"]["description"]
    assert REFERENCE_REFUSAL in bound(scope, {"image_url": str(tmp_path / "0.png")})["error"]
    assert len(sent) == 1


def test_unregistered_explicit_backend_does_not_advertise_fal_edits(tmp_path, backend, monkeypatch):
    _, event, _ = prepare(tmp_path, "native")
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    assert bound(scope, {"image_url": label})["success"]
    monkeypatch.setattr("tools.image_generation_tool._read_configured_image_provider", lambda: "missing")
    monkeypatch.setattr("tools.image_generation_tool._get_plugin_provider", lambda name: None)
    _, refused, _ = prepare(tmp_path, "native")
    assert refused._group_photo_scope is None
    assert REFERENCE_REFUSAL in bound(scope, {"image_url": label})["error"]
    assert len(backend) == 1


@pytest.mark.parametrize("disabled", ["safe", "image_gen"])
def test_disabled_composite_removes_photo_authority(tmp_path, backend, monkeypatch, disabled):
    _, event, _ = prepare(tmp_path, "native")
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    assert bound(scope, {"image_url": label})["success"]
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"agent": {"disabled_toolsets": [disabled]}})
    _, event, text = prepare(tmp_path, "native")
    assert event._group_photo_scope is None and "чтобы перерисовать" not in text
    assert REFERENCE_REFUSAL in bound(None, {"image_url": label})["error"]


@pytest.mark.parametrize("change", ["endpoint", "credential"])
def test_route_change_between_validation_and_dispatch_sends_no_bytes(tmp_path, backend, monkeypatch, change):
    import tools.image_generation_tool as tool
    import plugins.image_gen.openai as provider
    route = ["http://localhost/v1", "test"]
    monkeypatch.setattr(provider, "_resolve_endpoint", lambda: tuple(route))
    _, event, _ = prepare(tmp_path, "native")
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    assert bound(scope, {"image_url": label})["success"]
    original = tool._confine_source_images
    def confine(*args):
        route[0 if change == "endpoint" else 1] = "http://changed.local/v1" if change == "endpoint" else "other-key"
        return original(*args)
    monkeypatch.setattr(tool, "_confine_source_images", confine)
    assert bound(scope, {"image_url": label})["error_type"] == "route_changed"
    assert len(backend) == 1
    assert backend.probes == [("http://localhost/v1/images/models", "Bearer test")]


@pytest.mark.parametrize("identifier", ["", "   "])
def test_blank_editing_model_is_not_authority(tmp_path, backend, monkeypatch, identifier):
    monkeypatch.setattr("plugins.image_gen.openai._resolve_endpoint", lambda: ("http://localhost/v1", "test"))
    backend.catalogue = {"models": [{"id": identifier, "edits": True}]}
    _, event, text = prepare(tmp_path, "native")
    assert event._group_photo_scope is None and "чтобы перерисовать" not in text
    assert REFERENCE_REFUSAL in bound(None, {"image_url": "ref:invented:1"})["error"]
    assert not backend


def test_unavailable_selected_provider_cannot_borrow_fal_credential(tmp_path, backend, monkeypatch):
    import tools.image_generation_tool as tool
    _, event, _ = prepare(tmp_path, "native")
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    assert bound(scope, {"image_url": label})["success"]
    monkeypatch.setattr(tool, "check_fal_api_key", lambda: True)
    monkeypatch.setattr(tool, "_load_fal_client", lambda: None)
    monkeypatch.setattr(tool, "_read_configured_image_provider", lambda: "unavailable")
    provider = SimpleNamespace(is_available=lambda: False, edit_capabilities=lambda **kw: {
        "modalities": ["text", "image"], "max_reference_images": 3})
    monkeypatch.setattr(tool, "_get_plugin_provider", lambda name: provider)
    _, refused, _ = prepare(tmp_path, "native")
    assert refused._group_photo_scope is None
    assert REFERENCE_REFUSAL in bound(scope, {"image_url": label})["error"]
    assert len(backend) == 1


def test_oversized_decoded_raster_is_rejected_before_load(tmp_path, backend, monkeypatch):
    import struct
    import zlib
    from PIL import PngImagePlugin
    paths = photos(tmp_path, 2)
    raw = bytearray(paths[0].read_bytes())
    raw[16:24] = struct.pack(">II", refs.MAX_PHOTO_PIXELS + 1, 1)
    raw[29:33] = struct.pack(">I", zlib.crc32(raw[12:29]))
    paths[0].write_bytes(raw)
    real_load = PngImagePlugin.PngImageFile.load
    def load(image, *a, **kw):
        assert image.width * image.height <= refs.MAX_PHOTO_PIXELS
        return real_load(image, *a, **kw)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", load)
    _, event, _ = prepare(tmp_path, "text", paths)
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    assert label.endswith(":2")
    assert bound(scope, {"image_url": label})["success"]
    assert REFERENCE_REFUSAL in bound(scope, {"image_url": label.rsplit(":", 1)[0] + ":1"})["error"]


def test_animated_png_is_retained_as_one_static_frame(tmp_path, backend):
    paths = photos(tmp_path, 1)
    Image.new("RGB", (16, 16), "red").save(paths[0], "PNG", save_all=True,
        append_images=[Image.new("RGB", (16, 16), "green")], duration=100)
    _, event, _ = prepare(tmp_path, "text", paths)
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    assert bound(scope, {"image_url": label})["success"]
    with Image.open(io.BytesIO(backend[-1][1][0])) as image:
        assert not getattr(image, "is_animated", False)
    assert REFERENCE_REFUSAL in bound(scope, {"image_url": str(paths[0])})["error"]


def test_empty_primary_cannot_hide_in_valid_reference_list(tmp_path, backend):
    _, event, _ = prepare(tmp_path, "native")
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    assert bound(scope, {"image_url": label})["success"]
    assert REFERENCE_REFUSAL in bound(scope, {"image_url": "", "reference_image_urls": [label]})["error"]
    assert len(backend) == 1


def test_catalogue_cache_coalesces_concurrent_misses(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import plugins.image_gen.openai as provider
    entered, release = threading.Event(), threading.Event()
    calls = []
    def probe(*a):
        calls.append(a)
        entered.set()
        assert release.wait(10)
        return {"modalities": ["text", "image"], "max_reference_images": 1}
    monkeypatch.setattr(provider, "_probe_edit_capabilities", probe)
    provider._EDIT_CAPABILITIES_CACHE.clear()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(provider._cached_edit_capabilities, "http://unique.local/v1", "key", "model")
        assert entered.wait(10)
        second = pool.submit(provider._cached_edit_capabilities, "http://unique.local/v1", "key", "model")
        release.set()
        assert first.result(10) == second.result(10)
    assert len(calls) == 1


def test_cached_catalogue_is_scoped_to_dispatched_model(tmp_path, backend, monkeypatch):
    import plugins.image_gen.openai as provider
    monkeypatch.setattr(provider, "_resolve_endpoint", lambda: ("http://localhost/v1", "test"))
    model = ["first-editor"]
    monkeypatch.setattr(provider, "_resolve_model", lambda: (model[0], {"api_model": model[0], "quality": None}))
    backend.catalogue = {"models": [{"id": name, "edits": True} for name in ["first-editor", "second-editor"]]}
    _, event, _ = prepare(tmp_path, "native")
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    assert bound(scope, {"image_url": label})["success"]
    model[0] = "second-editor"
    assert bound(scope, {"image_url": label})["success"]
    assert backend.models == backend.usage_models == ["first-editor", "second-editor"]
    assert len(backend.probes) == 2
    assert REFERENCE_REFUSAL in bound(scope, {"image_url": "ref:invented:1"})["error"]


@pytest.mark.parametrize("actual", [None, "proxy-selected-editor"])
def test_proxy_model_metadata_never_claims_requested_model_was_executed(tmp_path, backend, monkeypatch, actual):
    monkeypatch.setattr("plugins.image_gen.openai._resolve_endpoint", lambda: ("http://localhost/v1", "test"))
    backend.response_model = actual
    _, event, _ = prepare(tmp_path, "native")
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    result = bound(scope, {"image_url": label})
    assert result["success"]
    assert result["requested_model"] == backend.models[-1] == "editor"
    assert result["model"] == backend.usage_models[-1] == (actual or "unknown")
    assert REFERENCE_REFUSAL in bound(scope, {"image_url": "ref:invented:1"})["error"]
