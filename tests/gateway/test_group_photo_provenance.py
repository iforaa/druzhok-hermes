"""Native Telegram attachment authority survives admission; viewer uploads never acquire it."""

import asyncio
import contextvars
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from gateway import group_image_refs as refs
from gateway.group_limits import REFERENCE_REFUSAL
from gateway.platforms.base import merge_pending_message_event
from gateway.platforms.event import MessageType
from gateway.session_context import set_session_vars
from tests.gateway.test_group_images import GROUP_ID, _adapter, _event, _group, _inbound_runner, _prepare
from tests.gateway.test_group_photo_refs import bound, png, prepare
from tests.gateway.test_group_photo_routes import photos

pytest_plugins = ("tests.gateway.test_group_photo_refs",)


def native_photo(adapter, path):
    event = _event(_group(), "redraw", message_type=MessageType.PHOTO,
                   media_urls=[str(path)], media_types=["image/png"])
    # The transport boundary stamps provenance before it needs the message-handler wiring.
    asyncio.run(adapter.handle_message(event))
    return event


def prepare_event(runner, event):
    with patch("tools.vision_tools.vision_analyze_tool", AsyncMock(return_value='{"success":true,"analysis":"red"}')):
        return _prepare(runner, event)


@pytest.mark.parametrize("mode", ["native", "text"])
def test_viewer_upload_on_same_group_route_has_no_redraw_authority(tmp_path, backend, mode):
    runner, _ = _inbound_runner({"group_media": ["image"], "group_toolsets": ["image_gen"]}, mode)
    paths = photos(tmp_path, 2)
    native = native_photo(_adapter(), paths[0])
    prepare_event(runner, native)
    scope = native._group_photo_scope
    label = next(iter(scope.photos))
    assert bound(scope, {"image_url": label})["success"]
    # session.submit reconstructs this event with the group's source and viewer attachments.
    viewer = _event(_group(), f"redraw {label}", media_urls=[str(paths[1])], media_types=["image/png"],
                    metadata={"native_media_origin": ["telegram", GROUP_ID]})
    text = prepare_event(runner, viewer)
    assert viewer._group_photo_scope is None
    assert "чтобы перерисовать" not in text
    assert REFERENCE_REFUSAL in bound(None, {"image_url": label})["error"]
    assert len(backend) == 1


@pytest.mark.parametrize("mode", ["native", "text"])
def test_native_admission_retains_attachment_provenance(tmp_path, backend, mode):
    from gateway.session_envelope import snapshot_native, restore_native
    runner, _ = _inbound_runner({"group_media": ["image"], "group_toolsets": ["image_gen"]}, mode)
    runner.session_store = SimpleNamespace(_generate_session_key=lambda source: "k")
    runner._adapter_flag = lambda *a: False
    runner._is_user_authorized_for_source = lambda *a, **kw: True
    event = native_photo(_adapter(), photos(tmp_path, 1)[0])
    payload = snapshot_native(runner, event)
    restored = restore_native(payload)
    prepare_event(runner, restored)
    scope = restored._group_photo_scope
    label = next(iter(scope.photos))
    assert bound(scope, {"image_url": label})["success"]
    # Old or unmarked envelopes contain no retained proof and cannot grant authority.
    payload["native_text_v1"].pop("native_media_origin")
    unmarked = restore_native(payload)
    prepare_event(runner, unmarked)
    assert unmarked._group_photo_scope is None
    assert REFERENCE_REFUSAL in bound(None, {"image_url": label})["error"]
    event._native_media_origin = ["telegram", GROUP_ID]  # a JSON-like value is not a native stamp
    malformed = restore_native(snapshot_native(runner, event))
    prepare_event(runner, malformed)
    assert malformed._group_photo_scope is None
    assert len(backend) == 1


def test_native_batch_cannot_upgrade_viewer_attachment_provenance(tmp_path, backend):
    runner, _ = _inbound_runner({"group_media": ["image"], "group_toolsets": ["image_gen"]})
    adapter = _adapter()
    paths = photos(tmp_path, 3)
    first, second = native_photo(adapter, paths[0]), native_photo(adapter, paths[1])
    pending = {"k": first}
    merge_pending_message_event(pending, "k", second)
    prepare_event(runner, first)
    labels = list(first._group_photo_scope.photos)
    assert bound(first._group_photo_scope, {"reference_image_urls": labels})["success"]
    viewer = _event(_group(), "private upload", message_type=MessageType.PHOTO,
                    media_urls=[str(paths[2])], media_types=["image/png"])
    merge_pending_message_event(pending, "k", viewer)
    prepare_event(runner, first)
    assert first._group_photo_scope is None
    assert REFERENCE_REFUSAL in bound(None, {"image_url": labels[0]})["error"]
    assert len(backend) == 1


@pytest.mark.parametrize("mode", ["native", "text"])
@pytest.mark.parametrize("broken", ["png_crc", "jpeg_scan"])
def test_invalid_pixels_skip_only_the_bad_photo(tmp_path, backend, mode, broken):
    paths = photos(tmp_path, 2)
    if broken == "png_crc":
        raw = bytearray(png())
        index = raw.index(b"IDAT")
        crc = index + 4 + int.from_bytes(raw[index-4:index], "big")
        raw[crc] ^= 255
        paths[0].write_bytes(raw)
    else:
        Image.effect_noise((256, 256), 100).convert("RGB").save(paths[0], "JPEG")
        paths[0].write_bytes(paths[0].read_bytes()[:-20])
    _, event, _ = prepare(tmp_path, mode, paths)
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    assert label.endswith(":2")
    assert bound(scope, {"image_url": label})["success"]
    assert REFERENCE_REFUSAL in bound(scope, {"image_url": label.rsplit(":", 1)[0] + ":1"})["error"]
    assert len(backend) == 1


def describe_group():
    from model_tools import handle_function_call
    def run():
        set_session_vars(platform="telegram", chat_id=GROUP_ID, chat_type="group")
        return json.loads(handle_function_call("tool_describe", {"names": ["image_generate"]}, enabled_toolsets=["image_gen"]))
    return contextvars.copy_context().run(run)["tools"]["image_generate"]


def test_group_schema_and_hints_share_proxy_source_budget(tmp_path, backend, monkeypatch):
    monkeypatch.setattr("plugins.image_gen.openai._resolve_endpoint", lambda: ("http://localhost/v1", "test"))
    _, event, _ = prepare(tmp_path, "native", photos(tmp_path, 2))
    scope = event._group_photo_scope
    labels = list(scope.photos)
    assert "до 1 фото суммарно" in scope.hint(event.media_urls[0])
    schema = describe_group()
    assert "image_url" in schema["parameters"]["properties"]
    assert "reference_image_urls" not in schema["parameters"]["properties"]
    assert bound(scope, {"image_url": labels[0]})["success"]
    assert refs.TOO_MANY in bound(scope, {"reference_image_urls": labels})["error"]


def test_missing_backend_and_unexposed_tool_never_issue_labels(tmp_path, backend, monkeypatch):
    import tools.image_generation_tool as tool
    from tools.registry import registry, invalidate_check_fn_cache
    _, event, _ = prepare(tmp_path, "native")
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    assert bound(scope, {"image_url": label})["success"]
    with monkeypatch.context() as m:
        m.setattr(tool, "_read_configured_image_provider", lambda: None)
        m.setattr(tool, "check_fal_api_key", lambda: False)
        _, refused, _ = prepare(tmp_path, "native")
        assert refused._group_photo_scope is None
        assert REFERENCE_REFUSAL in bound(scope, {"image_url": label})["error"]
    monkeypatch.setattr(registry.get_entry("image_generate"), "check_fn", lambda: False)
    invalidate_check_fn_cache()
    _, refused, _ = prepare(tmp_path, "native")
    assert refused._group_photo_scope is None
    assert "not available" in bound(scope, {"image_url": label})["error"]
    assert len(backend) == 1


@pytest.mark.parametrize("surface", ["chat", "images"])
def test_openrouter_requires_image_input_support(tmp_path, backend, monkeypatch, surface):
    import plugins.image_gen.openrouter as provider
    import tools.image_generation_tool as tool
    from tests.gateway.test_group_photo_routes import _post_recorder
    model = "test/image-output-model"
    instance = provider.OpenRouterCompatImageProvider(
        provider_name="openrouter", display_name="OpenRouter", runtime_name="openrouter", config_key="openrouter",
        model_env_var="OPENROUTER_IMAGE_MODEL", setup_schema=None, supports_image_api=True)
    monkeypatch.setattr(instance, "_credentials", lambda: ("test", "http://localhost/v1"))
    monkeypatch.setattr(provider, "_select_surface", lambda *a: surface)
    rows = [{"id": model, "input_modalities": ["text", "image"], "max_reference_images": 2}]
    monkeypatch.setattr(instance, "_image_api_live_models", lambda: rows)
    monkeypatch.setattr(instance, "_live_models", lambda: rows)
    monkeypatch.setattr(tool, "_read_configured_image_provider", lambda: "openrouter")
    monkeypatch.setattr(tool, "_read_configured_image_model", lambda: model)
    monkeypatch.setattr(tool, "_get_plugin_provider", lambda name: instance)
    sent = _post_recorder(provider, monkeypatch)
    _, event, _ = prepare(tmp_path, "native", photos(tmp_path, 2))
    scope = event._group_photo_scope
    labels = list(scope.photos)
    assert bound(scope, {"reference_image_urls": labels})["success"]
    rows[0]["input_modalities"] = ["text"]
    _, refused, _ = prepare(tmp_path, "native")
    assert refused._group_photo_scope is None
    assert REFERENCE_REFUSAL in bound(scope, {"image_url": labels[0]})["error"]
    assert len(sent) == 1


def test_stale_turn_cannot_replace_successors_revocable_scope(tmp_path, backend):
    from gateway.session_state import SessionState
    from gateway.turn_context import TurnContext
    runner, old, _ = prepare(tmp_path, "native")
    old_scope = old._group_photo_scope
    old_label = next(iter(old_scope.photos))
    assert bound(old_scope, {"image_url": old_label})["success"]
    _, successor, _ = prepare(tmp_path, "native")
    current_scope = successor._group_photo_scope
    current_label = next(iter(current_scope.photos))
    state = SessionState()
    state.turn.group_photo_scope = current_scope
    runner._session_state = lambda key: state
    runner._peek_session_state = lambda key: state
    stale = TurnContext(source=_group(), session_key="k", _run_still_current=lambda: False)
    def run():
        set_session_vars(platform="telegram", chat_id=GROUP_ID, chat_type="group", turn_event=old)
        with refs.run_photo_scope(runner, stale):
            from tests.gateway.test_group_photo_refs import dispatch
            assert REFERENCE_REFUSAL in dispatch({"image_url": old_label})["error"]
    contextvars.copy_context().run(run)
    assert state.turn.group_photo_scope is current_scope
    assert bound(current_scope, {"image_url": current_label})["success"]
    runner._begin_session_run_generation("k")
    assert REFERENCE_REFUSAL in bound(current_scope, {"image_url": current_label})["error"]
    assert len(backend) == 2


@pytest.mark.parametrize("origin", [("telegram", "111"), ("telegram", "-1009999"), ("discord", GROUP_ID)])
def test_native_photo_rerouted_from_other_chat_or_platform_has_no_authority(tmp_path, backend, origin):
    runner, event, _ = prepare(tmp_path, "native")
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    assert bound(scope, {"image_url": label})["success"]
    event._native_media_origin = origin
    text = prepare_event(runner, event)
    assert event._group_photo_scope is None and "чтобы перерисовать" not in text
    assert REFERENCE_REFUSAL in bound(None, {"image_url": label})["error"]
    assert len(backend) == 1


def test_live_edit_catalogue_does_not_borrow_another_profiles_input_modalities(tmp_path, backend, monkeypatch):
    import plugins.image_gen.openrouter as provider
    import tools.image_generation_tool as tool
    from tests.gateway.test_group_photo_routes import _post_recorder
    instance = provider.OpenRouterCompatImageProvider(
        provider_name="openrouter", display_name="OpenRouter", runtime_name="openrouter", config_key="openrouter",
        model_env_var="OPENROUTER_IMAGE_MODEL", setup_schema=None, supports_image_api=True)
    key = ["account-a"]
    monkeypatch.setattr(instance, "_credentials", lambda: (key[0], "http://localhost/v1"))
    monkeypatch.setattr(provider, "_select_surface", lambda *a: "chat")
    probes = []
    def catalogue(base_url, api_key, **kw):
        probes.append((base_url, api_key))
        return [{"id": "custom-image", "input_modalities": ["image"] if api_key == "account-a" else ["text"],
                 "max_reference_images": 2}]
    monkeypatch.setattr(provider, "_fetch_catalog", catalogue)
    monkeypatch.setattr(tool, "_read_configured_image_provider", lambda: "openrouter")
    monkeypatch.setattr(tool, "_read_configured_image_model", lambda: "custom-image")
    monkeypatch.setattr(tool, "_get_plugin_provider", lambda name: instance)
    sent = _post_recorder(provider, monkeypatch)
    _, event, _ = prepare(tmp_path, "native")
    scope = event._group_photo_scope
    label = next(iter(scope.photos))
    assert bound(scope, {"image_url": label})["success"]
    key[0] = "account-b"
    _, refused, _ = prepare(tmp_path, "native")
    assert refused._group_photo_scope is None
    assert REFERENCE_REFUSAL in bound(scope, {"image_url": label})["error"]
    assert probes == [("http://localhost/v1", "account-a"), ("http://localhost/v1", "account-b")]
    assert len(sent) == 1


@pytest.mark.parametrize("fingerprinted", [False, True])
@pytest.mark.parametrize("changed", [None, "text", "bytes", "raw"])
def test_legacy_native_media_retry_preserves_old_authority(tmp_path, backend, monkeypatch, fingerprinted, changed):
    import copy
    import sqlite3
    from gateway.session_envelope import snapshot_native, restore_native
    from gateway import session_ingress_media as media
    runner, _ = _inbound_runner({"group_media": ["image"], "group_toolsets": ["image_gen"]})
    runner.session_store = SimpleNamespace(_generate_session_key=lambda source: "k")
    runner._adapter_flag = lambda *a: False
    runner._is_user_authorized_for_source = lambda *a, **kw: True
    event = native_photo(_adapter(), photos(tmp_path, 1)[0])
    retained = snapshot_native(runner, event)
    retained["native_text_v1"].pop("native_media_origin")
    raw = {"message": "original"}
    if fingerprinted:
        retained["native_text_v1"]["ingress_sha256"] = media.ingress_fingerprint(retained, raw)
    fresh = copy.deepcopy(retained)
    fresh["native_text_v1"]["native_media_origin"] = ["telegram", GROUP_ID]
    fresh["native_text_v1"]["media"][0]["path"] += "-retry"
    if changed == "text":
        fresh["text"] = "changed"
    if changed == "bytes":
        fresh["native_text_v1"]["media"][0]["sha256"] = "0"*64
    if changed == "raw":
        raw = {"message": "changed"}
    conn = sqlite3.connect(tmp_path / "retry.sqlite")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE session_admissions (status TEXT, payload_json TEXT, principal_id TEXT, target_session_id TEXT, request_id TEXT)")
    conn.execute("INSERT INTO session_admissions VALUES (?, ?, ?, ?, ?)", ("terminal", json.dumps(retained), "p", "s", "r"))
    conn.commit()
    monkeypatch.setattr(media, "release_unheld_media", lambda *a, **kw: None)
    monkeypatch.setattr(media, "drop_capture_lease", lambda *a: None)
    try:
        resolved = media.reconcile_native_retry(SimpleNamespace(_read_ctx=lambda: conn), principal_id="p",
            session_id="s", request_id="r", payload=fresh, raw=raw)
    finally:
        conn.close()
    if changed is None or (changed == "raw" and not fingerprinted):
        assert resolved == retained
        assert restore_native(resolved)._native_media_origin is None
    else:
        assert resolved != retained  # admission's exact-payload digest still refuses the changed retry


@pytest.mark.parametrize("first", ["dm", "group"])
def test_cached_schema_keeps_group_edit_policy_and_text_controls_separate(tmp_path, backend, monkeypatch, first):
    import model_tools
    import tools.image_generation_tool as tool
    provider = SimpleNamespace(is_available=lambda: True, display_name="Style", default_model=lambda: "style",
        capabilities=lambda: {"modalities": ["text", "image"], "source_image_role": "style",
            "max_reference_images": 2, "creative_controls": ["intensity"], "supports_upscale": True},
        edit_capabilities=lambda **kw: {"modalities": ["text", "image"], "source_image_role": "style",
            "max_reference_images": 2})
    monkeypatch.setattr(tool, "_get_plugin_provider", lambda name: provider)
    model_tools._tool_defs_cache.clear()
    def schema(kind):
        def run():
            set_session_vars(platform="telegram", chat_id=GROUP_ID if kind == "group" else "111", chat_type=kind)
            tools = model_tools.get_tool_definitions(enabled_toolsets=["image_gen"], quiet_mode=True, skip_tool_search_assembly=True)
            return next(t["function"]["parameters"]["properties"] for t in tools if t["function"]["name"] == "image_generate")
        return contextvars.copy_context().run(run)
    order = [first, "group" if first == "dm" else "dm", first]
    results = {kind: schema(kind) for kind in order}
    assert "image_url" in results["dm"] and "image_url" not in results["group"]
    assert {"intensity", "upscale"} <= results["dm"].keys() & results["group"].keys()
    _, refused, _ = prepare(tmp_path, "native")
    assert refused._group_photo_scope is None
    assert REFERENCE_REFUSAL in bound(None, {"image_url": "ref:invented:1"})["error"]
