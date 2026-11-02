"""session_search must return nothing a stranger wrote in a Telegram group.

The owner's DM turn has terminal and memory, so group text reaching it is a prompt-injection path.
Each test takes the owner's side: a real SessionDB, the real tool, and a search for the group's words.
"""
import json

import pytest

from gateway.config import Platform
from gateway.group_limits import is_limited_group_chat
from hermes_state import SessionDB
from tools.registry import registry
from tools.session_search_tool import session_search

GROUP_WORD = "strangerpayload"
DM_WORD = "ownerdiary"


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _session(db, sid, *, source="telegram", chat_id="111", chat_type="dm", text=DM_WORD, **kwargs):
    db.create_session(sid, source=source, chat_id=chat_id, chat_type=chat_type, **kwargs)
    message_id = db.append_message(sid, role="user", content=f"{text} from {sid}") if text else None
    return message_id


def _group(db, sid="group", **kwargs):
    kwargs.setdefault("chat_id", "-1001")
    kwargs.setdefault("chat_type", "group")
    return _session(db, sid, text=GROUP_WORD, **kwargs)


def _search(db, **kwargs):
    return json.loads(session_search(db=db, **kwargs))


def _found(payload):
    return {entry["session_id"] for entry in payload.get("results", [])}


def _assert_clean(payload):
    echoed = {key: value for key, value in payload.items() if key != "query"}  # the caller's own words
    assert GROUP_WORD not in json.dumps(echoed)


def test_keyword_search_finds_the_dm_and_not_the_group(db):
    _session(db, "dm")
    _group(db)
    db.append_message("dm", role="user", content=f"{GROUP_WORD} quoted by the owner")
    db.append_message("group", role="user", content=DM_WORD)

    payload = _search(db, query=DM_WORD, detail="full")

    assert _found(payload) == {"dm"}


def test_keyword_search_with_only_a_group_match_is_empty(db):
    _session(db, "dm")
    _group(db)

    payload = _search(db, query=GROUP_WORD)

    assert payload["success"] is True
    assert _found(payload) == set()
    _assert_clean(payload)


def test_cjk_search_route_is_filtered_too(db):
    _session(db, "dm", text="你好世界日记")
    _group(db)
    db.append_message("group", role="user", content="你好世界陌生人")

    assert _found(_search(db, query="你好世界")) == {"dm"}


def test_tool_output_search_is_filtered_too(db):
    _session(db, "dm")
    _group(db)
    db.append_message("group", role="tool", content=f"{GROUP_WORD} tool output", tool_call_id="c1")

    payload = _search(db, query=GROUP_WORD, role_filter="tool")

    assert _found(payload) == set()
    _assert_clean(payload)


def test_a_busy_group_does_not_crowd_the_dm_out_of_the_scan(db):
    _session(db, "dm", text=f"{DM_WORD} needle")
    _session(db, "group", chat_id="-1001", chat_type="group", text=None)
    db.append_messages_batch("group", [{"role": "user", "content": "needle needle needle"} for _ in range(320)])

    assert _found(_search(db, query="needle")) == {"dm"}


def test_title_match_returns_nothing_from_a_group(db):
    _group(db)
    db.set_session_title("group", "Secret plans")

    payload = _search(db, query="Secret plans")

    assert _found(payload) == set()
    _assert_clean(payload)


def test_title_match_still_finds_a_dm(db):
    _session(db, "dm")
    db.set_session_title("dm", "Owner plans")

    assert _found(_search(db, query="Owner plans")) == {"dm"}


def test_recent_sessions_mode_lists_the_dm_and_not_the_group(db):
    _session(db, "dm")
    _group(db)

    payload = _search(db)

    assert {entry["session_id"] for entry in payload["results"]} == {"dm"}
    _assert_clean(payload)


def test_many_recent_groups_do_not_crowd_the_dm_out_of_recent_sessions(db):
    _session(db, "dm")
    for index in range(150):
        _group(db, f"group{index}")

    assert {entry["session_id"] for entry in _search(db)["results"]} == {"dm"}


def test_read_by_id_refuses_a_group_and_serves_a_dm(db):
    _session(db, "dm")
    _group(db)

    refused = _search(db, session_id="group")

    assert refused["success"] is False
    _assert_clean(refused)
    assert DM_WORD in json.dumps(_search(db, session_id="dm"))


def test_scroll_by_id_refuses_a_group_and_serves_a_dm(db):
    dm_message = _session(db, "dm")
    group_message = _group(db)

    refused = _search(db, session_id="group", around_message_id=group_message)

    assert refused["success"] is False
    _assert_clean(refused)
    assert DM_WORD in json.dumps(_search(db, session_id="dm", around_message_id=dm_message))


def test_lineage_dm_child_of_a_group_is_hidden_from_every_path(db):
    """A compressed child holds a summary of its parent, whatever its own chat type says."""
    _group(db)
    db.end_session("group", "compression")
    message_id = _session(db, "dm_child", text=f"{GROUP_WORD} summarised", parent_session_id="group")

    for payload in (
        _search(db, query=GROUP_WORD),
        _search(db, session_id="dm_child"),
        _search(db, session_id="dm_child", around_message_id=message_id),
        _search(db),
    ):
        _assert_clean(payload)
        assert "dm_child" not in _found(payload)


def test_lineage_dm_parent_keeps_its_own_text_but_not_its_group_child(db):
    dm_message = _session(db, "dm")
    db.end_session("dm", "compression")
    group_message = _group(db, "group_child", parent_session_id="dm")

    assert _found(_search(db, query=DM_WORD)) == {"dm"}
    assert GROUP_WORD not in json.dumps(_search(db, query=DM_WORD, detail="full"))
    assert GROUP_WORD not in json.dumps(_search(db, session_id="dm"))
    assert GROUP_WORD not in json.dumps(_search(db, session_id="dm", around_message_id=dm_message))
    # The scroll shape rebinds a parent id to a descendant's anchor; that hop must not reach the group.
    rebound = _search(db, session_id="dm", around_message_id=group_message)
    assert rebound["success"] is False
    _assert_clean(rebound)


def test_lineage_recent_sessions_does_not_project_a_dm_root_onto_a_group_tip(db):
    _session(db, "dm")
    db.end_session("dm", "compression")
    _group(db, "group_tip", parent_session_id="dm")

    _assert_clean(_search(db))


def test_lineage_subagent_of_a_group_is_refused_by_id(db):
    """Its source is not Telegram, but its text came from the group's turn."""
    _group(db)
    _session(db, "child", source="subagent", chat_id=None, chat_type=None, text=GROUP_WORD,
             parent_session_id="group")

    refused = _search(db, session_id="child")

    assert refused["success"] is False
    _assert_clean(refused)


def test_a_dm_lineage_is_not_hidden(db):
    _session(db, "dm_root", text=f"{DM_WORD} early")
    db.end_session("dm_root", "compression")
    _session(db, "dm_tip", text="ownerlater", parent_session_id="dm_root")

    assert _found(_search(db, query=DM_WORD)) == {"dm_root"}
    assert "ownerlater" in json.dumps(_search(db, session_id="dm_tip"))


def test_lineage_proven_dm_descending_from_a_legacy_row_stays_found(db):
    """Rows older than the chat columns have none, and a per-turn refresh only fills the live one."""
    _session(db, "legacy", chat_id=None, chat_type=None, text="legacyword")
    db.end_session("legacy", "compression")
    message_id = _session(db, "dm_tip", parent_session_id="legacy")

    assert _found(_search(db, query=DM_WORD)) == {"dm_tip"}
    assert _search(db, session_id="dm_tip")["success"] is True
    assert _search(db, session_id="dm_tip", around_message_id=message_id)["success"] is True
    assert "dm_tip" in {entry["session_id"] for entry in _search(db)["results"]}
    # The legacy row itself proves nothing, so it stays hidden.
    assert _search(db, session_id="legacy")["success"] is False
    assert _found(_search(db, query="legacyword")) == set()


def test_lineage_group_above_a_legacy_row_still_hides_a_dm_descendant(db):
    _group(db)
    db.end_session("group", "compression")
    _session(db, "legacy", chat_id=None, chat_type=None, text=GROUP_WORD, parent_session_id="group")
    db.end_session("legacy", "compression")
    _session(db, "dm_tip", text=GROUP_WORD, parent_session_id="legacy")

    assert _search(db, session_id="dm_tip")["success"] is False


def test_lineage_unproven_child_of_a_legacy_row_is_hidden(db):
    _session(db, "legacy", chat_id=None, chat_type=None, text=GROUP_WORD)
    _session(db, "child", source="subagent", chat_id=None, chat_type=None, text=GROUP_WORD,
             parent_session_id="legacy")

    assert _search(db, session_id="child")["success"] is False


@pytest.mark.parametrize("chat_type", ["group", "channel", "thread", "supergroup", "GROUP", " forum "])
def test_every_non_dm_chat_type_is_hidden(db, chat_type):
    _group(db, chat_type=chat_type, chat_id="-1001")

    assert _search(db, session_id="group")["success"] is False


@pytest.mark.parametrize("chat_type", ["dm", "private", "DM", " dm "])
def test_dm_chat_types_are_found(db, chat_type):
    _session(db, "dm", chat_type=chat_type)

    assert _found(_search(db, query=DM_WORD)) == {"dm"}


@pytest.mark.parametrize("chat_type", ["dm", "private", "group", "channel", "thread", "GROUP", " dm ", "weird"])
def test_rule_agrees_with_group_limits(db, chat_type):
    _session(db, "row", chat_type=chat_type, chat_id="111")

    assert db.session_lineage_has_telegram_group("row") is is_limited_group_chat(Platform.TELEGRAM, chat_type)


@pytest.mark.parametrize("chat_type", [None, "", "  "])
def test_empty_chat_type_with_a_positive_chat_id_is_a_dm(db, chat_type):
    """Rows from before the chat_type column: Telegram DM ids are positive, so this one is the owner's."""
    _session(db, "dm", chat_type=chat_type, chat_id="111")

    assert _found(_search(db, query=DM_WORD)) == {"dm"}


@pytest.mark.parametrize("chat_id", ["-1001", None, "", "not-a-number"])
@pytest.mark.parametrize("chat_type", [None, ""])
def test_empty_chat_type_without_a_dm_id_fails_closed(db, chat_type, chat_id):
    _group(db, chat_type=chat_type, chat_id=chat_id)

    assert _search(db, session_id="group")["success"] is False
    assert _found(_search(db, query=GROUP_WORD)) == set()


def test_a_negative_chat_id_is_a_group_whatever_the_row_says(db):
    _group(db, chat_type="dm", chat_id="-1001")

    assert _search(db, session_id="group")["success"] is False


@pytest.mark.parametrize("source,chat_id,chat_type", [
    ("cli", None, None),
    ("discord", "-5", "group"),
    ("slack", "C1", "channel"),
])
def test_other_platforms_stay_searchable(db, source, chat_id, chat_type):
    _session(db, "other", source=source, chat_id=chat_id, chat_type=chat_type)

    assert _found(_search(db, query=DM_WORD)) == {"other"}
    assert _search(db, session_id="other")["success"] is True


def test_another_profiles_group_is_hidden(db, tmp_path, monkeypatch):
    other = SessionDB(tmp_path / "other.db")
    _group(other)
    _session(other, "other_dm")
    monkeypatch.setattr("tools.session_search_tool._resolve_profile_db", lambda _profile: other)

    assert _found(_search(db, query=GROUP_WORD, profile="work")) == set()
    assert _search(db, session_id="group", profile="work")["success"] is False
    assert _found(_search(db, query=DM_WORD, profile="work")) == {"other_dm"}


def test_a_failed_group_lookup_hides_the_session(db, monkeypatch):
    _session(db, "dm")

    def _boom(_session_id):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(db, "session_lineage_has_telegram_group", _boom)

    assert _search(db, session_id="dm")["success"] is False


def test_real_tool_dispatch_is_filtered(db):
    _session(db, "dm")
    _group(db)

    payload = json.loads(registry.dispatch("session_search", {"query": GROUP_WORD}, db=db))

    assert _found(payload) == set()
    _assert_clean(payload)
