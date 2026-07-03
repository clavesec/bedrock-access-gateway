"""Trifecta taint-rule tests (external-content m1 Phase D, D9).

The state machine, exercised against real conditional-write semantics
(tests/ddb_fake.py):

1. **gmail → web blocked** and **web → gmail blocked** — at both layers:
   the advisory filter drops the opposing class, and the authoritative
   ``record_tool_use`` conditional write refuses it.
2. **Client-tool passthrough untouched** — client-declared tools never
   enter the taint system; ``_parse_request`` builds their toolConfig
   identically whatever the taint state.
3. **Fail closed** — unconfigured table or store failure raises
   ``TaintStoreError``; classified tools must not execute on unknown state.
"""

import asyncio
import time

import pytest
from starlette.requests import Request

import api.taint as taint
from tests.conftest import expected_hmac
from tests.ddb_fake import FakeDynamoDB

IDENTITY = expected_hmac("owui-email", "alice@example.com")
OTHER_IDENTITY = expected_hmac("api-key-user", "svc-key-7")
CHAT_SCOPE = taint.resolve_taint_scope(IDENTITY, "chat-1234")
SESSION_SCOPE = taint.resolve_taint_scope(IDENTITY, None)

TAINT_TABLE = "TPAI-TEST-gateway-taint"


@pytest.fixture
def fake_ddb(monkeypatch):
    fake = FakeDynamoDB()
    monkeypatch.setattr(taint, "_ddb", lambda: fake)
    monkeypatch.setattr(taint, "TAINT_TABLE", TAINT_TABLE)
    return fake


# --- Classification --------------------------------------------------------------


def test_classification_registry():
    assert taint.classify_tool("web_fetch") == taint.OUTBOUND_FETCH
    assert taint.classify_tool("gmail_search") == taint.INBOUND_PRIVATE
    assert taint.classify_tool("gmail_get_message") == taint.INBOUND_PRIVATE
    # Anything else is a client tool — outside the taint system.
    assert taint.classify_tool("my_calculator") is None
    assert taint.classify_tool("") is None


# --- Scope resolution -------------------------------------------------------------


def test_scope_is_conversation_keyed_with_identity_embedded():
    assert CHAT_SCOPE == f"chat#{IDENTITY}#chat-1234"


def test_scope_falls_back_to_session_key_without_chat_id():
    assert SESSION_SCOPE == f"session#{IDENTITY}"


def test_same_chat_id_different_identities_do_not_share_scope():
    a = taint.resolve_taint_scope(IDENTITY, "chat-1234")
    b = taint.resolve_taint_scope(OTHER_IDENTITY, "chat-1234")
    assert a != b


def test_scope_requires_identity():
    with pytest.raises(ValueError):
        taint.resolve_taint_scope("", "chat-1234")


# --- Chat-Id header capture --------------------------------------------------------


def _request_with_headers(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "method": "POST", "path": "/", "headers": raw, "query_string": b""})


def _captured_chat_id(headers: dict[str, str]):
    request = _request_with_headers(headers)
    asyncio.run(taint.capture_conversation(request))
    return request.state.tpai_chat_id


def test_capture_conversation_accepts_uuid_chat_id():
    assert (
        _captured_chat_id({taint.CHAT_ID_HEADER: "9c2f4f4e-1b2a-4a0e-9a5f-3c2d1e0f9a8b"})
        == "9c2f4f4e-1b2a-4a0e-9a5f-3c2d1e0f9a8b"
    )


def test_capture_conversation_absent_header_is_none():
    assert _captured_chat_id({}) is None


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "a" * 129,  # too long
        "chat id with spaces",
        "chat#injection",  # key-structure characters
        "chat\nid",
    ],
)
def test_capture_conversation_rejects_malformed_ids(bad):
    # Malformed → treated as absent → the stricter session-keyed fallback.
    assert _captured_chat_id({taint.CHAT_ID_HEADER: bad}) is None


# --- The state machine: gmail → web blocked ---------------------------------------


def test_gmail_then_web_blocked(fake_ddb):
    assert taint.get_taint(CHAT_SCOPE) is None
    assert taint.record_tool_use(CHAT_SCOPE, "gmail_search") == taint.INBOUND_PRIVATE
    assert taint.get_taint(CHAT_SCOPE) == taint.INBOUND_PRIVATE

    # Advisory layer: the opposing class disappears from the injection list.
    allowed, dropped = taint.filter_tools(taint.get_taint(CHAT_SCOPE), ["web_fetch", "gmail_search"])
    assert allowed == ["gmail_search"]
    assert dropped == ["web_fetch"]

    # Authoritative layer: the conditional write refuses the opposing class.
    with pytest.raises(taint.TaintConflictError) as exc_info:
        taint.record_tool_use(CHAT_SCOPE, "web_fetch")
    assert exc_info.value.tainted_class == taint.INBOUND_PRIVATE
    assert exc_info.value.tool == "web_fetch"


def test_web_then_gmail_blocked(fake_ddb):
    assert taint.record_tool_use(CHAT_SCOPE, "web_fetch") == taint.OUTBOUND_FETCH

    allowed, dropped = taint.filter_tools(
        taint.get_taint(CHAT_SCOPE), ["web_fetch", "gmail_search", "gmail_get_message"]
    )
    assert allowed == ["web_fetch"]
    assert dropped == ["gmail_search", "gmail_get_message"]

    for gmail_tool in ("gmail_search", "gmail_get_message"):
        with pytest.raises(taint.TaintConflictError):
            taint.record_tool_use(CHAT_SCOPE, gmail_tool)


def test_same_class_stays_allowed_and_sticky(fake_ddb):
    taint.record_tool_use(CHAT_SCOPE, "gmail_search")
    # Same class, different tool: allowed, taint unchanged.
    assert taint.record_tool_use(CHAT_SCOPE, "gmail_get_message") == taint.INBOUND_PRIVATE
    item = fake_ddb.tables[TAINT_TABLE][CHAT_SCOPE]
    assert item["tainted_class"]["S"] == taint.INBOUND_PRIVATE
    # first_tool records the taint origin, not the latest use.
    assert item["first_tool"]["S"] == "gmail_search"


def test_scopes_are_independent(fake_ddb):
    taint.record_tool_use(CHAT_SCOPE, "gmail_search")
    other_chat = taint.resolve_taint_scope(IDENTITY, "chat-5678")
    # A different conversation of the same user is untainted.
    assert taint.get_taint(other_chat) is None
    assert taint.record_tool_use(other_chat, "web_fetch") == taint.OUTBOUND_FETCH
    # And the session fallback scope is independent of both.
    assert taint.get_taint(SESSION_SCOPE) is None


def test_untainted_scope_filter_passes_everything(fake_ddb):
    allowed, dropped = taint.filter_tools(None, ["web_fetch", "gmail_search"])
    assert allowed == ["web_fetch", "gmail_search"]
    assert dropped == []


def test_filter_never_drops_unclassified_names():
    allowed, dropped = taint.filter_tools(taint.INBOUND_PRIVATE, ["some_future_tool"])
    assert allowed == ["some_future_tool"]
    assert dropped == []


def test_record_refuses_client_tools(fake_ddb):
    with pytest.raises(ValueError):
        taint.record_tool_use(CHAT_SCOPE, "my_calculator")
    assert fake_ddb.tables.get(TAINT_TABLE, {}) == {}


# --- TTL semantics -----------------------------------------------------------------


def test_session_scope_ttl_is_sliding_15_minutes(fake_ddb, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_000_000.0)
    taint.record_tool_use(SESSION_SCOPE, "web_fetch")
    item = fake_ddb.tables[TAINT_TABLE][SESSION_SCOPE]
    assert int(item["ttl"]["N"]) == 1_000_000 + taint.SESSION_TAINT_TTL_SECONDS

    # A later same-class use slides the window forward.
    monkeypatch.setattr(time, "time", lambda: 1_000_600.0)
    taint.record_tool_use(SESSION_SCOPE, "web_fetch")
    item = fake_ddb.tables[TAINT_TABLE][SESSION_SCOPE]
    assert int(item["ttl"]["N"]) == 1_000_600 + taint.SESSION_TAINT_TTL_SECONDS


def test_conversation_scope_ttl_is_long_lived(fake_ddb, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_000_000.0)
    taint.record_tool_use(CHAT_SCOPE, "gmail_search")
    item = fake_ddb.tables[TAINT_TABLE][CHAT_SCOPE]
    assert int(item["ttl"]["N"]) == 1_000_000 + taint.CONVERSATION_TAINT_TTL_DAYS * 86400


# --- Fail closed -------------------------------------------------------------------


def test_unconfigured_table_fails_closed(monkeypatch):
    monkeypatch.setattr(taint, "TAINT_TABLE", "")
    with pytest.raises(taint.TaintStoreError):
        taint.get_taint(CHAT_SCOPE)
    with pytest.raises(taint.TaintStoreError):
        taint.record_tool_use(CHAT_SCOPE, "web_fetch")


def test_store_failure_fails_closed(monkeypatch):
    fake = FakeDynamoDB(fail_with=RuntimeError("dynamodb unavailable"))
    monkeypatch.setattr(taint, "_ddb", lambda: fake)
    monkeypatch.setattr(taint, "TAINT_TABLE", TAINT_TABLE)
    with pytest.raises(taint.TaintStoreError):
        taint.get_taint(CHAT_SCOPE)
    with pytest.raises(taint.TaintStoreError):
        taint.record_tool_use(CHAT_SCOPE, "web_fetch")


# --- Client-tool passthrough untouched ---------------------------------------------


def test_client_tool_passthrough_untouched(fake_ddb):
    """Client-declared tools — even ones that shadow server-tool names —
    reach the Bedrock toolConfig unfiltered, whatever the taint state."""
    from api.models.bedrock import BedrockModel
    from api.schema import ChatRequest

    # Poison the scope as heavily as possible first.
    taint.record_tool_use(CHAT_SCOPE, "gmail_search")

    request = ChatRequest(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {"type": "function", "function": {"name": "web_fetch", "parameters": {}}},
            {"type": "function", "function": {"name": "my_calculator", "parameters": {}}},
        ],
    )
    args = BedrockModel()._parse_request(request)
    names = [t["toolSpec"]["name"] for t in args["toolConfig"]["tools"]]
    assert names == ["web_fetch", "my_calculator"]
