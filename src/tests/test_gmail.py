"""Tests for api.tools.gmail — the m3 G3 tool pair.

gmail_search must consume the S15 metadata layer exactly per the
consumption contract (DEK from metadata-key verified against the locally
derived read path, AES-GCM AAD binding, 403 = authoritative deny, DEK
cache ≤ minutes) and gmail_get_message must map the connector's sanitized
get surface onto the shared tool taxonomy. E3: failures never log queries,
message ids, or header values.
"""

import base64
import json
import logging
import time

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from api import mint, setting
from api.tools import base, executor, gmail, web_fetch

IDENTITY = "a1b2c3d4" * 8
CONNECTOR_URL = "https://vpce-test.example"
BUCKET = "tpai-gmail-metadata-tpaitest-405894864934"
DEK = b"\x11" * 32

CTX = executor.ServerToolContext(
    identity=IDENTITY,
    binding=mint.BINDING_OWUI_SESSION,
    subject_id="user-1",
    chat_id="chat-1",
)

RECORDS = [
    {
        "message_id": "msg-alpha",
        "thread_id": "t-1",
        "from": "Alice <alice@example.com>",
        "to": "me@example.com",
        "cc": "",
        "date": "Mon, 1 Jul 2026 09:00:00 -0400",
        "subject": "Quarterly budget review",
    },
    {
        "message_id": "msg-beta",
        "thread_id": "t-2",
        "from": "Bob <bob@example.com>",
        "to": "me@example.com",
        "cc": "carol@example.com",
        "date": "Tue, 2 Jul 2026 10:00:00 -0400",
        "subject": "Lunch on Friday?",
    },
]


def encrypted_index(dek=DEK, identity=IDENTITY, records=RECORDS, truncated=False):
    document = {
        "schema": gmail.INDEX_SCHEMA,
        "identity": identity,
        "synced_at": "2026-07-05T12:00:00Z",
        "truncated": truncated,
        "message_count": len(records),
        "messages": records,
    }
    plaintext = json.dumps(document).encode("utf-8")
    nonce = b"\x00" * 12
    aad = f"{gmail.INDEX_SCHEMA}|{identity}".encode("utf-8")
    return nonce + AESGCM(dek).encrypt(nonce, plaintext, aad)


class FakeResponse:
    def __init__(self, status_code, body, raise_mid_body=None):
        self.status_code = status_code
        self._body = json.dumps(body).encode("utf-8")
        self._raise_mid_body = raise_mid_body

    def iter_content(self, chunk_size=65536):
        if self._raise_mid_body is not None:
            raise self._raise_mid_body
        yield self._body

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Stands in for the pinned connector session."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.responses: dict[str, list[FakeResponse]] = {}

    def queue(self, path, response):
        self.responses.setdefault(path, []).append(response)

    def request(self, method, url, json=None, headers=None, timeout=None, stream=None, allow_redirects=None):
        path = url.replace(CONNECTOR_URL, "")
        self.calls.append((method, path, json))
        queued = self.responses.get(path)
        if not queued:
            raise AssertionError(f"unexpected connector call: {method} {path}")
        return queued.pop(0)


class FakeS3:
    def __init__(self, objects=None):
        self.objects = objects if objects is not None else {}
        self.gets: list[tuple[str, str]] = []

    def get_object(self, Bucket, Key):
        self.gets.append((Bucket, Key))
        if Key not in self.objects:
            error = type("ClientError", (Exception,), {})()
            error.response = {"Error": {"Code": "NoSuchKey"}}
            raise error

        class Body:
            def __init__(self, blob):
                self.blob = blob

            def read(self, amt=None):
                return self.blob if amt is None else self.blob[:amt]

        return {"Body": Body(self.objects[Key])}


def metadata_key_body(dek=DEK, bucket=BUCKET, identity=IDENTITY):
    return {
        "schema": gmail.METADATA_KEY_RESPONSE_SCHEMA,
        "key_b64": base64.b64encode(dek).decode("ascii"),
        "bucket": bucket,
        "object_key": gmail.INDEX_OBJECT_KEY_FMT.format(identity=identity),
    }


@pytest.fixture
def session(monkeypatch):
    fake = FakeSession()
    monkeypatch.setattr(web_fetch, "connector_session", lambda: fake)
    monkeypatch.setattr(setting, "TPAI_CONNECTOR_URL", CONNECTOR_URL)
    return fake


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3({gmail.INDEX_OBJECT_KEY_FMT.format(identity=IDENTITY): encrypted_index()})
    monkeypatch.setattr(gmail, "_s3", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def gmail_env(monkeypatch):
    monkeypatch.setattr(setting, "TPAI_GMAIL_METADATA_BUCKET", BUCKET)
    gmail._dek_cache.clear()
    yield
    gmail._dek_cache.clear()


# ------------------------------------------------------- target resolution


def test_resolve_search_target_accepts_query():
    assert gmail.resolve_search_target(None, {"query": "budget"}) == "query:budget"
    assert gmail.resolve_search_target(None, {"query": ""}) == "query:"


def test_search_tool_spec_does_not_advertise_max_results():
    # The ToolHandler contract funnels resolve->execute through one target
    # string, so a per-call max_results could not survive the hop — the spec
    # must not advertise a knob the executor structurally drops.
    props = gmail.build_search_tool_spec([])["toolSpec"]["inputSchema"]["json"]["properties"]
    assert "max_results" not in props
    assert set(props) == {"query"}


@pytest.mark.parametrize(
    "tool_input",
    [
        "not-a-dict",
        {},
        {"query": 42},
        {"query": "x" * (gmail.MAX_QUERY_CHARS + 1)},
    ],
)
def test_resolve_search_target_rejects_malformed_input(tool_input):
    assert gmail.resolve_search_target(None, tool_input) is None


def test_resolve_get_target_enforces_the_connector_id_shape():
    assert gmail.resolve_get_target(None, {"message_id": "abc_DEF-123"}) == "abc_DEF-123"
    for bad in [None, {}, {"message_id": ""}, {"message_id": "x" * 129},
                {"message_id": "has space"}, {"message_id": 5}, "raw"]:
        assert gmail.resolve_get_target(None, bad if isinstance(bad, (dict, str)) else {}) is None


def test_tool_specs_pin_names_and_reject_extra_properties():
    for spec, name in (
        (gmail.build_search_tool_spec([]), gmail.SEARCH_TOOL_NAME),
        (gmail.build_get_tool_spec([]), gmail.GET_TOOL_NAME),
    ):
        tool_spec = spec["toolSpec"]
        assert tool_spec["name"] == name
        assert tool_spec["inputSchema"]["json"]["additionalProperties"] is False
        assert "untrusted external data" in tool_spec["description"]


# ------------------------------------------------------------ gmail_search


def test_search_decrypts_filters_and_fences_nothing_itself(session, s3):
    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body()))
    result = gmail.execute_search(CTX, "query:budget", "jwt")
    document = json.loads(result.text)
    assert [m["message_id"] for m in document["messages"]] == ["msg-alpha"]
    assert document["result_count"] == 1
    assert document["truncated"] is False
    assert result.source == "the user's Gmail inbox index"
    # R4: the typed record carries exactly the seven index fields.
    assert set(document["messages"][0]) == {
        "message_id", "thread_id", "from", "to", "cc", "date", "subject",
    }


def test_search_empty_query_returns_most_recent(session, s3):
    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body()))
    document = json.loads(gmail.execute_search(CTX, "query:", "jwt").text)
    assert document["result_count"] == 2


def test_search_match_is_case_insensitive_across_header_fields(session, s3):
    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body()))
    document = json.loads(gmail.execute_search(CTX, "query:CAROL friday", "jwt").text)
    assert [m["message_id"] for m in document["messages"]] == ["msg-beta"]


def test_search_result_cap_sets_truncated(session, s3, monkeypatch):
    monkeypatch.setattr(setting, "GMAIL_SEARCH_MAX_RESULTS", 1)
    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body()))
    document = json.loads(gmail.execute_search(CTX, "query:", "jwt").text)
    assert document["result_count"] == 1
    assert document["truncated"] is True


def test_search_not_connected_is_a_clean_deny_and_drops_the_dek(session, s3):
    gmail._dek_cache[IDENTITY] = (DEK, time.monotonic() + 999)
    # Cache is bypassed only after expiry — simulate the post-TTL refresh.
    gmail._dek_cache[IDENTITY] = (DEK, time.monotonic() - 1)
    session.queue("/v1/gmail/metadata-key", FakeResponse(403, {"reason": "not-connected"}))
    with pytest.raises(gmail.GmailDenied) as exc_info:
        gmail.execute_search(CTX, "query:budget", "jwt")
    assert exc_info.value.reason == "not-connected"
    assert IDENTITY not in gmail._dek_cache
    assert "Settings → Connectors" in gmail.deny_text("not-connected")


def test_search_unsynced_index_returns_empty_with_note(session, s3):
    s3.objects.clear()
    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body()))
    result = gmail.execute_search(CTX, "query:budget", "jwt")
    document = json.loads(result.text)
    assert document["messages"] == []
    assert "not synced yet" in document["note"]


def test_search_verifies_the_connector_named_read_path(session, s3):
    """A compromised connector must not steer the gateway's S3 read."""
    session.queue(
        "/v1/gmail/metadata-key",
        FakeResponse(200, metadata_key_body(bucket="attacker-bucket")),
    )
    with pytest.raises(gmail.GmailToolError, match="unexpected bucket"):
        gmail.execute_search(CTX, "query:budget", "jwt")
    assert s3.gets == []

    session.queue(
        "/v1/gmail/metadata-key",
        FakeResponse(200, {**metadata_key_body(), "object_key": "metadata/other/index.v1.enc"}),
    )
    with pytest.raises(gmail.GmailToolError, match="unexpected object key"):
        gmail.execute_search(CTX, "query:budget", "jwt")
    assert s3.gets == []


def test_search_rejects_a_non_256_bit_key(session, s3):
    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body(dek=b"\x11" * 16)))
    with pytest.raises(gmail.GmailToolError, match="256-bit"):
        gmail.execute_search(CTX, "query:budget", "jwt")


def test_search_401_maps_to_auth_error_for_the_remint(session, s3):
    session.queue("/v1/gmail/metadata-key", FakeResponse(401, {"detail": "invalid_token:expired"}))
    with pytest.raises(gmail.GmailAuthError):
        gmail.execute_search(CTX, "query:budget", "jwt")


def test_dek_cache_serves_within_ttl_and_expires(session, s3, monkeypatch):
    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body()))
    gmail.execute_search(CTX, "query:budget", "jwt")
    # Second search: no queued metadata-key response — a connector call
    # would raise, so success proves the cache served.
    gmail.execute_search(CTX, "query:budget", "jwt")
    assert len([c for c in session.calls if c[1] == "/v1/gmail/metadata-key"]) == 1

    monkeypatch.setattr(setting, "GMAIL_DEK_CACHE_TTL_S", 0)
    gmail._dek_cache.clear()
    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body()))
    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body()))
    gmail.execute_search(CTX, "query:budget", "jwt")
    gmail.execute_search(CTX, "query:budget", "jwt")
    assert len([c for c in session.calls if c[1] == "/v1/gmail/metadata-key"]) == 3


def test_stale_cached_dek_retries_once_with_a_fresh_key(session, s3):
    """After a broken->reconnect cycle the old DEK is shredded; a cached
    stale DEK must not fail the search when a fresh fetch can decrypt."""
    gmail._dek_cache[IDENTITY] = (b"\x99" * 32, time.monotonic() + 999)
    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body()))
    document = json.loads(gmail.execute_search(CTX, "query:budget", "jwt").text)
    assert document["result_count"] == 1


def test_wrong_identity_index_fails_authentication(session, s3):
    """The AAD binds ciphertext to the identity — a blob copied from
    another identity's path must fail even under the right DEK."""
    other = "d4" * 32
    s3.objects[gmail.INDEX_OBJECT_KEY_FMT.format(identity=IDENTITY)] = encrypted_index(
        identity=other
    )
    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body()))
    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body()))
    with pytest.raises(gmail.GmailToolError):
        gmail.execute_search(CTX, "query:budget", "jwt")


def test_search_without_bucket_config_fails_closed(session, s3, monkeypatch):
    monkeypatch.setattr(setting, "TPAI_GMAIL_METADATA_BUCKET", "")
    with pytest.raises(gmail.GmailToolError, match="TPAI_GMAIL_METADATA_BUCKET"):
        gmail.execute_search(CTX, "query:budget", "jwt")
    assert session.calls == []


def test_index_over_byte_cap_fails_closed(session, s3, monkeypatch):
    monkeypatch.setattr(setting, "GMAIL_INDEX_MAX_BYTES", 8)
    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body()))
    with pytest.raises(gmail.GmailToolError, match="byte cap"):
        gmail.execute_search(CTX, "query:budget", "jwt")


# ------------------------------------------------------- gmail_get_message


def message_body(**overrides):
    message = {
        "message_id": "msg-alpha",
        "thread_id": "t-1",
        "from": "Alice <alice@example.com>",
        "to": "me@example.com",
        "cc": "",
        "date": "Mon, 1 Jul 2026 09:00:00 -0400",
        "subject": "Quarterly budget review",
        "body_text": "Numbers attached. <<<injected>>>",
        "body_bytes": 33,
        "truncated": False,
        "attachments": [{"filename": "q.pdf", "mime_type": "application/pdf", "size_bytes": 100}],
    }
    message.update(overrides)
    return {"schema": gmail.MESSAGE_RESPONSE_SCHEMA, "message": message}


def test_get_message_returns_typed_output(session):
    session.queue("/v1/gmail/get", FakeResponse(200, message_body()))
    result = gmail.execute_get(CTX, "msg-alpha", "jwt")
    document = json.loads(result.text)
    assert document["subject"] == "Quarterly budget review"
    assert document["body_text"].startswith("Numbers attached.")
    assert result.source == "the user's Gmail message msg-alpha"
    method, path, payload = session.calls[0]
    assert (method, path) == ("POST", "/v1/gmail/get")
    assert payload == {"schema": gmail.GET_REQUEST_SCHEMA, "message_id": "msg-alpha"}


def test_get_message_char_cap_truncates_to_valid_json(session, monkeypatch):
    # A cap below headers+body must shrink body_text and re-serialize — NEVER
    # slice the JSON string mid-structure (that hands the model malformed
    # JSON). Cap chosen above the header envelope, below headers+body.
    monkeypatch.setattr(setting, "GMAIL_MAX_CHARS", 1000)
    session.queue(
        "/v1/gmail/get",
        FakeResponse(200, message_body(body_text="x" * 5000)),
    )
    result = gmail.execute_get(CTX, "msg-alpha", "jwt")
    assert len(result.text) <= 1000
    assert result.truncated is True
    # Still parseable JSON with the headers intact and the body shrunk.
    parsed = json.loads(result.text)
    assert parsed["subject"] == "Quarterly budget review"
    assert 0 < len(parsed["body_text"]) < 5000


def test_search_over_char_cap_drops_records_keeping_valid_json(session, s3, monkeypatch):
    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body()))
    # Cap between the one-record and two-record document sizes: the second
    # record is dropped whole, and the result stays valid JSON.
    monkeypatch.setattr(setting, "GMAIL_MAX_CHARS", 300)
    result = gmail.execute_search(CTX, "query:", "jwt")
    parsed = json.loads(result.text)  # never mid-structure garbage
    assert parsed["truncated"] is True
    assert parsed["result_count"] == len(parsed["messages"])
    assert len(parsed["messages"]) < 2  # at least one record dropped to fit


def test_get_message_reconnect_required_is_a_clean_deny(session):
    session.queue("/v1/gmail/get", FakeResponse(403, {"reason": "reconnect-required"}))
    with pytest.raises(gmail.GmailDenied) as exc_info:
        gmail.execute_get(CTX, "msg-alpha", "jwt")
    assert exc_info.value.reason == "reconnect-required"
    assert "re-authorized" in gmail.deny_text("reconnect-required")


def test_get_message_5xx_maps_to_execution_error(session):
    session.queue("/v1/gmail/get", FakeResponse(502, {"detail": "quarantine-failed"}))
    with pytest.raises(gmail.GmailToolError):
        gmail.execute_get(CTX, "msg-alpha", "jwt")


def test_get_message_mid_body_transport_error_maps_to_tool_error(session):
    # A raw requests exception during the streamed body read must be mapped
    # to GmailToolError (a bare requests exception would bypass the executor's
    # audit branch — the guarantee web_fetch documents, shared via base).
    import requests

    session.queue(
        "/v1/gmail/get",
        FakeResponse(200, {}, raise_mid_body=requests.exceptions.ConnectionError("reset")),
    )
    with pytest.raises(gmail.GmailToolError):
        gmail.execute_get(CTX, "msg-alpha", "jwt")


def test_get_message_mid_body_timeout_maps_to_timeout_outcome(session):
    import requests

    session.queue(
        "/v1/gmail/get",
        FakeResponse(200, {}, raise_mid_body=requests.exceptions.Timeout("slow")),
    )
    with pytest.raises(gmail.GmailToolError) as exc_info:
        gmail.execute_get(CTX, "msg-alpha", "jwt")
    assert exc_info.value.outcome == "timeout"


def test_get_message_unexpected_schema_fails_closed(session):
    session.queue("/v1/gmail/get", FakeResponse(200, {"schema": "wrong", "message": {}}))
    with pytest.raises(gmail.GmailToolError, match="unexpected schema"):
        gmail.execute_get(CTX, "msg-alpha", "jwt")


# ------------------------------------------------- executor integration/E3


def test_executor_success_path_fences_gmail_output(session, s3, monkeypatch):
    from api import audit, budget, taint

    monkeypatch.setattr(setting, "ENABLE_GMAIL_TOOLS", True)
    monkeypatch.setattr(taint, "get_taint", lambda scope: None)
    monkeypatch.setattr(budget, "check_and_consume", lambda identity, scope: None)
    monkeypatch.setattr(taint, "record_tool_use", lambda scope, name: taint.INBOUND_PRIVATE)
    monkeypatch.setattr(
        mint,
        "get_connector_token",
        lambda identity, binding, subject_id: mint.MintedToken(
            token="jwt", expires_at=2**31, subject_id=subject_id
        ),
    )
    records = []
    monkeypatch.setattr(audit, "emit_audit_record", lambda **kwargs: records.append(kwargs))

    from api.schema import ChatRequest

    request = ChatRequest(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[{"role": "user", "content": "any recent email about budget?"}],
    )
    plan = executor.plan_server_tools(request, CTX)
    assert plan is not None
    assert plan.injected == [gmail.SEARCH_TOOL_NAME, gmail.GET_TOOL_NAME]

    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body()))
    outcome = executor.run_server_tool(plan, CTX, "gmail_search", {"query": "budget"})
    assert outcome.ok
    assert "TPAI-EXTERNAL-CONTENT" in outcome.result_text
    assert records[0]["target"] == "query:budget"
    assert records[0]["policy_reason"] == "own-mailbox"
    assert records[0]["tool"] == "gmail_search"

    # Fence-token neutralization (R7) on the get path.
    session.queue("/v1/gmail/get", FakeResponse(200, message_body()))
    outcome = executor.run_server_tool(plan, CTX, "gmail_get_message", {"message_id": "msg-alpha"})
    assert outcome.ok
    assert "<<<injected>>>" not in outcome.result_text


def test_e3_failures_never_log_the_query_or_message_id(session, s3, caplog):
    caplog.set_level(logging.DEBUG)
    s3.objects.clear()

    def broken_get(Bucket, Key):
        raise RuntimeError("s3 down")

    s3.get_object = broken_get
    session.queue("/v1/gmail/metadata-key", FakeResponse(200, metadata_key_body()))
    with pytest.raises(gmail.GmailToolError):
        gmail.execute_search(CTX, "query:secret-medical-topic", "jwt")
    assert "secret-medical-topic" not in caplog.text
    assert "jwt" not in caplog.text
