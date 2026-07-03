"""WORM audit-record emitter tests (external-content m1 Phase B(ii), R10/R11).

Three properties:

1. **Schema** — a synthetic record serializes to exactly the ratified R11
   field set (schema-versioned, date-partitioned key, deterministic JSON).
2. **Fail closed** — any failure to durably write (S3 error, missing
   bucket config) raises ``AuditEmitError``; schema violations raise
   ``ValueError``. There is no drop path.
3. **No leakage** — the record (in particular the full URL target) never
   appears in a log line, success or failure: CloudWatch stays
   metadata-only (E3); full URLs belong only in the WORM trail.
"""

import json
import logging

import pytest

import api.audit as audit
from tests.conftest import expected_hmac

IDENTITY = expected_hmac("owui-email", "alice@example.com")
TARGET = "https://example.com/reports/q3?token=SENTINEL-URL-SECRET"

SYNTHETIC = dict(
    identity=IDENTITY,
    tool="web_fetch",
    target=TARGET,
    policy_decision="allow",
    policy_reason="beta-allow-all",
    outcome="success",
    bytes_returned=20480,
    latency_ms=812,
    conversation_id="chat-1234",
)


class FakeS3:
    def __init__(self, fail_with: Exception | None = None):
        self.calls = []
        self.fail_with = fail_with

    def put_object(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_with is not None:
            raise self.fail_with


@pytest.fixture
def fake_s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(audit, "_s3", lambda: fake)
    monkeypatch.setattr(audit, "AUDIT_BUCKET", "tpai-audit-test-000000000000")
    return fake


# --- Schema (the R11 contract) --------------------------------------------------


def test_synthetic_record_matches_ratified_schema(fake_s3):
    key = audit.emit_audit_record(**SYNTHETIC)

    assert len(fake_s3.calls) == 1
    call = fake_s3.calls[0]
    assert call["Bucket"] == "tpai-audit-test-000000000000"
    assert call["Key"] == key
    assert call["ContentType"] == "application/json"
    # Object Lock buckets require a checksum header; it must be explicit,
    # not inherited from a botocore-version default.
    assert call["ChecksumAlgorithm"] == "CRC32"

    record = json.loads(call["Body"])
    ts = record.pop("ts")
    assert record == {
        "schema": "tpai.gateway.tool-call.v1",
        "identity": IDENTITY,
        "tool": "web_fetch",
        "target": TARGET,
        "policy_decision": "allow",
        "policy_reason": "beta-allow-all",
        "outcome": "success",
        "bytes": 20480,
        "latency_ms": 812,
        "conversation_id": "chat-1234",
    }
    # ISO-8601 UTC with millisecond precision, e.g. 2026-07-02T15:04:05.123+00:00
    assert ts.startswith("20") and "T" in ts and "+00:00" in ts


def test_key_is_date_partitioned_under_prefix(fake_s3):
    key = audit.emit_audit_record(**SYNTHETIC)
    prefix, y, m, d, leaf = key.split("/")
    assert prefix == "gateway"
    assert len(y) == 4 and len(m) == 2 and len(d) == 2
    assert leaf.endswith(".json")


def test_concurrent_writers_never_collide_on_key(fake_s3):
    keys = {audit.emit_audit_record(**SYNTHETIC) for _ in range(50)}
    assert len(keys) == 50


def test_json_is_deterministic_and_compact(fake_s3):
    audit.emit_audit_record(**SYNTHETIC)
    body = fake_s3.calls[0]["Body"].decode()
    assert body == json.dumps(json.loads(body), sort_keys=True, separators=(",", ":"))


def test_optional_fields_serialize_as_null(fake_s3):
    """bytes and conversation_id are nullable (denied calls return no bytes;
    api-proxy callers have no chat id) — the fields still exist."""
    audit.emit_audit_record(
        **{
            **SYNTHETIC,
            "bytes_returned": None,
            "conversation_id": None,
            "outcome": "denied",
            "policy_decision": "deny",
            "policy_reason": "budget-exhausted",
        }
    )
    record = json.loads(fake_s3.calls[0]["Body"])
    assert record["bytes"] is None
    assert record["conversation_id"] is None
    assert record["policy_decision"] == "deny"


# --- Fail closed ----------------------------------------------------------------


def test_missing_identity_is_a_schema_violation(fake_s3):
    with pytest.raises(ValueError, match="identity"):
        audit.emit_audit_record(**{**SYNTHETIC, "identity": ""})
    assert fake_s3.calls == []


@pytest.mark.parametrize("field", ["tool", "target"])
def test_missing_tool_or_target_is_a_schema_violation(fake_s3, field):
    with pytest.raises(ValueError):
        audit.emit_audit_record(**{**SYNTHETIC, field: ""})
    assert fake_s3.calls == []


def test_invalid_enums_rejected(fake_s3):
    with pytest.raises(ValueError, match="policy_decision"):
        audit.emit_audit_record(**{**SYNTHETIC, "policy_decision": "maybe"})
    with pytest.raises(ValueError, match="outcome"):
        audit.emit_audit_record(**{**SYNTHETIC, "outcome": "partial"})
    assert fake_s3.calls == []


def test_unconfigured_bucket_refuses(monkeypatch):
    monkeypatch.setattr(audit, "AUDIT_BUCKET", "")
    with pytest.raises(audit.AuditEmitError, match="TPAI_AUDIT_BUCKET"):
        audit.emit_audit_record(**SYNTHETIC)


def test_s3_failure_raises_audit_emit_error(monkeypatch):
    fake = FakeS3(fail_with=ConnectionError("endpoint unreachable"))
    monkeypatch.setattr(audit, "_s3", lambda: fake)
    monkeypatch.setattr(audit, "AUDIT_BUCKET", "tpai-audit-test-000000000000")
    with pytest.raises(audit.AuditEmitError) as excinfo:
        audit.emit_audit_record(**SYNTHETIC)
    assert isinstance(excinfo.value.__cause__, ConnectionError)


# --- No leakage into logs (E3 stays intact) -------------------------------------


def assert_no_record_fields_logged(caplog):
    for r in caplog.records:
        rendered = r.getMessage()
        assert TARGET not in rendered, f"full URL leaked into log from {r.name}"
        assert "SENTINEL-URL-SECRET" not in rendered
        assert IDENTITY not in rendered
        assert "chat-1234" not in rendered


def test_success_logs_nothing(fake_s3, caplog):
    caplog.set_level(logging.DEBUG)
    audit.emit_audit_record(**SYNTHETIC)
    assert_no_record_fields_logged(caplog)


def test_failure_log_carries_no_record_fields(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    fake = FakeS3(fail_with=ConnectionError("endpoint unreachable"))
    monkeypatch.setattr(audit, "_s3", lambda: fake)
    monkeypatch.setattr(audit, "AUDIT_BUCKET", "tpai-audit-test-000000000000")
    with pytest.raises(audit.AuditEmitError):
        audit.emit_audit_record(**SYNTHETIC)
    # The failure line exists (operators must see audit-write failures)…
    assert any("audit record write failed" in r.getMessage() for r in caplog.records)
    # …but never the record: URL/identity live only in the WORM trail.
    assert_no_record_fields_logged(caplog)


def test_exception_chain_carries_no_record_fields(monkeypatch):
    """AuditEmitError propagates to the tool loop, whose error handling may
    log it — the exception text itself must stay record-free."""
    fake = FakeS3(fail_with=ConnectionError("endpoint unreachable"))
    monkeypatch.setattr(audit, "_s3", lambda: fake)
    monkeypatch.setattr(audit, "AUDIT_BUCKET", "tpai-audit-test-000000000000")
    with pytest.raises(audit.AuditEmitError) as excinfo:
        audit.emit_audit_record(**SYNTHETIC)
    assert TARGET not in str(excinfo.value)
    assert IDENTITY not in str(excinfo.value)
