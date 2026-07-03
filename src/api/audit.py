"""Per-tool-call WORM audit records (TPAI external-content program,
m1 Phase B(ii); decisions R10/R11).

Every server-side tool execution (web_fetch in m2, gmail_* in m3) produces
exactly one audit record answering "who fetched what, when, and why was it
allowed". Records are written as individual JSON objects to an S3 bucket
with Object Lock in COMPLIANCE mode (7-year retention, R10) — the bucket is
stood up by the main repo's audit-sink stack; this module only writes.

Schema v1 (ratified R11 — full documentation in ``docs/AuditRecords.md``):

    schema           "tpai.gateway.tool-call.v1"
    ts               ISO-8601 UTC, millisecond precision
    identity         pseudonymous HMAC identity (api.identity) — required
    tool             tool name, e.g. "web_fetch", "gmail_search"
    target           the full URL (web tools) or message id (gmail tools)
    policy_decision  "allow" | "deny"
    policy_reason    why, e.g. "beta-allow-all", "budget-exhausted"
    outcome          "success" | "denied" | "error" | "timeout"
    bytes            content bytes returned to the model, None if none
    latency_ms       wall-clock duration of the tool execution
    conversation_id  X-OpenWebUI-Chat-Id when present, else None

Operating contract:

- **Fail closed.** ``emit_audit_record`` raises ``AuditEmitError`` on any
  failure, including an unconfigured bucket. Tool executors must treat that
  as a failed tool call — a retrieval that cannot be audited must not
  happen (ADR-1: "every retrieval is logged"). There is no fire-and-forget
  mode and no buffer-and-drop fallback.
- **The record goes to S3, never to logs.** Full URLs belong in the WORM
  trail, not in CloudWatch where operators read (the E3 metadata line
  deliberately carries the tool name only). On failure this module logs the
  exception class and the S3 key — never any record field.
- **Synchronous.** boto3 is blocking; async callers (the Converse tool
  loop) must wrap calls in ``starlette.concurrency.run_in_threadpool`` the
  same way ``models/bedrock.py`` wraps Bedrock calls.

This path is dark today: the bucket wiring ships with m1 Phase B(ii) but no
server-side tool executes until m2 Phase 0 lands and its flag is enabled.
"""

import datetime
import json
import logging
import os
import threading
import uuid

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

# Injected by the audit-sink CDK wiring (bedrock-gateway-stack). Unset means
# this deployment cannot audit — emit_audit_record refuses rather than drops.
AUDIT_BUCKET = os.environ.get("TPAI_AUDIT_BUCKET", "")
# Deliberately a constant, NOT an env var: the task role's IAM grant is
# pinned to exactly this prefix (gateway/*), so any other value turns every
# put into AccessDenied — and fail-closed turns that into a tool-call
# outage. Changing the prefix requires a CDK change anyway; keep the two
# pinned together. (The Phase E connector writes under connector/.)
AUDIT_PREFIX = "gateway/"

SCHEMA = "tpai.gateway.tool-call.v1"

VALID_POLICY_DECISIONS = ("allow", "deny")
VALID_OUTCOMES = ("success", "denied", "error", "timeout")

# The audit put sits on the tool-call critical path (fail closed) and each
# in-flight call pins a threadpool thread, so the worst case must stay small:
# 2 attempts x (3s connect + 5s read) + backoff ≈ ~17s ceiling for a <1KB
# put to a same-region gateway endpoint — vs ~50s with boto3's defaults.
_S3_CONFIG = Config(
    connect_timeout=3,
    read_timeout=5,
    retries={"max_attempts": 2, "mode": "standard"},
)

_s3_lock = threading.Lock()
_s3_client = None


class AuditEmitError(RuntimeError):
    """The audit record could not be durably written. Callers must fail the
    tool call — never execute-and-drop."""


def _s3():
    """Lazy singleton S3 client (unlike bedrock.py's import-time clients, so
    importing the app never resolves AWS credentials for a dark code path)."""
    global _s3_client
    with _s3_lock:
        if _s3_client is None:
            _s3_client = boto3.client(
                "s3",
                region_name=os.environ.get("AWS_REGION"),
                config=_S3_CONFIG,
            )
        return _s3_client


def emit_audit_record(
    *,
    identity: str,
    tool: str,
    target: str,
    policy_decision: str,
    policy_reason: str,
    outcome: str,
    bytes_returned: int | None,
    latency_ms: int,
    conversation_id: str | None,
) -> str:
    """Write one tool-call audit record to the WORM sink; return its S3 key.

    Raises ``AuditEmitError`` if the record cannot be durably written (S3
    failure or no bucket configured) and ``ValueError`` for records that
    violate the schema — both mean the tool call must fail.
    """
    if not identity:
        # Tool execution is only reachable behind require_identity; a record
        # without an accountable identity is a bug, not a degraded mode.
        raise ValueError("audit record requires a non-empty HMAC identity")
    if not tool or not target:
        raise ValueError("audit record requires non-empty tool and target")
    if policy_decision not in VALID_POLICY_DECISIONS:
        raise ValueError(f"policy_decision must be one of {VALID_POLICY_DECISIONS}")
    if outcome not in VALID_OUTCOMES:
        raise ValueError(f"outcome must be one of {VALID_OUTCOMES}")
    if not AUDIT_BUCKET:
        raise AuditEmitError("TPAI_AUDIT_BUCKET is not configured - refusing to execute an unauditable tool call")

    ts = datetime.datetime.now(datetime.timezone.utc)
    record = {
        "schema": SCHEMA,
        "ts": ts.isoformat(timespec="milliseconds"),
        "identity": identity,
        "tool": tool,
        "target": target,
        "policy_decision": policy_decision,
        "policy_reason": policy_reason,
        "outcome": outcome,
        "bytes": bytes_returned,
        "latency_ms": latency_ms,
        "conversation_id": conversation_id,
    }
    # Date-partitioned key so 7 years of records stay listable/queryable;
    # uuid suffix makes concurrent writers collision-free. Object Lock
    # COMPLIANCE retention is the bucket default — no per-object retention
    # call (and the task role has no s3:PutObjectRetention to change it).
    key = f"{AUDIT_PREFIX}{ts:%Y/%m/%d}/{ts:%Y%m%dT%H%M%S}-{uuid.uuid4().hex}.json"
    body = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        _s3().put_object(
            Bucket=AUDIT_BUCKET,
            Key=key,
            Body=body,
            ContentType="application/json",
            # Object Lock buckets require a Content-MD5/x-amz-checksum header.
            # botocore >= 1.36 would add one by default; pass it explicitly so
            # the invariant doesn't live in a transitive SDK default.
            ChecksumAlgorithm="CRC32",
        )
    except Exception as exc:
        # Never log record fields here: the full URL belongs only in the
        # WORM trail (E3 keeps CloudWatch metadata-only).
        logger.error("audit record write failed (%s): %s", type(exc).__name__, key)
        raise AuditEmitError(f"failed to write audit record {key}") from exc
    return key
