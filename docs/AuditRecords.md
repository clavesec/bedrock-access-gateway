# Tool-Call Audit Records (WORM sink)

Part of the TPAI external-content program, m1 Phase B(ii)
(`plans/external-content/m1-substrate/plan.md` in the main repo). Schema
ratified 2026-07-02 (program decisions **R10** — 7-year WORM retention —
and **R11** — field set incl. full URL, latency, policy-decision reason).

Every server-side tool execution the gateway performs (`web_fetch` from m2,
`gmail_*` from m3) writes exactly one JSON record to an S3 bucket with
Object Lock in **COMPLIANCE mode, 7-year retention**, stood up by the main
repo's `audit-sink-stack` (Product account). The Phase E connector writes a
matching record on its side of the PrivateLink boundary; the pair answers
*"who fetched what, when, and why was it allowed."*

Emitter: [`src/api/audit.py`](../src/api/audit.py) ·
Tests: [`src/tests/test_audit.py`](../src/tests/test_audit.py)

## Schema — `tpai.gateway.tool-call.v1`

One JSON object per record, sorted keys, compact separators. Every field is
always present (nullable fields serialize as `null`, never omitted).

| Field | Type | Semantics |
|---|---|---|
| `schema` | string | Constant `tpai.gateway.tool-call.v1`. Bump the suffix on any field change — 7-year-old records must stay parseable. |
| `ts` | string | ISO-8601 UTC, millisecond precision (e.g. `2026-07-02T15:04:05.123+00:00`). Emission time = tool-call completion. |
| `identity` | string | Pseudonymous HMAC identity from `api/identity.py` (dedicated Product-account key, decision E2). Required — tool execution is only reachable behind `require_identity`. Never the raw email. |
| `tool` | string | Tool name, e.g. `web_fetch`, `gmail_search`, `gmail_get_message`. |
| `target` | string | **The full URL** (web tools — R11 requires full URL, not just host) or the **message id** (gmail tools). |
| `policy_decision` | string | `allow` \| `deny` — whether the gateway let the call proceed. |
| `policy_reason` | string | Why, e.g. `beta-allow-all`, `allowlist-hit`, `allowlist-miss`, `taint-blocked`, `budget-exhausted`, `iteration-cap`, `invalid-url-index`, `mint-refused:<why>`, `budget-store-error`, `executor-error` (R11: the allowlist/policy-decision reason is part of the record). |
| `outcome` | string | `success` \| `denied` \| `error` \| `timeout` — how the call actually ended. Pairings (m2 executor): `deny`+`denied` = clean policy deny; `deny`+`error` = **fail-closed control failure** (a store/mint/transport control could not run, so the call was refused — `policy_reason` names the control, e.g. `taint-store-error`); `allow`+`error`/`timeout` = execution failed after policy allowed the fetch. |
| `bytes` | int \| null | Content bytes returned to the model. `null` when nothing was returned (denied/error paths). |
| `latency_ms` | int | Wall-clock duration of the tool execution. |
| `conversation_id` | string \| null | `X-OpenWebUI-Chat-Id` when present (arrives with the m1 Phase D cherry-pick). `null` for background generations and api-proxy callers — those use the session-keyed taint fallback (D9). |

Example:

```json
{"bytes":20480,"conversation_id":"chat-1234","identity":"9f2c…","latency_ms":812,"outcome":"success","policy_decision":"allow","policy_reason":"beta-allow-all","schema":"tpai.gateway.tool-call.v1","target":"https://example.com/reports/q3","tool":"web_fetch","ts":"2026-07-02T15:04:05.123+00:00"}
```

## Storage layout

```
s3://tpai-audit-<env>-<account>/
  gateway/YYYY/MM/DD/<YYYYMMDDTHHMMSS>-<uuid>.json   ← this emitter
  connector/…                                        ← Phase E connector records
```

Date-partitioned for review tooling (Athena/S3 Select); the uuid suffix
makes concurrent writers collision-free. Object Lock retention is the
bucket default — the emitter never calls `PutObjectRetention` and the task
role is not granted it. Puts carry an explicit `ChecksumAlgorithm=CRC32`
(Object Lock requires a checksum header; do not rely on the botocore
default).

## Operating contract

- **Fail closed.** `emit_audit_record` raises `AuditEmitError` on any
  write failure *including an unconfigured bucket*. Tool executors must
  fail the tool call — a retrieval that cannot be audited must not happen
  (ADR-1: "every retrieval is logged"). No fire-and-forget, no
  buffer-and-drop.
- **Records never touch logs.** Full URLs and identities exist only in the
  WORM trail; CloudWatch keeps the metadata-only access line (E3, which
  carries the tool *name* only). Failure logs carry the exception class
  and S3 key, never record fields — `test_audit.py` asserts this.
- **Configuration:** `TPAI_AUDIT_BUCKET` only (injected by the CDK wiring).
  The `gateway/` prefix is a code constant, deliberately not configurable:
  the task role's IAM grant is pinned to exactly `gateway/*`, so any other
  prefix is an AccessDenied outage. Bounded latency: the S3 client is
  capped at 2 attempts × (3s connect + 5s read) — worst case ~17s before
  the tool call fails closed.
- **m2 wiring requirement:** route every tool execution through a single
  audited choke point that emits on *all* branches — success, denied,
  error, and timeout. Per-branch emit calls scattered through the executor
  is how a branch gets missed and a retrieval happens unaudited; the m3 G4
  audit-pair review assumes gateway/connector records exist for every
  fetch.
- **Dark today:** no server-side tool executes until m2 Phase 0 ships and
  its flag family is enabled; this module has no production call sites yet.
