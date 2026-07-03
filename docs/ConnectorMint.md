# Connector-JWT mint path (`api.mint`)

TPAI external-content program, **m1 Phase E(i)** — decision **D8** with the
api-proxy analogue **R12**. Status: **dark**. No server-side tool executes
until m2 Phase 0, so nothing calls `get_connector_token` today; the module
fails closed (`MintError`) when `TPAI_CONNECTOR_MINT_FUNCTION_ARN` is unset.

## What it does

Server-side tool calls (m2 `web_fetch`, m3 `gmail_*`) authenticate to the
external-content connector with a **short-TTL, per-user, audience-scoped
JWT** (`aud=tpai-connector`, `iss=tpai-auth`, RS256). The gateway does not
sign tokens. It invokes a dedicated **mint Lambda** in the auth-server stack
(`tpai-connector-mint-<env>`) over a **Lambda interface VPC endpoint** whose
policy is pinned to exactly that function ARN, using plain **IAM SigV4**
(task-role `lambda:InvokeFunction` on exactly that ARN). There is **no
shared secret anywhere in the mint path**: IAM authenticates the caller,
a KMS asymmetric key signs the token, and the connector verifies with the
public key.

## The live-ness cross-check (why the auth server mints)

The mint Lambda refuses to sign unless the caller is *live*, which is the
property that justified auth-server minting over gateway self-signing:

| Binding | Subject (`subject_id`) | Cross-check | Property bound |
|---|---|---|---|
| `owui-session` | `X-OpenWebUI-User-Id` — the enrollment-space `user_id` (= `email_hmac`) | an unexpired VPN session exists for the subject (`tpai-vpn-sessions`, `user_id-index`) | **live login** |
| `api-key` | `X-TPAI-ApiKey-User` — the `tpai-api-keys` per-user id, **case-preserved** (it is an exact DynamoDB key; only the HMAC input is lowercased) | a `tpai-api-keys` record exists for the subject and is not revoked (any present, non-`false` `revoked` marker counts as revoked — fail closed on marker-type drift) | **live credential** (R12) |

> **Record for the S09 THREAT_MODEL rewrite:** OWUI identities are bound to
> a *live login* (an active VPN session — walk away, session expires, tools
> stop minting). api-proxy identities have no OWUI session, so their binding
> is a *live credential*: the key's continued existence/non-revocation in
> `tpai-api-keys` at mint time. Revoking an api-key therefore cuts off
> connector tool access within one token TTL (≤15 min), which is the
> revocation story the THREAT_MODEL §data-flows update must state.
> (`tpai-api-keys` has no revocation attribute today; the Lambda enforces
> the contract `revoked == true → refuse` from day one, so revocation
> tooling only has to set that attribute.)

## Identity spaces (E2)

The request deliberately carries **both pseudonym spaces**:

- `identity` — the gateway's audit-space HMAC (`api.identity`, dedicated E2
  key). This becomes the JWT `sub`, and is what the connector logs/keys on.
- `subject_id` — the enrollment-space pseudonym that keys the session and
  api-key tables. It never leaves the mint request: not in the JWT, not in
  gateway or Lambda logs.

The pairing of the two spaces exists only in this request and the Lambda's
memory. Logging them together anywhere would create the linkage E2 exists
to prevent; `tests/test_mint.py::test_token_and_subject_never_logged`
asserts the gateway side, and the Lambda logs only
`identity.slice(0, 8)` + outcome.

## Wire contract (`tpai.connector-mint.request.v1`)

Request (Lambda `RequestResponse` payload):

```json
{
  "schema": "tpai.connector-mint.request.v1",
  "identity": "<64-hex audit-space HMAC>",
  "binding": "owui-session | api-key",
  "subject_id": "<enrollment-space pseudonym>"
}
```

Success / refusal:

```json
{ "ok": true, "token": "<RS256 JWT>", "expires_at": 1751500000 }
{ "ok": false, "reason": "no_active_session | unknown_api_key | revoked_api_key | invalid_request" }
```

JWT claims: `sub` (identity), `iss=tpai-auth`, `aud=tpai-connector`, `iat`,
`exp` (= iat + 900s default), `jti`, `tpai_binding` (`session` semantics
above). The 15-minute TTL is the same D8 window `api.taint`'s session-keyed
fallback is pinned to — keep them aligned.

## Error contract (mirrors `api.audit` / `api.taint`)

| Condition | Raised | Caller must |
|---|---|---|
| ARN unset, transport failure, Lambda `FunctionError`, malformed response | `MintError` | fail the tool call (5xx-class; **fail closed**) |
| Lambda refusal (`ok: false`) or missing `subject_id` | `MintRefusedError` (`.reason`) | fail the tool call as a clean deny (the audit record's `policy_reason`) |
| Malformed identity/binding arguments | `ValueError` | programming bug — fix the call site |

Refusals are never cached; a user can log in and retry immediately.
Successful tokens are cached per identity until `exp − 30s`, and a cache
hit additionally requires the presented `subject_id` to equal the one the
token's live-ness check ran against — a changed subject re-mints.
(`invalidate(identity)` drops one entry, e.g. after a connector 401.)

## m2 integration (the choke point)

`get_connector_token` is called from the same **single audited choke point**
`docs/TaintAndBudgets.md` mandates — never scattered per tool. Ordered
sequence per classified server-side tool call:

1. `taint.resolve_taint_scope` / advisory `filter_tools`
2. `budget.check_and_consume` (clean deny → audit `denied`)
3. `taint.record_tool_use`
4. `mint.get_connector_token(request.state.tpai_identity,
   request.state.tpai_mint_binding, request.state.tpai_mint_subject_id)`
   — `MintRefusedError` → audit `denied` with its `.reason`;
   `MintError` → audit `error`
5. `audit.emit_audit_record` + execute against the connector with
   `Authorization: Bearer <token>`

All of these are synchronous boto3 calls — the async Converse loop must
wrap each in `starlette.concurrency.run_in_threadpool`, exactly as
`models/bedrock.py` wraps Bedrock calls.

## Configuration

| Env var | Source | Meaning |
|---|---|---|
| `TPAI_CONNECTOR_MINT_FUNCTION_ARN` | bedrock-gateway-stack (deterministic ARN) | unset ⇒ dark; set ⇒ mint path armed (still no call sites until m2) |
| `TPAI_CONNECTOR_MINT_ENDPOINT_URL` | bedrock-gateway-stack, from the network stack's endpoint DNS output | explicit `https://vpce-…` address of the mint interface endpoint. The endpoint has **no private DNS** (it would capture all in-VPC `lambda.<region>` resolution — including VPN operators' CLI — under its deny-all-but-mint policy), so this client must address it explicitly. Unset ⇒ default resolver ⇒ unroutable in the airgapped VPC ⇒ fail closed |
| `AWS_REGION` | task definition | client region |
