# Gmail server-side tools (`gmail_search` / `gmail_get_message`)

TPAI external-content program, m3 G3. Companions: `docs/WebFetch.md` (the
m2 tool and loop invariants — all of which apply here), `docs/
TaintAndBudgets.md` (the executor contract), `docs/ConnectorMint.md` (the
per-user JWT), `docs/AuditRecords.md` (the WORM schema). The authoritative
consumption contract for the metadata layer is the S15 overview in the
main repo: `plans/external-content/m3-gmail/s15-metadata-layer-overview.md`.

## Flags (all default off — dark)

| Env var | Default | Meaning |
|---|---|---|
| `ENABLE_GMAIL_TOOLS` | `false` | Master switch for the tool pair AND the `/connectors/gmail` passthrough surface |
| `GMAIL_MODELS_ALL` | `false` | Offer to every tool-capable model, not just Claude |
| `GMAIL_SEARCH_MAX_RESULTS` | `20` | Index records per search |
| `GMAIL_MAX_CHARS` | `50000` | Typed-output char cap per call |
| `GMAIL_CONNECTOR_TIMEOUT_S` | `30` | Read timeout for connector gmail calls |
| `GMAIL_CONNECTOR_MAX_BYTES` | 2 MiB | Byte cap on connector gmail responses |
| `GMAIL_INDEX_MAX_BYTES` | 1 MiB | Byte cap on the S3 index object |
| `GMAIL_DEK_CACHE_TTL_S` | `120` | Per-identity DEK cache TTL — **must stay ≤ minutes** (S15 R-5: the metadata-key 403 is the revocation signal) |
| `TPAI_GMAIL_METADATA_BUCKET` | unset | The S15 metadata bucket; unset keeps `gmail_search` fail-closed |

Shared with web_fetch: `TPAI_CONNECTOR_URL`, `TPAI_CONNECTOR_CA_B64` (one
pinned session — same CA trust and SAN assertion), the mint configuration,
and `WEB_FETCH_MAX_ITERATIONS` / `WEB_FETCH_STREAM_STATUS` /
`WEB_FETCH_PROMPT_CACHE`, which govern the whole server-tool loop.

## The two halves (red-team Alt 2 / R4)

- **`gmail_search`** never calls Gmail. It searches the S15 async metadata
  index (headers + subjects only): DEK from `GET /v1/gmail/metadata-key`
  (JWT; 403 = not-connected/revoked, honored immediately and cached at most
  `GMAIL_DEK_CACHE_TTL_S`), S3 `GetObject` in-account, local AES-256-GCM
  decrypt with AAD `tpai.gmail-metadata-index.v1|<identity>`, in-memory
  term filtering. The connector's response `bucket`/`object_key` are
  **verified against locally derived values** — a compromised connector
  cannot steer the gateway's read path — and the object key is always
  derived from the requesting identity, never from wire input.
- **`gmail_get_message`** is the query-time body fetch: one message via the
  connector's `POST /v1/gmail/get`, which sanitizes (contract in the
  superseded gmail plan) and runs the mandatory Haiku-class quarantine pass
  (R3). The gateway adds byte/char caps and R7 fencing on top.

Both ride the m1 loop unchanged: same executor choke point, same taint
classification (`inbound-private` — a gmail turn drops `web_fetch` from
`toolConfig` and vice versa, D9), same budgets, same one-WORM-record-per-
call rule (`target` = `query:<text>` or the message id, R11), same mint
(D8) with the once-only re-mint on 401.

## `/api/v1/connectors/gmail/*` (passthrough for OWUI Settings → Connectors, R14)

`status` / `consent-session` / `confirm` / `disconnect` — authenticated
like chat (API key + forwarded identity), minted per request, relayed only
for the {200, 201, 403} status set. These manage the connection and return
no mailbox content, so taint/budgets don't apply; the connector WORM-audits
the state-changing calls. `confirm` closes S14 residual R-1: the connector
finalizes a consent only when the confirming JWT subject equals the
consent session's bound identity (see the connector's docs) — the gateway
relays the browser-supplied nonce opaquely and never treats it as
authorization.
