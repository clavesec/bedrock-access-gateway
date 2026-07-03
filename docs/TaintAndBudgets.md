# Trifecta taint rule + tool-call budgets

TPAI external-content program, m1 Phase D (decisions D9, R12; plan:
`plans/external-content/m1-substrate/plan.md`). Modules: `api/taint.py`,
`api/budget.py`. **Dark until m2 Phase 0** — no server-side tool executes
today; this document is the binding integration contract for the m2/m3
Converse-loop tool executor.

## The rule

Server-side tools are classified:

| class | tools | risk |
|---|---|---|
| `inbound-private` | `gmail_search`, `gmail_get_message` | private data enters the context window |
| `outbound-fetch` | `web_fetch` | attacker content enters; requests leave (exfil channel) |

The first **executed** use of either class removes the other class for the
remainder of the taint scope — sticky, deterministic, no per-call prompting.
Client-declared tools (anything OWUI or an API caller sends in
`chat_request.tools`) are outside the system: never classified, never
filtered, never recorded, never budgeted.

## Taint scope

| context | scope key | lifetime |
|---|---|---|
| Chat turn with `X-OpenWebUI-Chat-Id` | `chat#{identity}#{chat_id}` | `TPAI_TAINT_CONVERSATION_TTL_DAYS` (default 90d), refreshed on use |
| No chat id (background generations, api-proxy callers) | `session#{identity}` | sliding 15 min (`SESSION_TAINT_TTL_SECONDS`), refreshed on use |

The identity component is the pseudonymous HMAC from `api.identity`
(`request.state.tpai_identity`); the chat id is captured and validated by the
`capture_conversation` router dependency (`request.state.tpai_chat_id`).
Chat ids are client-asserted — embedding the identity in the key means a
forged chat id can only ever partition the caller's *own* taint/budget state,
and partitioning is bounded by the user-day budget.

## State

Two DynamoDB tables, injected by the bedrock-gateway CDK stack:

- `TPAI_TAINT_TABLE` — item `{pk: <scope>, tainted_class, first_tool, ttl}`
- `TPAI_BUDGET_TABLE` — item `{pk: <counter>, n, ttl}` with counters
  `day#{identity}#{YYYYMMDD}` and `scope#{scope}`

DynamoDB, not process memory: multiple gateway tasks run behind the ALB and
taint must be sticky across all of them. TTL (`ttl`, epoch seconds) is
hygiene, not semantics — an expired-but-unreaped taint item still counts as
tainted.

## Budgets

Consumed **before** every server-side tool execution, both windows, keyed by
HMAC identity (api-proxy identities included from v1 — R12):

| window | counter | limit (env, default) |
|---|---|---|
| user-day | all tool calls per identity per UTC day | `TPAI_BUDGET_USER_DAILY_LIMIT`, 512 |
| scope | all tool calls per taint scope | `TPAI_BUDGET_SCOPE_LIMIT`, 128 |

Atomic conditional increments (`ADD n 1` guarded by `n < limit`) — no
overshoot under concurrency. User-day is consumed first; a scope-denied call
burns one user-day unit (bounded, accepted — no rollback writes).

## Executor contract (m2 Phase 0 MUST follow this order)

```
scope   = taint.resolve_taint_scope(identity, chat_id)

# 1. Advisory filter at toolConfig build: keep blocked tools out of the
#    model's sight. Never touches client tools (per the loop invariant,
#    server tools are only injected when the client declared none).
allowed, dropped = taint.filter_tools(taint.get_taint(scope), server_tools)

# 2. Before executing a classified tool call:
budget.check_and_consume(identity, scope)   # BudgetExceededError → deny
taint.record_tool_use(scope, tool_name)     # TaintConflictError  → deny
audit.emit_audit_record(...)                # AuditEmitError      → fail
# 3. Execute the tool.
```

`record_tool_use` — not the filter — is the enforcement point: the
conditional write is the mutual exclusion, so a race between two concurrent
first uses of different classes has exactly one winner.

Error semantics:

| exception | meaning | audit `policy_decision` / `policy_reason` | surfaced as |
|---|---|---|---|
| `TaintConflictError` | trifecta rule blocked the class | `deny` / `taint-blocked` | clean toolResult error |
| `BudgetExceededError` | window exhausted | `deny` / `budget-exhausted` | clean toolResult error |
| `TaintStoreError`, `BudgetStoreError` | state unreadable/unwritable | failed tool call | error (fail closed) |

All three modules share the fail-closed posture of `api.audit`: an
unconfigured table refuses classified-tool execution rather than silently
skipping the control.
