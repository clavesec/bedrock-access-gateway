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
`capture_conversation` router dependency (`request.state.tpai_chat_id` —
only routes declaring that dependency are guaranteed the attribute; read it
defensively via `getattr(request.state, "tpai_chat_id", None)` anywhere
else). Chat ids are client-asserted — embedding the identity in the key
means a forged chat id can only ever partition the caller's *own*
taint/budget state, and partitioning is bounded by the user-day budget.
(Threat model: the taint rule defends against **injected content in the
conversation**, which cannot set transport headers. A caller who rotates
their own scope keys can already exfiltrate anything they can read with
their own code.)

## State

Two DynamoDB tables, injected by the bedrock-gateway CDK stack:

- `TPAI_TAINT_TABLE` — item `{pk: <scope>, tainted_class, ttl}`
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

Per `docs/AuditRecords.md`, tool execution must route through a **single
choke point** — one function every tool-dispatch branch calls — because
per-branch control calls are how a branch gets missed. The taint/budget
sequence below is part of that same choke point, not code to scatter across
call sites.

```
scope   = taint.resolve_taint_scope(identity, chat_id)
#         identity = request.state.tpai_identity (never absent here: tool
#         execution is only reachable with identity enforcement on);
#         chat_id = the VALIDATED request.state.tpai_chat_id — the same
#         value must feed audit's conversation_id, so the audit record
#         always correlates with the scope that governed the call.

# 1. Advisory filter at toolConfig build: keep blocked tools out of the
#    model's sight. Never touches client tools (per the loop invariant,
#    server tools are only injected when the client declared none), and
#    the injected list is built from the gateway's own registry
#    (TOOL_CLASSES keys gated by feature flags) — NEVER from tool names
#    appearing in the request.
allowed, dropped = taint.filter_tools(taint.get_taint(scope), server_tools)

# 2. Before executing a classified tool call, in order:
#    a. If the tool is in `dropped` (the model called a tool the filter
#       removed — possible via stale context), deny WITHOUT consuming
#       budget: the taint outcome is already known, no store round-trip.
#    b. budget.check_and_consume(identity, scope)   # BudgetExceededError → deny
#    c. taint.record_tool_use(scope, tool_name)     # TaintConflictError  → deny
#    d. audit.emit_audit_record(...)                # AuditEmitError      → fail
# 3. Execute the tool.
```

All of these are blocking boto3 calls — from the async Converse loop, wrap
each in `starlette.concurrency.run_in_threadpool`, exactly as `api.audit`'s
docstring mandates and `models/bedrock.py` already does.

`record_tool_use` — not the filter — is the enforcement point: the
conditional write is the mutual exclusion, so a race between two concurrent
first uses of different classes has exactly one winner. It also refuses
(`ValueError`) any tool name not in `TOOL_CLASSES`: an unregistered server
tool is a registration bug and fails closed at the enforcement point rather
than silently escaping the trifecta rule.

Error semantics:

| exception | meaning | audit `policy_decision` / `policy_reason` | surfaced as |
|---|---|---|---|
| `TaintConflictError` | trifecta rule blocked the class | `deny` / `taint-blocked` | clean toolResult error |
| `BudgetExceededError` | window exhausted | `deny` / `budget-exhausted` | clean toolResult error |
| `TaintStoreError`, `BudgetStoreError` | state unreadable/unwritable | failed tool call | error (fail closed) |
| `ValueError` | contract violation (unclassified tool, missing identity/scope) | failed tool call | error (fail closed — treat as fatal, it is a bug) |

All three modules share the fail-closed posture of `api.audit`: an
unconfigured table refuses classified-tool execution rather than silently
skipping the control.

Budget accounting caveats (accepted, documented in `api/budget.py`): the
user-day counter is consumed before the scope counter, so a scope-denied
call burns one user-day unit; step 2a removes the taint-deny burn except
under a genuine write race; session-scope counters are caps per sliding
15-minute window (the durable backstop is user-day); a retried DynamoDB
`ADD` can over-count by one — the limit invariant holds, erring toward
denying sooner.
