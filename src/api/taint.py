"""Trifecta taint rule (TPAI external-content program, m1 Phase D; decision D9).

The lethal trifecta — private data (gmail_*), attacker-controlled content
(web_fetch results), and an exfiltration channel (web_fetch requests) — is
assembled only when both tool classes are usable in the same context window.
The rule that prevents it: the first executed use of either class **removes
the other class for the remainder of the taint scope** (sticky, deterministic,
no per-call prompting).

Tool classes (server-side tools only — tools the gateway itself injects and
executes in the Converse loop; client-declared tools are never classified,
never filtered, never recorded):

    inbound-private   gmail_search, gmail_get_message   (m3)
    outbound-fetch    web_fetch                          (m2)

The separation between the two tool worlds is positional, not nominal: the
m2 executor must build its injection list from the gateway's own registry
(``TOOL_CLASSES`` keys gated by their feature flags) and must never derive
it from names appearing in the request — a client tool that happens to be
named ``web_fetch`` is still a client tool. ``record_tool_use`` backstops
this: it refuses (ValueError, fail closed) any tool name it cannot classify.

Taint scope (D9, resolved 2026-07-02):

- **Conversation-keyed** via ``X-OpenWebUI-Chat-Id`` (forwarded by the OWUI
  fork's cherry-pick of upstream ``671f577``, gated by
  ``ENABLE_FORWARD_USER_INFO_HEADERS`` — enabled since the S02 flip).
  The scope key includes the HMAC identity, so one user's asserted chat id
  can never read or set another user's taint state.
- **Session-keyed fallback** where a chat id is legitimately absent
  (background generations such as title/tags, and api-proxy callers — R12):
  per HMAC identity with a sliding 15-minute window matching the D8
  connector-JWT TTL. Sliding (refreshed on every tool use), not bucketed:
  a fixed time bucket would reset taint mid-burst at the bucket boundary.

Threat model note: scope keys are partly client-asserted (the chat id), so
an authenticated caller can partition their *own* state across scopes — but
the attacker the taint rule defends against is **injected content inside
the conversation**, which cannot set transport headers. A caller malicious
enough to rotate scope keys can already exfiltrate anything they can read
using their own code; cross-scope rotation is additionally bounded by the
user-day budget (api.budget).

State lives in DynamoDB (``TPAI_TAINT_TABLE``), not in process memory —
multiple gateway tasks run behind the ALB, and taint must be sticky across
all of them. Items carry a TTL for hygiene; conversation taint is deliberately
long-lived (``TPAI_TAINT_CONVERSATION_TTL_DAYS``, default 90 — pinned
explicitly by the CDK env block) because a dormant conversation still holds
its attacker content when resumed.

Enforcement layers (the m2 Converse loop MUST use both — see
``docs/TaintAndBudgets.md`` for the full integration contract):

1. **Filter (advisory):** before building ``toolConfig``, drop the opposing
   class from the injected server-tool list (``filter_tools`` against
   ``get_taint``). This keeps the blocked tool out of the model's sight.
2. **Record (authoritative):** immediately before executing a classified
   tool, call ``record_tool_use``. It atomically sets-or-confirms the
   scope's tainted class with a DynamoDB conditional write; a conflicting
   class raises ``TaintConflictError`` and the executor must fail that tool
   call (audit ``policy_decision=deny``, ``policy_reason=taint-blocked``).
   The conditional write is the mutual exclusion — two concurrent first
   uses of different classes cannot both win.

Operating contract (same posture as ``api.audit``):

- **Fail closed.** Store errors — including an unconfigured table or a
  malformed stored item — raise ``TaintStoreError``; a classified tool call
  whose taint state cannot be read or recorded must not execute.
- **Blocking I/O.** These are synchronous boto3 calls; async callers (the
  m2 Converse loop) must wrap them in
  ``starlette.concurrency.run_in_threadpool`` the same way ``api.audit``
  and ``models/bedrock.py`` do.
- **Dark today.** No server-side tool executes until m2 Phase 0 lands and
  its flag is enabled; nothing calls into the store in production. The
  ``capture_conversation`` dependency (chat-id header capture) is live but
  behaviorally inert — it only stashes a validated header value on
  ``request.state``.
"""

import logging
import os
import re
import time

from fastapi import Request

from api import ddb

logger = logging.getLogger(__name__)

CHAT_ID_HEADER = "X-OpenWebUI-Chat-Id"

INBOUND_PRIVATE = "inbound-private"
OUTBOUND_FETCH = "outbound-fetch"

# The classification registry for server-side tools. m2/m3 sessions extend
# this map when they add tools; anything not listed here is a client tool
# and passes through the gateway untouched. Every class added here MUST get
# a matching _OPPOSING entry — record_tool_use and filter_tools key on it.
TOOL_CLASSES = {
    "web_fetch": OUTBOUND_FETCH,
    "gmail_search": INBOUND_PRIVATE,
    "gmail_get_message": INBOUND_PRIVATE,
}

_OPPOSING = {
    INBOUND_PRIVATE: OUTBOUND_FETCH,
    OUTBOUND_FETCH: INBOUND_PRIVATE,
}

TAINT_TABLE = os.environ.get("TPAI_TAINT_TABLE", "")

# Sliding session-scope window; matches the D8 connector-JWT TTL (15 min).
# Deliberately a constant, not an env var: the window is a property of the
# session model, and letting it drift from the JWT TTL by configuration
# would silently widen or split the taint scope.
SESSION_TAINT_TTL_SECONDS = 900

# Conversation taint outlives the conversation's active use: a dormant chat
# still holds its attacker content when resumed, so keep this generous.
CONVERSATION_TAINT_TTL_DAYS = int(os.environ.get("TPAI_TAINT_CONVERSATION_TTL_DAYS", "90"))

# OWUI chat ids are UUIDs; accept a slightly wider url-safe charset but
# reject anything that could smuggle structure into a DynamoDB key.
_CHAT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class TaintStoreError(RuntimeError):
    """Taint state could not be read or durably recorded. Callers must fail
    the tool call — never execute with unknown taint state."""


class TaintConflictError(RuntimeError):
    """The scope is already tainted with a conflicting tool class; this tool
    call is blocked by the trifecta rule (D9)."""

    def __init__(self, scope: str, tool: str, tainted_class: str):
        self.scope = scope
        self.tool = tool
        self.tainted_class = tainted_class
        super().__init__(
            f"tool {tool!r} blocked: scope is tainted {tainted_class!r}"
        )


def _ddb():
    """Indirection over the shared client so tests can stub this module's
    store without touching the other consumers."""
    return ddb.client()


def classify_tool(tool_name: str) -> str | None:
    """Return the tool's taint class, or None for client tools (which are
    outside the taint system entirely)."""
    return TOOL_CLASSES.get(tool_name)


async def capture_conversation(request: Request) -> None:
    """FastAPI dependency: stash the validated conversation id on
    ``request.state.tpai_chat_id`` (None when absent or malformed).

    Malformed values are treated as absent — the caller then takes the
    session-keyed taint fallback, which is the stricter scope. Consumers
    outside the /chat router must read this defensively
    (``getattr(request.state, "tpai_chat_id", None)``); only routes that
    declare this dependency are guaranteed the attribute.
    """
    raw = (request.headers.get(CHAT_ID_HEADER) or "").strip()
    request.state.tpai_chat_id = raw if _CHAT_ID_RE.match(raw) else None


def resolve_taint_scope(identity: str, chat_id: str | None) -> str:
    """Build the taint-scope key for this request.

    The identity is embedded in both forms: chat ids are client-asserted,
    so without it one caller could poison or observe another's scope.
    """
    if not identity:
        # Tool execution is only reachable behind require_identity with
        # enforcement on; a scope without an identity is a bug.
        raise ValueError("taint scope requires a non-empty HMAC identity")
    if chat_id:
        return f"chat#{identity}#{chat_id}"
    return f"session#{identity}"


def scope_ttl_epoch(scope: str, now: float) -> int:
    """TTL epoch for state keyed by this scope.

    Public because a scope's lifetime is a property of the scope itself:
    api.budget derives its scope-counter TTL from here too, so a taint item
    and its budget counter can never expire on different schedules.
    """
    if scope.startswith("session#"):
        return int(now) + SESSION_TAINT_TTL_SECONDS
    return int(now) + CONVERSATION_TAINT_TTL_DAYS * 86400


def get_taint(scope: str) -> str | None:
    """Read the scope's tainted class (strongly consistent), or None.

    An expired-but-not-yet-reaped item (DynamoDB TTL lags) still counts as
    tainted — TTL here is hygiene, not semantics; erring sticky is safe.
    """
    if not TAINT_TABLE:
        raise TaintStoreError(
            "TPAI_TAINT_TABLE is not configured - refusing to evaluate taint state"
        )
    try:
        resp = _ddb().get_item(
            TableName=TAINT_TABLE,
            Key={"pk": {"S": scope}},
            ConsistentRead=True,
        )
    except Exception as exc:
        logger.error("taint read failed (%s)", type(exc).__name__)
        raise TaintStoreError("failed to read taint state") from exc
    item = resp.get("Item")
    if not item:
        return None
    try:
        return item["tainted_class"]["S"]
    except (KeyError, TypeError) as exc:
        # Only reachable via an out-of-band write (the task role can only
        # GetItem/UpdateItem, and record_tool_use always SETs the class) —
        # still, unknown state must fail closed, not leak a KeyError.
        logger.error("malformed taint item for scope (missing tainted_class)")
        raise TaintStoreError("malformed taint item") from exc


def _stored_class_from(exc: Exception) -> str | None:
    """The pre-existing tainted_class carried on a conditional-check failure
    (ReturnValuesOnConditionCheckFailure=ALL_OLD), if present."""
    response = getattr(exc, "response", None) or {}
    item = response.get("Item") or {}
    tainted = item.get("tainted_class") or {}
    return tainted.get("S")


def record_tool_use(scope: str, tool_name: str) -> str:
    """Authoritatively taint the scope with this tool's class; return it.

    Atomic set-or-confirm: succeeds when the scope is untainted or already
    tainted with the same class (refreshing the sliding TTL); raises
    ``TaintConflictError`` when a conflicting class holds the scope. Call
    this immediately before executing every classified tool — the
    conditional write, not the advisory filter, is the enforcement point.
    """
    tool_class = classify_tool(tool_name)
    if tool_class is None:
        raise ValueError(
            f"tool {tool_name!r} is not a classified server-side tool - "
            "client tools must never enter the taint system"
        )
    if not TAINT_TABLE:
        raise TaintStoreError(
            "TPAI_TAINT_TABLE is not configured - refusing to execute a classified tool"
        )
    try:
        _ddb().update_item(
            TableName=TAINT_TABLE,
            Key={"pk": {"S": scope}},
            UpdateExpression="SET tainted_class = :cls, #ttl = :ttl",
            ConditionExpression="attribute_not_exists(tainted_class) OR tainted_class = :cls",
            ExpressionAttributeNames={"#ttl": "ttl"},
            ExpressionAttributeValues={
                ":cls": {"S": tool_class},
                ":ttl": {"N": str(scope_ttl_epoch(scope, time.time()))},
            },
            # On refusal, carry the stored item back so the error names the
            # class that actually holds the scope — never inferred, so it
            # stays correct if a third tool class ever exists.
            ReturnValuesOnConditionCheckFailure="ALL_OLD",
        )
    except Exception as exc:
        if ddb.is_conditional_check_failure(exc):
            stored = _stored_class_from(exc) or _OPPOSING[tool_class]
            raise TaintConflictError(scope, tool_name, stored)
        logger.error("taint write failed (%s)", type(exc).__name__)
        raise TaintStoreError("failed to record tool use") from exc
    return tool_class


def filter_tools(tainted_class: str | None, tool_names: list[str]) -> tuple[list[str], list[str]]:
    """Advisory filter for the server-tool injection list: given the scope's
    taint state, return ``(allowed, dropped)`` — ``dropped`` is for the m2
    loop's status/stream reporting.

    Pure function — pass it ``get_taint(scope)``. Unclassified names pass
    through (they are not governed by the trifecta rule), but the injection
    list should only ever contain classified server tools. An unknown
    *taint* class (out-of-band write, or a newer writer this build doesn't
    know) fails toward fewer capabilities: every classified tool is dropped.
    """
    if tainted_class is None:
        return list(tool_names), []
    if tainted_class in _OPPOSING:
        blocked = {_OPPOSING[tainted_class]}
    else:
        blocked = set(_OPPOSING)
    allowed: list[str] = []
    dropped: list[str] = []
    for name in tool_names:
        (dropped if classify_tool(name) in blocked else allowed).append(name)
    return allowed, dropped
